from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import torchvision
    import torchvision.models as tv_models
    import torchvision.transforms as T
    from torchvision.datasets import FakeData, ImageFolder

    TORCHVISION_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment-specific fallback
    torchvision = None
    tv_models = None
    T = None
    FakeData = None
    ImageFolder = None
    TORCHVISION_AVAILABLE = False
    _TORCHVISION_IMPORT_ERROR = exc

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# Configuration and reproducibility
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Configuration shared by training, evaluation, comparison, and inference."""

    data_path: str = "./pneumoniamnist.npz"
    output_dir: str = "./outputs_med_agcnet"
    model_name: str = "med_agcnet_full"

    image_size: int = 224
    in_channels: int = 3
    num_classes: int = 2

    batch_size: int = 32
    epochs: int = 20
    comparison_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0

    seed: int = 42
    deterministic: bool = True
    device: str = "auto"  # auto | cpu | cuda | cuda:0 | mps
    use_amp: bool = True

    # Explicit imbalance handling avoids accidentally applying two corrections.
    imbalance_strategy: str = "weighted_loss"  # none | weighted_loss | weighted_sampler | both

    tune_threshold: bool = True
    decision_threshold: float = 0.5
    threshold_objective: str = "balanced_accuracy"  # balanced_accuracy | f1
    monitor_metric: str = "balanced_accuracy"  # balanced_accuracy | accuracy | val_loss

    early_stopping_patience: int = 0  # 0 disables early stopping
    min_delta: float = 0.0
    grad_clip_norm: float = 0.0  # 0 disables clipping

    use_fake_data: bool = False
    fake_train_size: int = 128
    fake_val_size: int = 32
    fake_test_size: int = 32

    pretrained_baselines: bool = False
    save_every_checkpoint: bool = False

    def resolved_device(self) -> str:
        requested = str(self.device).lower().strip()
        if requested != "auto":
            if requested.startswith("cuda") and not torch.cuda.is_available():
                warnings.warn("CUDA requested but unavailable; falling back to CPU.")
                return "cpu"
            if requested == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                warnings.warn("MPS requested but unavailable; falling back to CPU.")
                return "cpu"
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.comparison_epochs <= 0:
            raise ValueError("comparison_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0 < self.decision_threshold < 1:
            raise ValueError("decision_threshold must be between 0 and 1")
        if self.imbalance_strategy not in {"none", "weighted_loss", "weighted_sampler", "both"}:
            raise ValueError("imbalance_strategy must be one of: none, weighted_loss, weighted_sampler, both")
        if self.threshold_objective not in {"balanced_accuracy", "f1"}:
            raise ValueError("threshold_objective must be balanced_accuracy or f1")
        if self.monitor_metric not in {"balanced_accuracy", "accuracy", "val_loss"}:
            raise ValueError("monitor_metric must be balanced_accuracy, accuracy, or val_loss")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch and optionally enforce deterministic CUDA behavior."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(data: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def collect_environment(device: str) -> Dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": getattr(torchvision, "__version__", None) if TORCHVISION_AVAILABLE else None,
        "numpy": np.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class GlobalContextAttention(nn.Module):
    """Lightweight global-context attention for long-range dependency modeling."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.attn_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.LayerNorm([hidden, 1, 1]),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mask = self.attn_mask(x).view(b, 1, h * w)
        mask = F.softmax(mask, dim=-1)
        features = x.view(b, c, h * w)
        context = torch.bmm(features, mask.transpose(1, 2)).view(b, c, 1, 1)
        weights = self.transform(context)
        return x * weights


class AdaptiveGlobalContextBlock(nn.Module):
    """Ablation-ready Adaptive Global Context Block (AGCB)."""

    def __init__(
        self,
        channels: int,
        large_kernel: int = 7,
        use_global_context: bool = True,
        use_large_rf: bool = True,
        use_adaptive_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.use_global_context = use_global_context
        self.use_large_rf = use_large_rf
        self.use_adaptive_fusion = use_adaptive_fusion

        self.local_branch = ConvBNAct(channels, channels, kernel_size=3, groups=channels)
        self.large_rf_branch = (
            ConvBNAct(channels, channels, kernel_size=large_kernel, groups=channels)
            if use_large_rf
            else None
        )
        self.global_branch = GlobalContextAttention(channels) if use_global_context else None

        branch_count = 1 + int(use_large_rf) + int(use_global_context)
        self.branch_count = branch_count
        self.pointwise = ConvBNAct(channels, channels, kernel_size=1)

        if use_adaptive_fusion:
            hidden = max(channels // 4, 16)
            self.fusion_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden, branch_count * channels, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.fusion_gate = None

        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        branches: List[torch.Tensor] = [self.local_branch(x)]

        if self.large_rf_branch is not None:
            branches.append(self.large_rf_branch(x))
        if self.global_branch is not None:
            branches.append(self.global_branch(x))

        if self.fusion_gate is not None:
            gates = self.fusion_gate(x)
            gate_chunks = torch.chunk(gates, chunks=len(branches), dim=1)
            fused = sum(g * b for g, b in zip(gate_chunks, branches))
        else:
            fused = sum(branches) / float(len(branches))

        fused = self.pointwise(fused)
        return self.norm(fused + residual)


class MedAGCNet(nn.Module):
    """Med-AGCNet medical image classifier."""

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        widths: Tuple[int, int, int, int] = (32, 64, 128, 256),
        blocks_per_stage: Tuple[int, int, int, int] = (1, 2, 2, 1),
        use_global_context: bool = True,
        use_large_rf: bool = True,
        use_adaptive_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, widths[0], kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages: List[nn.Module] = []
        in_ch = widths[0]
        for stage_idx, out_ch in enumerate(widths):
            if stage_idx > 0:
                stages.append(ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2))
            for _ in range(blocks_per_stage[stage_idx]):
                stages.append(
                    AdaptiveGlobalContextBlock(
                        channels=out_ch,
                        large_kernel=7 if stage_idx < 2 else 9,
                        use_global_context=use_global_context,
                        use_large_rf=use_large_rf,
                        use_adaptive_fusion=use_adaptive_fusion,
                    )
                )
            in_ch = out_ch

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(widths[-1]),
            nn.Dropout(0.2),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN baseline."""

    def __init__(self, num_classes: int = 2, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 2),
            ConvBNAct(32, 64, 3, 2),
            ConvBNAct(64, 128, 3, 2),
            ConvBNAct(128, 256, 3, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


MODEL_NAMES = [
    "simplecnn",
    "resnet18",
    "efficientnet_b0",
    "med_agcnet_full",
    "med_agcnet_no_global",
    "med_agcnet_no_large_rf",
    "med_agcnet_no_fusion",
    "med_agcnet_local_only",
]


def build_model(
    model_name: str,
    num_classes: int,
    in_channels: int = 3,
    pretrained_baselines: bool = False,
) -> nn.Module:
    name = model_name.lower().strip()

    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    if name in {"resnet18", "efficientnet_b0"}:
        if not TORCHVISION_AVAILABLE or tv_models is None:
            raise RuntimeError(
                f"{name} requires torchvision. Original import error: "
                f"{globals().get('_TORCHVISION_IMPORT_ERROR', 'unknown')}"
            )
        if in_channels != 3:
            raise ValueError(f"{name} currently expects 3-channel input")

    if name == "resnet18":
        if pretrained_baselines:
            weights = getattr(tv_models, "ResNet18_Weights", None)
            model = tv_models.resnet18(weights=weights.DEFAULT if weights else "DEFAULT")
        else:
            model = tv_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        if pretrained_baselines:
            weights = getattr(tv_models, "EfficientNet_B0_Weights", None)
            model = tv_models.efficientnet_b0(weights=weights.DEFAULT if weights else "DEFAULT")
        else:
            model = tv_models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    kwargs = dict(num_classes=num_classes, in_channels=in_channels)
    if name == "med_agcnet_full":
        return MedAGCNet(**kwargs)
    if name == "med_agcnet_no_global":
        return MedAGCNet(**kwargs, use_global_context=False)
    if name == "med_agcnet_no_large_rf":
        return MedAGCNet(**kwargs, use_large_rf=False)
    if name == "med_agcnet_no_fusion":
        return MedAGCNet(**kwargs, use_adaptive_fusion=False)
    if name == "med_agcnet_local_only":
        return MedAGCNet(
            **kwargs,
            use_global_context=False,
            use_large_rf=False,
            use_adaptive_fusion=False,
        )

    raise ValueError(f"Unknown model_name={model_name!r}. Valid names: {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Grad-CAM
# -----------------------------------------------------------------------------


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: List[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
            self.activations = output.detach()

        def backward_hook(
            _module: nn.Module,
            _grad_input: Tuple[Any, ...],
            grad_output: Tuple[torch.Tensor, ...],
        ) -> None:
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, image: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(image)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    if isinstance(model, MedAGCNet):
        return model.features[-1]
    if isinstance(model, SimpleCNN):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Could not determine a Grad-CAM target layer")


def create_heatmap_overlay(
    original: Image.Image,
    cam: np.ndarray,
    image_size: int,
    alpha: float = 0.45,
) -> Image.Image:
    original_rgb = original.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    original_np = np.asarray(original_rgb, dtype=np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap = cmap(np.clip(cam, 0.0, 1.0))[..., :3].astype(np.float32)
    overlay = np.clip((1.0 - alpha) * original_np + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


# -----------------------------------------------------------------------------
# Transforms and datasets
# -----------------------------------------------------------------------------


class ComposeFallback:
    def __init__(self, transforms: Sequence[Callable[[Any], Any]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: Any) -> Any:
        for transform in self.transforms:
            image = transform(image)
        return image


class ResizeFallback:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, Image.BILINEAR)


class RandomHorizontalFlipFallback:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.mirror(image) if random.random() < self.p else image


class RandomRotationFallback:
    def __init__(self, degrees: float = 10.0) -> None:
        self.degrees = degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.rotate(random.uniform(-self.degrees, self.degrees))


class ColorJitterFallback:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image


class ToTensorFallback:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


class NormalizeFallback:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def build_transforms(image_size: int) -> Tuple[Any, Any]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if TORCHVISION_AVAILABLE and T is not None:
        train_tfms = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        eval_tfms = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )
        return train_tfms, eval_tfms

    train_tfms = ComposeFallback(
        [
            ResizeFallback((image_size, image_size)),
            RandomHorizontalFlipFallback(0.5),
            RandomRotationFallback(10),
            ColorJitterFallback(0.1, 0.1),
            ToTensorFallback(),
            NormalizeFallback(mean, std),
        ]
    )
    eval_tfms = ComposeFallback(
        [
            ResizeFallback((image_size, image_size)),
            ToTensorFallback(),
            NormalizeFallback(mean, std),
        ]
    )
    return train_tfms, eval_tfms


class SimpleFakeData(Dataset):
    """Deterministic synthetic data used only for software smoke tests."""

    def __init__(
        self,
        size: int,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 2,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        seed: int = 12345,
    ) -> None:
        self.size = int(size)
        self.image_size = image_size
        self.num_classes = int(num_classes)
        self.transform = transform
        self.seed = int(seed)
        self.labels = np.arange(self.size, dtype=np.int64) % self.num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        _c, h, w = self.image_size
        rng = np.random.default_rng(self.seed + int(index))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        image = Image.fromarray(arr, mode="RGB")
        target = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class PneumoniaMNISTNPZDataset(Dataset):
    """Adapter for the official PneumoniaMNIST NPZ split arrays."""

    def __init__(self, npz_path: str | Path, split: str, transform: Optional[Callable] = None) -> None:
        self.npz_path = Path(npz_path)
        self.split = split.lower().strip()
        self.transform = transform

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {self.npz_path.resolve()}")

        with np.load(self.npz_path) as data:
            image_key = f"{self.split}_images"
            label_key = f"{self.split}_labels"
            missing = [k for k in (image_key, label_key) if k not in data.files]
            if missing:
                raise KeyError(
                    f"Missing keys in {self.npz_path.name}: {missing}. Expected train/val/test images and labels."
                )
            self.images = np.asarray(data[image_key])
            self.labels = np.asarray(data[label_key]).reshape(-1).astype(np.int64)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Split {self.split}: {len(self.images)} images but {len(self.labels)} labels"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        image = self.images[index]
        if image.ndim == 2:
            pil_image = Image.fromarray(image.astype(np.uint8), mode="L").convert("RGB")
        elif image.ndim == 3 and image.shape[-1] == 1:
            pil_image = Image.fromarray(image.squeeze(-1).astype(np.uint8), mode="L").convert("RGB")
        elif image.ndim == 3 and image.shape[-1] == 3:
            pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, int(self.labels[index])


def is_npz_dataset_path(data_path: str | Path) -> bool:
    return str(data_path).lower().endswith(".npz")


def validate_npz_dataset(npz_path: str | Path) -> Tuple[bool, str, List[str]]:
    path = Path(npz_path)
    if not path.exists():
        return False, f"NPZ file does not exist: {path.resolve()}", []

    try:
        with np.load(path) as data:
            required = [
                "train_images",
                "val_images",
                "test_images",
                "train_labels",
                "val_labels",
                "test_labels",
            ]
            missing = [k for k in required if k not in data.files]
            if missing:
                return False, f"Missing NPZ keys: {missing}", []

            lines = ["NPZ dataset structure is valid."]
            label_values: set[int] = set()
            for split in ("train", "val", "test"):
                images = np.asarray(data[f"{split}_images"])
                labels = np.asarray(data[f"{split}_labels"]).reshape(-1)
                if len(images) != len(labels):
                    return False, f"Mismatch in {split}: {len(images)} images vs {len(labels)} labels", []
                label_values.update(labels.astype(int).tolist())
                lines.append(
                    f"{split}: images={tuple(images.shape)}, labels={tuple(labels.shape)}, dtype={images.dtype}"
                )

        sorted_labels = sorted(label_values)
        if sorted_labels == [0, 1]:
            class_names = ["normal", "pneumonia"]
        else:
            if sorted_labels != list(range(len(sorted_labels))):
                return False, f"Labels must be contiguous integers from 0. Found: {sorted_labels}", []
            class_names = [f"class_{i}" for i in sorted_labels]
        lines.append(f"classes={class_names}")
        return True, "\n".join(lines), class_names
    except Exception as exc:
        return False, f"Could not read NPZ dataset: {exc}", []


def validate_folder_dataset(data_path: str | Path) -> Tuple[bool, str, List[str]]:
    root = Path(data_path)
    if not root.exists():
        return False, f"Dataset root does not exist: {root.resolve()}", []

    split_classes: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            return False, f"Missing folder: {split_dir}", []
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if not classes:
            return False, f"No class folders found in {split_dir}", []
        split_classes[split] = classes

    if not (split_classes["train"] == split_classes["val"] == split_classes["test"]):
        return False, f"Class folders differ across splits: {split_classes}", []

    return True, "Folder dataset structure is valid.", split_classes["train"]


def validate_dataset(cfg: ExperimentConfig) -> Tuple[bool, str, List[str]]:
    if cfg.use_fake_data:
        return True, "FakeData mode enabled (software validation only).", [
            f"class_{i}" for i in range(cfg.num_classes)
        ]
    if is_npz_dataset_path(cfg.data_path):
        return validate_npz_dataset(cfg.data_path)
    return validate_folder_dataset(cfg.data_path)


def extract_labels(dataset: Dataset) -> Optional[np.ndarray]:
    """Extract labels without repeatedly decoding/augmenting images when possible."""

    for attr in ("labels", "targets"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if torch.is_tensor(values):
                values = values.detach().cpu().numpy()
            return np.asarray(values, dtype=np.int64).reshape(-1)
    return None


def build_datasets(cfg: ExperimentConfig) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    train_tfms, eval_tfms = build_transforms(cfg.image_size)

    if cfg.use_fake_data:
        class_names = [f"class_{i}" for i in range(cfg.num_classes)]
        # Custom FakeData exposes deterministic labels, making sampling reproducible.
        train_ds = SimpleFakeData(
            cfg.fake_train_size,
            image_size=(3, cfg.image_size, cfg.image_size),
            num_classes=cfg.num_classes,
            transform=train_tfms,
            seed=cfg.seed + 1000,
        )
        val_ds = SimpleFakeData(
            cfg.fake_val_size,
            image_size=(3, cfg.image_size, cfg.image_size),
            num_classes=cfg.num_classes,
            transform=eval_tfms,
            seed=cfg.seed + 2000,
        )
        test_ds = SimpleFakeData(
            cfg.fake_test_size,
            image_size=(3, cfg.image_size, cfg.image_size),
            num_classes=cfg.num_classes,
            transform=eval_tfms,
            seed=cfg.seed + 3000,
        )
        return train_ds, val_ds, test_ds, class_names

    valid, message, class_names = validate_dataset(cfg)
    if not valid:
        raise FileNotFoundError(message)

    if is_npz_dataset_path(cfg.data_path):
        train_ds = PneumoniaMNISTNPZDataset(cfg.data_path, "train", train_tfms)
        val_ds = PneumoniaMNISTNPZDataset(cfg.data_path, "val", eval_tfms)
        test_ds = PneumoniaMNISTNPZDataset(cfg.data_path, "test", eval_tfms)
        return train_ds, val_ds, test_ds, class_names

    if not TORCHVISION_AVAILABLE or ImageFolder is None:
        raise RuntimeError("Folder dataset mode requires torchvision")
    root = Path(cfg.data_path)
    train_ds = ImageFolder(root / "train", transform=train_tfms)
    val_ds = ImageFolder(root / "val", transform=eval_tfms)
    test_ds = ImageFolder(root / "test", transform=eval_tfms)
    return train_ds, val_ds, test_ds, list(train_ds.classes)


def build_dataloaders(
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_ds, val_ds, test_ds, class_names = build_datasets(cfg)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    sampler = None
    shuffle = True
    if cfg.imbalance_strategy in {"weighted_sampler", "both"}:
        labels = extract_labels(train_ds)
        if labels is None:
            warnings.warn("Could not extract labels efficiently; weighted sampler disabled.")
        else:
            class_counts = np.bincount(labels, minlength=len(class_names)).astype(np.float64)
            class_counts[class_counts == 0] = 1.0
            sample_weights = 1.0 / class_counts[labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )
            shuffle = False

    pin_memory = cfg.resolved_device().startswith("cuda")
    persistent_workers = cfg.num_workers > 0
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker if cfg.num_workers > 0 else None,
        generator=generator,
    )

    train_loader = DataLoader(train_ds, shuffle=shuffle, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_names


def compute_class_weights(dataset: Dataset, num_classes: int, device: str) -> Optional[torch.Tensor]:
    labels = extract_labels(dataset)
    if labels is None:
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# -----------------------------------------------------------------------------
# Training/evaluation metrics
# -----------------------------------------------------------------------------


def _amp_context(device: str, enabled: bool):
    use_amp = bool(enabled and device.startswith("cuda"))
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=use_amp)
    return torch.cuda.amp.autocast(enabled=use_amp)


def _make_grad_scaler(device: str, enabled: bool):
    use_amp = bool(enabled and device.startswith("cuda"))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Any,
    use_amp: bool,
    grad_clip_norm: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    preds: List[int] = []
    targets_all: List[int] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _amp_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets_all.extend(targets.detach().cpu().tolist())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    accuracy = accuracy_score(targets_all, preds)
    return float(mean_loss), float(accuracy)


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probs_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _amp_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        probs = F.softmax(logits, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        probs_all.append(probs.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

    mean_loss = total_loss / max(len(loader.dataset), 1)
    return float(mean_loss), np.concatenate(probs_all), np.concatenate(targets_all)


def tune_binary_threshold(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    objective: str = "balanced_accuracy",
) -> Tuple[float, float]:
    best_threshold = 0.5
    best_score = -math.inf
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (prob_positive >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_true, pred, zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12
            and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def predictions_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if probs.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape={probs.shape}")
    if probs.shape[1] == 2:
        return (probs[:, 1] >= float(threshold)).astype(np.int64)
    return probs.argmax(axis=1).astype(np.int64)


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    preds = predictions_from_probs(probs, threshold)
    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "weighted_precision": float(precision_score(y_true, preds, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, preds, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, preds)),
    }

    if len(class_names) == 2:
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn + 1e-12))
        specificity = float(tn / (tn + fp + 1e-12))
        metrics.update(
            {
                "sensitivity_positive_recall": sensitivity,
                "specificity_negative_recall": specificity,
                "false_positive_count": int(fp),
                "false_negative_count": int(fn),
            }
        )
        if [str(x).lower() for x in class_names] == ["normal", "pneumonia"]:
            metrics.update(
                {
                    "sensitivity_pneumonia_recall": sensitivity,
                    "specificity_normal_recall": specificity,
                    "false_positive_normal_as_pneumonia": int(fp),
                    "false_negative_pneumonia_as_normal": int(fn),
                }
            )
        prob_positive = probs[:, 1]
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_positive))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, prob_positive))
        except ValueError:
            metrics["pr_auc"] = None

    return metrics


def selection_score(
    cfg: ExperimentConfig,
    val_loss: float,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> float:
    preds = predictions_from_probs(probs, threshold)
    if cfg.monitor_metric == "val_loss":
        return -float(val_loss)
    if cfg.monitor_metric == "accuracy":
        return float(accuracy_score(y_true, preds))
    return float(balanced_accuracy_score(y_true, preds))


# -----------------------------------------------------------------------------
# Plotting/export helpers
# -----------------------------------------------------------------------------


def save_history_csv(history: Dict[str, List[float]], path: Path) -> None:
    ensure_dir(path.parent)
    keys = list(history.keys())
    rows = zip(*(history[k] for k in keys))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)


def save_training_curves(history: Dict[str, List[float]], output_dir: Path) -> None:
    ensure_dir(output_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.plot(epochs, history["val_balanced_accuracy"], label="Validation Balanced Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training and Validation Performance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=220)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_binary_curves(y_true: np.ndarray, probs: np.ndarray, output_dir: Path) -> None:
    if probs.shape[1] != 2:
        return
    prob_positive = probs[:, 1]

    try:
        fpr, tpr, _ = roc_curve(y_true, prob_positive)
        auc = roc_auc_score(y_true, prob_positive)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=220)
        plt.close()
    except ValueError:
        pass

    try:
        precision, recall, _ = precision_recall_curve(y_true, prob_positive)
        ap = average_precision_score(y_true, prob_positive)
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision, label=f"PR-AUC/AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curve.png", dpi=220)
        plt.close()
    except ValueError:
        pass


def save_predictions_csv(
    y_true: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    class_names: Sequence[str],
    path: Path,
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "true_index", "true_class", "pred_index", "pred_class"]
            + [f"prob_{name}" for name in class_names]
        )
        for idx, (yt, yp, prob_row) in enumerate(zip(y_true, preds, probs)):
            writer.writerow(
                [idx, int(yt), class_names[int(yt)], int(yp), class_names[int(yp)]]
                + [float(x) for x in prob_row]
            )


def save_comparison_table(results: List[Dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    json_dump(results, json_path)
    if not results:
        return
    keys = sorted({key for row in results for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def save_comparison_chart(
    results: Sequence[Dict[str, Any]],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    names = [str(r["model"]) for r in results]
    values = [float(r.get(metric, 0.0) or 0.0) for r in results]
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Experiment API
# -----------------------------------------------------------------------------


class MedAGCNetExperiment:
    """Notebook-friendly experiment runner."""

    def __init__(self, cfg: Optional[ExperimentConfig] = None) -> None:
        self.cfg = cfg or ExperimentConfig()
        self.cfg.validate()
        self.device = self.cfg.resolved_device()
        self.output_dir = ensure_dir(self.cfg.output_dir)
        set_seed(self.cfg.seed, self.cfg.deterministic)
        self.environment = collect_environment(self.device)
        json_dump(asdict(self.cfg), self.output_dir / "config.json")
        json_dump(self.environment, self.output_dir / "environment.json")

    def log(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def validate_data(self) -> Tuple[bool, str, List[str]]:
        result = validate_dataset(self.cfg)
        valid, message, classes = result
        print(message)
        if classes:
            print("Classes:", classes)
        return result

    def _make_criterion(self, train_dataset: Dataset, num_classes: int) -> nn.Module:
        if self.cfg.imbalance_strategy in {"weighted_loss", "both"}:
            weights = compute_class_weights(train_dataset, num_classes, self.device)
            if weights is None:
                warnings.warn("Class labels unavailable; using unweighted CrossEntropyLoss.")
                return nn.CrossEntropyLoss()
            self.log(f"Class weights: {weights.detach().cpu().numpy().round(4).tolist()}")
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    def fit(
        self,
        model_name: Optional[str] = None,
        epochs: Optional[int] = None,
        run_dir: Optional[str | Path] = None,
        save_checkpoint: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train one model, select the best validation checkpoint, and evaluate on test."""

        model_name = (model_name or self.cfg.model_name).lower()
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model_name}")
        epochs = int(epochs or self.cfg.epochs)
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        set_seed(self.cfg.seed, self.cfg.deterministic)
        train_loader, val_loader, test_loader, class_names = build_dataloaders(self.cfg)
        num_classes = len(class_names)

        target_dir = ensure_dir(
            run_dir
            if run_dir is not None
            else self.output_dir / "runs" / model_name
        )
        json_dump(asdict(self.cfg), target_dir / "config.json")

        model = build_model(
            model_name,
            num_classes=num_classes,
            in_channels=self.cfg.in_channels,
            pretrained_baselines=self.cfg.pretrained_baselines,
        ).to(self.device)
        criterion = self._make_criterion(train_loader.dataset, num_classes)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = _make_grad_scaler(self.device, self.cfg.use_amp)

        history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_balanced_accuracy": [],
            "val_threshold": [],
        }

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_epoch = 0
        best_threshold = float(self.cfg.decision_threshold)
        best_selection = -math.inf
        best_val_metrics: Dict[str, Any] = {}
        epochs_without_improvement = 0
        start_time = time.perf_counter()

        self.log(
            f"Training {model_name} | device={self.device} | params={count_parameters(model):,} | epochs={epochs}"
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                self.device,
                scaler,
                self.cfg.use_amp,
                self.cfg.grad_clip_norm,
            )
            val_loss, val_probs, val_targets = predict_loader(
                model, val_loader, criterion, self.device, self.cfg.use_amp
            )

            threshold = float(self.cfg.decision_threshold)
            if self.cfg.tune_threshold and num_classes == 2:
                threshold, _ = tune_binary_threshold(
                    val_targets,
                    val_probs[:, 1],
                    objective=self.cfg.threshold_objective,
                )

            val_preds = predictions_from_probs(val_probs, threshold)
            val_acc = float(accuracy_score(val_targets, val_preds))
            val_bal_acc = float(balanced_accuracy_score(val_targets, val_preds))
            score = selection_score(self.cfg, val_loss, val_targets, val_probs, threshold)
            scheduler.step()

            history["epoch"].append(float(epoch))
            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(float(train_acc))
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_acc))
            history["val_balanced_accuracy"].append(float(val_bal_acc))
            history["val_threshold"].append(float(threshold))

            if verbose:
                self.log(
                    f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | "
                    f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | val_bal_acc={val_bal_acc:.4f} | thr={threshold:.3f}"
                )

            if score > best_selection + self.cfg.min_delta:
                best_selection = float(score)
                best_epoch = int(epoch)
                best_threshold = float(threshold)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = compute_metrics(val_targets, val_probs, threshold, class_names)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.cfg.save_every_checkpoint:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model_name,
                        "class_names": class_names,
                        "config": asdict(self.cfg),
                        "epoch": epoch,
                        "decision_threshold": threshold,
                    },
                    target_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                )

            if (
                self.cfg.early_stopping_patience > 0
                and epochs_without_improvement >= self.cfg.early_stopping_patience
            ):
                self.log(f"Early stopping at epoch {epoch}.")
                break

        if best_state is None:
            raise RuntimeError("Training ended without a valid checkpoint")

        model.load_state_dict(best_state)
        model.to(self.device).eval()
        runtime_seconds = time.perf_counter() - start_time

        test_loss, test_probs, test_targets = predict_loader(
            model, test_loader, criterion, self.device, self.cfg.use_amp
        )
        test_preds = predictions_from_probs(test_probs, best_threshold)
        test_metrics = compute_metrics(test_targets, test_probs, best_threshold, class_names)
        test_metrics.update(
            {
                "model": model_name,
                "test_loss": float(test_loss),
                "best_epoch": int(best_epoch),
                "best_validation_selection_score": float(best_selection),
                "best_validation_metrics": best_val_metrics,
                "parameters": int(count_parameters(model)),
                "runtime_seconds": float(runtime_seconds),
                "classes": list(class_names),
                "fake_data_warning": (
                    "Synthetic data: results validate software only and are not scientific evidence."
                    if self.cfg.use_fake_data
                    else "Real dataset mode."
                ),
            }
        )

        save_history_csv(history, target_dir / "history.csv")
        json_dump(history, target_dir / "history.json")
        save_training_curves(history, target_dir)
        save_confusion_matrix(test_targets, test_preds, class_names, target_dir / "confusion_matrix.png")
        save_binary_curves(test_targets, test_probs, target_dir)
        save_predictions_csv(
            test_targets,
            test_probs,
            test_preds,
            class_names,
            target_dir / "test_predictions.csv",
        )

        report = classification_report(
            test_targets,
            test_preds,
            labels=list(range(num_classes)),
            target_names=list(class_names),
            zero_division=0,
        )
        (target_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        json_dump(test_metrics, target_dir / "metrics.json")

        checkpoint_path = target_dir / "best_model.pth"
        if save_checkpoint:
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_name": model_name,
                    "class_names": list(class_names),
                    "config": asdict(self.cfg),
                    "best_epoch": best_epoch,
                    "best_validation_selection_score": best_selection,
                    "decision_threshold": best_threshold,
                    "metrics": test_metrics,
                },
                checkpoint_path,
            )
            test_metrics["checkpoint_path"] = str(checkpoint_path.resolve())
            json_dump(test_metrics, target_dir / "metrics.json")

        self.log(
            f"Completed {model_name} | test_acc={test_metrics['accuracy']:.4f} | "
            f"balanced_acc={test_metrics['balanced_accuracy']:.4f} | "
            f"weighted_f1={test_metrics['weighted_f1']:.4f}"
        )
        return test_metrics

    def train(self) -> Dict[str, Any]:
        return self.fit(self.cfg.model_name, self.cfg.epochs)

    def load_checkpoint(
        self, checkpoint_path: str | Path
    ) -> Tuple[nn.Module, List[str], float, Dict[str, Any]]:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Expected a dictionary checkpoint")

        class_names = list(checkpoint.get("class_names", ["class_0", "class_1"]))
        model_name = str(checkpoint.get("model_name", self.cfg.model_name))
        checkpoint_cfg = checkpoint.get("config", {})
        in_channels = int(checkpoint_cfg.get("in_channels", self.cfg.in_channels))
        model = build_model(
            model_name,
            num_classes=len(class_names),
            in_channels=in_channels,
            pretrained_baselines=False,
        ).to(self.device)
        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise KeyError("Checkpoint does not contain model_state_dict")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        threshold = float(checkpoint.get("decision_threshold", self.cfg.decision_threshold))
        return model, class_names, threshold, checkpoint

    def evaluate_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        set_seed(self.cfg.seed, self.cfg.deterministic)
        model, class_names, threshold, checkpoint = self.load_checkpoint(checkpoint_path)
        _, _, test_loader, dataset_classes = build_dataloaders(self.cfg)
        if list(dataset_classes) != list(class_names):
            raise ValueError(
                f"Checkpoint classes {class_names} do not match dataset classes {dataset_classes}"
            )
        criterion = nn.CrossEntropyLoss()
        test_loss, probs, targets = predict_loader(
            model, test_loader, criterion, self.device, self.cfg.use_amp
        )
        preds = predictions_from_probs(probs, threshold)
        metrics = compute_metrics(targets, probs, threshold, class_names)
        metrics.update(
            {
                "model": checkpoint.get("model_name", self.cfg.model_name),
                "test_loss": float(test_loss),
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
            }
        )

        out_dir = ensure_dir(self.output_dir / "evaluation")
        json_dump(metrics, out_dir / "metrics.json")
        save_confusion_matrix(targets, preds, class_names, out_dir / "confusion_matrix.png")
        save_binary_curves(targets, probs, out_dir)
        save_predictions_csv(targets, probs, preds, class_names, out_dir / "test_predictions.csv")
        report = classification_report(
            targets,
            preds,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
        (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        return metrics

    def run_baseline_comparison(self) -> List[Dict[str, Any]]:
        models = ["simplecnn", "resnet18", "efficientnet_b0", "med_agcnet_full"]
        return self._run_comparison("baseline", models)

    def run_ablation_study(self) -> List[Dict[str, Any]]:
        models = [
            "med_agcnet_full",
            "med_agcnet_no_global",
            "med_agcnet_no_large_rf",
            "med_agcnet_no_fusion",
            "med_agcnet_local_only",
        ]
        return self._run_comparison("ablation", models)

    def _run_comparison(self, name: str, model_names: Sequence[str]) -> List[Dict[str, Any]]:
        comparison_dir = ensure_dir(self.output_dir / "comparisons" / name)
        results: List[Dict[str, Any]] = []

        self.log(f"Starting {name} study: {list(model_names)}")
        for model_name in model_names:
            # Fresh seed + loader per model gives a cleaner, more reproducible comparison.
            set_seed(self.cfg.seed, self.cfg.deterministic)
            run_dir = comparison_dir / model_name
            metrics = self.fit(
                model_name=model_name,
                epochs=self.cfg.comparison_epochs,
                run_dir=run_dir,
                save_checkpoint=False,
                verbose=True,
            )
            compact = {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "weighted_precision": metrics.get("weighted_precision"),
                "weighted_recall": metrics.get("weighted_recall"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_f1": metrics.get("macro_f1"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "parameters": metrics.get("parameters"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "threshold": metrics.get("threshold"),
                "best_epoch": metrics.get("best_epoch"),
            }
            results.append(compact)

        save_comparison_table(
            results,
            comparison_dir / f"{name}_results.csv",
            comparison_dir / f"{name}_results.json",
        )
        save_comparison_chart(
            results,
            "balanced_accuracy",
            f"{name.title()} Study - Balanced Accuracy",
            comparison_dir / "balanced_accuracy.png",
        )
        save_comparison_chart(
            results,
            "weighted_f1",
            f"{name.title()} Study - Weighted F1",
            comparison_dir / "weighted_f1.png",
        )
        self.log(f"{name.title()} study completed.")
        return results

    def predict_image(
        self,
        checkpoint_path: str | Path,
        image_path: str | Path,
        save_gradcam: bool = True,
        show: bool = True,
    ) -> Dict[str, Any]:
        model, class_names, threshold, checkpoint = self.load_checkpoint(checkpoint_path)
        checkpoint_cfg = checkpoint.get("config", {})
        image_size = int(checkpoint_cfg.get("image_size", self.cfg.image_size))
        _, eval_transform = build_transforms(image_size)

        image_path = Path(image_path)
        original = Image.open(image_path).convert("RGB")
        tensor = eval_transform(original).unsqueeze(0).to(self.device)

        start = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        inference_ms = (time.perf_counter() - start) * 1000.0

        if len(class_names) == 2:
            pred_idx = int(probs[1] >= threshold)
        else:
            pred_idx = int(np.argmax(probs))

        result: Dict[str, Any] = {
            "image": str(image_path.resolve()),
            "prediction_index": pred_idx,
            "prediction_class": class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {name: float(prob) for name, prob in zip(class_names, probs)},
            "decision_threshold": float(threshold),
            "inference_ms": float(inference_ms),
            "device": self.device,
            "checkpoint": str(Path(checkpoint_path).resolve()),
        }

        infer_dir = ensure_dir(self.output_dir / "inference")
        stem = image_path.stem
        result_path = infer_dir / f"{stem}_prediction.json"

        if save_gradcam:
            gradcam = GradCAM(model, get_gradcam_target_layer(model))
            try:
                cam = gradcam.generate(tensor, class_idx=pred_idx)
            finally:
                gradcam.close()
            overlay = create_heatmap_overlay(original, cam, image_size=image_size)
            overlay_path = infer_dir / f"{stem}_gradcam.png"
            overlay.save(overlay_path)
            result["gradcam_path"] = str(overlay_path.resolve())

            if show:
                plt.figure(figsize=(6, 6))
                plt.imshow(overlay)
                plt.axis("off")
                plt.title(
                    f"{result['prediction_class']} | confidence={result['confidence']:.3f}"
                )
                plt.tight_layout()
                plt.show()

        json_dump(result, result_path)
        print(json.dumps(result, indent=2))
        return result

    def generate_research_report(self) -> Path:
        """Create a compact Markdown report from the artifacts that currently exist."""

        report_path = self.output_dir / "RESEARCH_REPORT.md"
        lines: List[str] = [
            "# Med-AGCNet Reproducible Research Report",
            "",
            f"Generated: {now_iso()}",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(asdict(self.cfg), indent=2),
            "```",
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(self.environment, indent=2),
            "```",
            "",
            "## Scientific note",
            "",
            (
                "Synthetic FakeData mode was enabled. Numerical results are only software-pipeline checks and must not be used as scientific evidence."
                if self.cfg.use_fake_data
                else "A real dataset mode was selected. Dataset provenance, split policy, and ethics/usage terms should be documented in the paper."
            ),
            "",
            "## Main architecture",
            "",
            "- CNN stem",
            "- Adaptive Global Context Blocks",
            "- Local convolution branch",
            "- Large receptive-field branch",
            "- Lightweight global-context attention branch",
            "- Adaptive fusion gate",
            "",
        ]

        main_metrics = self.output_dir / "runs" / self.cfg.model_name / "metrics.json"
        if main_metrics.exists():
            lines.extend(
                [
                    "## Main model metrics",
                    "",
                    "```json",
                    main_metrics.read_text(encoding="utf-8"),
                    "```",
                    "",
                ]
            )

        for name in ("baseline", "ablation"):
            result_file = self.output_dir / "comparisons" / name / f"{name}_results.json"
            if result_file.exists():
                lines.extend(
                    [
                        f"## {name.title()} results",
                        "",
                        "```json",
                        result_file.read_text(encoding="utf-8"),
                        "```",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Reproducibility / code availability",
                "",
                "Publish this project directory in a version-controlled repository (for example GitHub) and cite the commit or release used for the paper.",
                "For archival reproducibility, consider creating a release and depositing it in Zenodo to obtain a DOI.",
                "",
                "## Intended use",
                "",
                "Research and educational use only. This code is not a diagnostic medical device.",
                "",
            ]
        )

        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.log(f"Research report written to {report_path.resolve()}")
        return report_path

    def run_all(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["main"] = self.train()
        results["baseline"] = self.run_baseline_comparison()
        results["ablation"] = self.run_ablation_study()
        results["report"] = str(self.generate_research_report().resolve())
        return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Med-AGCNet reproducible research pipeline (Colab/Jupyter/CLI compatible)."
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "train", "evaluate", "baseline", "ablation", "infer", "report", "all"],
        default="train",
    )
    parser.add_argument("--data", dest="data_path", default="./pneumoniamnist.npz")
    parser.add_argument("--output", dest="output_dir", default="./outputs_med_agcnet")
    parser.add_argument("--model", dest="model_name", choices=MODEL_NAMES, default="med_agcnet_full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--comparison-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "weighted_loss", "weighted_sampler", "both"],
        default="weighted_loss",
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--pretrained-baselines", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        comparison_epochs=args.comparison_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        imbalance_strategy=args.imbalance_strategy,
        use_fake_data=args.fake_data,
        tune_threshold=not args.no_threshold_tuning,
        use_amp=not args.no_amp,
        deterministic=not args.nondeterministic,
        pretrained_baselines=args.pretrained_baselines,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    experiment = MedAGCNetExperiment(cfg)

    if args.mode == "validate":
        valid, _message, _classes = experiment.validate_data()
        if not valid:
            raise SystemExit(1)
        return

    if args.mode == "train":
        experiment.train()
    elif args.mode == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint is required for --mode evaluate")
        print(json.dumps(experiment.evaluate_checkpoint(args.checkpoint), indent=2))
    elif args.mode == "baseline":
        print(json.dumps(experiment.run_baseline_comparison(), indent=2))
    elif args.mode == "ablation":
        print(json.dumps(experiment.run_ablation_study(), indent=2))
    elif args.mode == "infer":
        if not args.checkpoint or not args.image:
            parser.error("--checkpoint and --image are required for --mode infer")
        experiment.predict_image(args.checkpoint, args.image)
    elif args.mode == "report":
        experiment.generate_research_report()
    elif args.mode == "all":
        print(json.dumps(experiment.run_all(), indent=2))


if __name__ == "__main__":
    main()