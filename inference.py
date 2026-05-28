from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet34

from core import config

IMG_SIZE = config.IMG_SIZE


@lru_cache(maxsize=1)
def _get_face_app() -> FaceAnalysis:
    app = FaceAnalysis(
        name=config.INSIGHTFACE_MODEL_NAME,
        allowed_modules=config.INSIGHTFACE_MODULES,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(
        ctx_id=0 if config.DEVICE == "cuda" else -1, det_size=(IMG_SIZE, IMG_SIZE)
    )
    return app


_normalize = transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)

mae_transform = transforms.Compose([transforms.ToTensor(), _normalize])


def extract_face_insightface(
    pil_image: Image.Image, target_size: int = IMG_SIZE, *, strict: bool = False
) -> torch.Tensor:
    """Return (1, C, H, W) tensor (CPU) after InsightFace crop + resize + normalize.

    If `strict` is False, falls back to the full image when no face is detected.
    If `strict` is True, raises RuntimeError when no face is detected.
    """

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = _get_face_app().get(bgr)

    if not faces:
        if strict:
            raise RuntimeError("No face detected by insightface")
        crop_pil = pil_image
    else:
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

    crop_pil = crop_pil.resize((target_size, target_size))
    tensor = transforms.ToTensor()(crop_pil)
    tensor = _normalize(tensor)
    return tensor.unsqueeze(0)


class AgeRegressionModelFairFace(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = resnet18(weights=None)

            in_features = 512

        elif backbone_name == "resnet34":
            backbone = resnet34(weights=None)

            in_features = 512

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):

        feats = self.feature_extractor(x)

        feats = torch.flatten(feats, 1)

        age = self.regression_head(feats)

        return age.squeeze(1)


class AgeRegressionModelMAE(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        if backbone_name == "resnet18":
            backbone = resnet18(weights=None)

            in_features = 512

        elif backbone_name == "resnet34":
            backbone = resnet34(weights=None)

            in_features = 512

        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.regression_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):

        feats = self.feature_extractor(x)

        feats = torch.flatten(feats, 1)

        age = self.regression_head(feats)

        return age.squeeze(1)


def load_model_fairface():

    model = AgeRegressionModelFairFace(
        backbone_name="resnet34", hidden_dim=512, dropout=0.4
    )

    ckpt_path = Path(config.CHECKPOINT_PATH_FAIRFACE)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"FairFace checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=config.DEVICE)

    # Support both plain state_dict and wrapped checkpoints
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try non-strict load for compatibility
        model.load_state_dict(state_dict, strict=False)

    model.to(config.DEVICE)

    model.eval()

    print(f"[OK] Loaded FairFace checkpoint: '{ckpt_path}' (device={config.DEVICE})")

    return model


def load_model_mae():

    model = AgeRegressionModelMAE(
        backbone_name=config.BACKBONE_MAE,
        hidden_dim=config.HIDDEN_DIM_MAE,
        dropout=config.DROPOUT_MAE,
    )

    ckpt_path = Path(config.CHECKPOINT_PATH_MAE)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=config.DEVICE)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    model.to(config.DEVICE)

    model.eval()

    print(f"[OK] Loaded MAE checkpoint: '{ckpt_path}' (device={config.DEVICE})")

    return model


def predict_age_fairface(model, image_path: str) -> float:

    image = Image.open(image_path).convert("RGB")

    tensor = extract_face_insightface(image)

    tensor = tensor.to(config.DEVICE)

    with torch.no_grad():
        age = model(tensor).item()

    return float(age)


def predict_age_mae(
    model,
    image_path: str,
    use_face_crop: bool = True,
) -> float:
    image = Image.open(image_path).convert("RGB")

    if use_face_crop:
        # Use strict insightface-based crop: require a detection. If detection
        # fails we do NOT fall back to a center-crop (per request); instead
        # return NaN to indicate no valid face-driven prediction.
        try:
            tensor = extract_face_insightface(
                image, target_size=config.IMG_SIZE, strict=True
            )
        except Exception as e:
            print(
                f"[warn] MAE face-crop requested but no face detected for {image_path}: {e}"
            )
            return float("nan")

    else:
        # Full-image path (kept for completeness). Prefer face-crop for accuracy.
        resized = image.resize((config.IMG_SIZE, config.IMG_SIZE))
        tensor = mae_transform(resized).unsqueeze(0)

    tensor = tensor.to(config.DEVICE)

    with torch.no_grad():
        age = model(tensor).item()

    return float(age)
