import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis
from core import config


to_tensor = transforms.ToTensor()

normalize = transforms.Normalize(
    mean=config.NORMALIZE_MEAN,
    std=config.NORMALIZE_STD,
)


_face_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    """Initialise InsightFace once and reuse across every call."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL_NAME,
            allowed_modules=config.INSIGHTFACE_MODULES,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(
            ctx_id=0 if config.DEVICE == "cuda" else -1,
            det_size=(config.IMG_SIZE, config.IMG_SIZE),
        )
    return _face_app


def extract_face(
    pil_image: Image.Image, target_size: int = config.IMG_SIZE
) -> torch.Tensor:
    """
    Detect and crop the largest face from a PIL image.

    Falls back to a plain centre-resize of the full image when no
    face is found.

    Parameters
    ----------
    pil_image   : PIL.Image.Image  — input image (any mode; converted to RGB internally)
    target_size : int              — output spatial size (default from config)

    Returns
    -------
    torch.Tensor  shape (3, H, W), dtype float32, values in [0, 1]
    """
    face_app = get_face_app()
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)

    if faces:
        # Largest face by bounding-box area
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb).resize((target_size, target_size))
    else:
        print("[WARNING] No face detected — using full image as fallback.")
        crop_pil = pil_image.resize((target_size, target_size))

    return to_tensor(crop_pil)


def preprocess(image_input) -> torch.Tensor:
    """
    End-to-end preprocessing: raw image  →  model-ready tensor.

    Parameters
    ----------
    image_input : str | os.PathLike | PIL.Image.Image
        File path or an already-opened PIL image.

    Returns
    -------
    torch.Tensor  shape (1, 3, 224, 224), float32, normalised
    """
    # Step 1 — load
    if isinstance(image_input, (str, os.PathLike)):
        pil_image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    else:
        raise TypeError(f"Expected a file path or PIL.Image, got {type(image_input)}")

    # Step 2 — detect & crop face
    face_tensor = extract_face(pil_image)  # (3, H, W)  [0, 1]

    # Step 3 — normalize with ImageNet stats
    face_tensor = normalize(face_tensor)  # (3, H, W)  normalised

    # Step 4 — add batch dimension
    return face_tensor.unsqueeze(0)  # (1, 3, H, W)
