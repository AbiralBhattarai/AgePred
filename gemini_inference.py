"""Gemini age inference helper.

Loads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and calls the Google GenAI SDK to
estimate age from a face photograph.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
from pathlib import Path

_GEMINI_MODEL_PRIMARY = "gemini-3.1-flash-lite-preview"  # may be retired
_GEMINI_MODEL_FALLBACK = "gemini-3.1-flash-lite"

_ENV_LOADED = False


def _load_env() -> None:
    """Load .env from the project directory (same folder as this file)."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # `python-dotenv` not installed; rely on process env vars.
        _ENV_LOADED = True
        return

    project_dir = Path(__file__).resolve().parent
    dotenv_path = project_dir / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)
    _ENV_LOADED = True


def _get_api_key() -> str | None:
    """Return the API key from env.

    Supports both `GEMINI_API_KEY` and `GOOGLE_API_KEY` (common convention).
    """

    _load_env()

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    key = key.strip()
    return key or None


def build_age_prompt() -> str:
    # Keep this prompt stable and machine-parseable.
    return (
        "You are estimating a person's age from a face photograph. "
        "Return ONLY valid JSON, no markdown, no extra text. "
        'Schema: {"age_years": <number>, "confidence": <number 0..1>}. '
        "age_years must be a realistic human age in years. "
        "If the image does not contain a clear single human face, return "
        '{"age_years": null, "confidence": 0}.'
    )


def _guess_mime_type(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    return mime or "image/jpeg"


def _parse_age_from_text(text: str) -> float | None:
    # Prefer JSON.
    try:
        obj = json.loads(text)
        age = obj.get("age_years")
        if age is None:
            return None
        return float(age)
    except Exception:
        pass

    # Fallback: extract first number.
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    return float(m.group(1))


def predict_age_gemini(image_path: str) -> float:
    """Predict age (years) using Gemini.

    Returns NaN if the API key is missing, request fails, or response is unparseable.
    """

    api_key = _get_api_key()
    if not api_key:
        return float("nan")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bytes = path.read_bytes()
    mime_type = _guess_mime_type(path)

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        # SDK not installed; treat Gemini as unavailable.
        return float("nan")

    client = genai.Client(api_key=api_key)

    prompt = build_age_prompt()

    debug = os.getenv("GEMINI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    resp = None
    for model_name in (_GEMINI_MODEL_PRIMARY, _GEMINI_MODEL_FALLBACK):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=image_bytes, mime_type=mime_type
                            ),
                        ],
                    )
                ],
            )
            break
        except Exception as e:
            msg = str(e)
            is_not_found = (
                ("404" in msg) or ("NOT_FOUND" in msg) or ("no longer available" in msg)
            )
            if is_not_found:
                if debug:
                    print(f"[gemini] model not available: {model_name}")
                continue
            if debug:
                print(
                    f"[gemini] request failed ({model_name}): {type(e).__name__}: {e}"
                )
            return float("nan")

    if resp is None:
        return float("nan")

    text = getattr(resp, "text", None)
    if not isinstance(text, str) or not text.strip():
        return float("nan")

    age = _parse_age_from_text(text)
    if age is None:
        return float("nan")

    return float(age)


async def predict_age_gemini_async(image_path: str) -> float:
    """Predict age (years) using Gemini asynchronously.

    Returns NaN if the API key is missing, request fails, or response is unparseable.
    """

    api_key = _get_api_key()
    if not api_key:
        return float("nan")

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_bytes = path.read_bytes()
    mime_type = _guess_mime_type(path)

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        return float("nan")

    client = genai.Client(api_key=api_key)
    prompt = build_age_prompt()
    debug = os.getenv("GEMINI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    resp = None
    for model_name in (_GEMINI_MODEL_PRIMARY, _GEMINI_MODEL_FALLBACK):
        try:
            # We use client.aio for asynchronous calls
            resp = await client.aio.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=image_bytes, mime_type=mime_type
                            ),
                        ],
                    )
                ],
            )
            break
        except Exception as e:
            msg = str(e)
            is_not_found = (
                ("404" in msg) or ("NOT_FOUND" in msg) or ("no longer available" in msg)
            )
            if is_not_found:
                if debug:
                    print(f"[gemini] model not available: {model_name}")
                continue
            if debug:
                print(
                    f"[gemini] request failed ({model_name}): {type(e).__name__}: {e}"
                )
            return float("nan")

    if resp is None:
        return float("nan")

    text = getattr(resp, "text", None)
    if not isinstance(text, str) or not text.strip():
        return float("nan")

    age = _parse_age_from_text(text)
    if age is None:
        return float("nan")

    return float(age)
