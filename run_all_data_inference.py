import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from inference import (
    load_model_mae,
    predict_age_mae,
    load_model_fairface,
    predict_age_fairface,
)

from gemini_inference import predict_age_gemini

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

AGE_SUFFIX_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)$")

IMAGES_PER_PAGE = 10
ROWS = 2
COLS = 5


def extract_actual_age(image_path: Path) -> float | None:
    match = AGE_SUFFIX_PATTERN.search(image_path.stem)
    return float(match.group(1)) if match else None


def collect_images(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_prediction_grids(
    image_paths: list[Path],
    mae_ages: list[float],
    fairface_ages: list[float],
    gemini_ages: list[float],
    actual_ages: list[float | None],
) -> list[Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n = len(image_paths)
    if not (
        n == len(mae_ages) == len(fairface_ages) == len(gemini_ages) == len(actual_ages)
    ):
        raise ValueError("All lists must have same length")

    output_paths: list[Path] = []
    for page_index, start in enumerate(range(0, n, IMAGES_PER_PAGE), start=1):
        end = start + IMAGES_PER_PAGE

        fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 4, ROWS * 4))
        flat_axes = list(axes.flatten())

        page = zip(
            flat_axes,
            image_paths[start:end],
            mae_ages[start:end],
            fairface_ages[start:end],
            gemini_ages[start:end],
            actual_ages[start:end],
            strict=False,
        )

        for ax, image_path, mae_age, fairface_age, gemini_age, actual_age in page:
            image = Image.open(image_path).convert("RGB")
            ax.imshow(image)
            ax.axis("off")

            actual_text = f"{actual_age:.1f} yrs" if actual_age is not None else "N/A"
            gemini_text = "N/A" if math.isnan(gemini_age) else f"{gemini_age:.1f} yrs"

            ax.set_title(
                f"MAE: {mae_age:.1f} yrs\n"
                f"FairFace: {fairface_age:.1f} yrs\n"
                f"Gemini: {gemini_text}\n"
                f"Actual: {actual_text}",
                fontsize=10,
            )

        for ax in flat_axes[min(IMAGES_PER_PAGE, end - start) :]:
            ax.axis("off")

        fig.suptitle(f"Age Predictions - Page {page_index}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = RESULTS_DIR / f"age_predictions_page_{page_index}.jpg"
        fig.savefig(
            output_path, bbox_inches="tight", pad_inches=0.2, dpi=200, format="jpg"
        )
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def main() -> None:
    image_paths = collect_images(DATA_DIR)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {DATA_DIR}")

    model_mae = load_model_mae()
    model_fairface = load_model_fairface()

    mae_ages: list[float] = []
    fairface_ages: list[float] = []
    gemini_ages: list[float] = []
    actual_ages: list[float | None] = []

    for image_path in image_paths:
        print("=" * 60)
        print(f"Name: {image_path.name}")

        predicted_age_mae = predict_age_mae(
            model_mae, str(image_path), use_face_crop=True
        )
        predicted_age_fairface = predict_age_fairface(model_fairface, str(image_path))
        predicted_age_gemini = predict_age_gemini(str(image_path))
        actual_age = extract_actual_age(image_path)

        mae_ages.append(predicted_age_mae)
        fairface_ages.append(predicted_age_fairface)
        gemini_ages.append(predicted_age_gemini)
        actual_ages.append(actual_age)

        print(f"MAE Prediction       : {predicted_age_mae:.1f} yrs")
        print(f"FairFace Prediction  : {predicted_age_fairface:.1f} yrs")
        if math.isnan(predicted_age_gemini):
            print(
                "Gemini Prediction    : N/A (set GEMINI_API_KEY or GOOGLE_API_KEY; install google-genai/python-dotenv if needed)"
            )
        else:
            print(f"Gemini Prediction    : {predicted_age_gemini:.1f} yrs")
        print(
            f"Actual Age           : {'N/A' if actual_age is None else f'{actual_age:.1f} yrs'}"
        )
        print("-" * 40)

    for output_path in save_prediction_grids(
        image_paths, mae_ages, fairface_ages, gemini_ages, actual_ages
    ):
        print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
