"""Benchmark MAE vs FairFace vs Gemini age predictions.

Writes per-image predictions and timing to results/predictions_benchmark.csv
and prints summary metrics (MAE, RMSE, avg latency) for images with labels.

Ground-truth is extracted from filenames by suffix: *_<age> (e.g. person_32.0.jpg)

Usage:
  python benchmark_models.py --data-dir data --results-dir results --mae-face-crop

Notes:
- MAE and FairFace require torch/torchvision/pillow.
- FairFace requires insightface + opencv-python.
- Gemini requires google-genai + python-dotenv and a GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import re
import time
from pathlib import Path

import numpy as np

import inference

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AGE_SUFFIX_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)$")


def collect_images(data_dir: Path) -> list[Path]:
    image_paths = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def extract_actual_age(image_path: Path) -> float | None:
    m = AGE_SUFFIX_PATTERN.search(image_path.stem)
    if not m:
        return None
    return float(m.group(1))


def compute_metrics(preds: list[float], trues: list[float]) -> dict:
    preds_arr = np.array(preds, dtype=float)
    trues_arr = np.array(trues, dtype=float)

    mask = ~np.isnan(trues_arr) & ~np.isnan(preds_arr)
    if mask.sum() == 0:
        return {"count": 0, "mae": float("nan"), "rmse": float("nan")}

    diffs = preds_arr[mask] - trues_arr[mask]
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(math.sqrt(np.mean(diffs**2)))
    return {"count": int(mask.sum()), "mae": mae, "rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark age predictors")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--mae-face-crop",
        action="store_true",
        help="Use strict insightface face-crop for MAE",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(data_dir)
    if not image_paths:
        print(f"No images found under {data_dir}")
        return

    print(f"Found {len(image_paths)} images")

    print("Loading local models...")
    model_mae = inference.load_model_mae()
    model_fairface = inference.load_model_fairface()

    out_csv = results_dir / "predictions_benchmark.csv"

    print("Running Gemini predictions asynchronously...")

    async def run_gemini_batch():
        from gemini_inference import predict_age_gemini_async

        async def fetch(p):
            t0 = time.perf_counter()
            pred = await predict_age_gemini_async(str(p))
            t1 = time.perf_counter()
            return pred, (t1 - t0) * 1000.0

        tasks = [fetch(p) for p in image_paths]
        return await asyncio.gather(*tasks)

    # Calculate effective throughput time per image for the batch
    batch_t0 = time.perf_counter()
    gemini_results = asyncio.run(run_gemini_batch())
    batch_t1 = time.perf_counter()
    effective_gemini_ms = ((batch_t1 - batch_t0) * 1000.0) / len(image_paths)

    mae_preds: list[float] = []
    fair_preds: list[float] = []
    gemini_preds: list[float] = []
    actuals: list[float] = []

    mae_times: list[float] = []
    fair_times: list[float] = []
    gemini_times: list[float] = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "actual",
                "mae_pred",
                "mae_time_ms",
                "fairface_pred",
                "fairface_time_ms",
                "gemini_pred",
                "gemini_time_ms",
            ]
        )

        for i, p in enumerate(image_paths):
            actual = extract_actual_age(p)
            actual_val = actual if actual is not None else float("nan")

            # MAE
            t0 = time.perf_counter()
            try:
                mae_pred = inference.predict_age_mae(
                    model_mae,
                    str(p),
                    use_face_crop=bool(args.mae_face_crop),
                )
            except Exception as e:
                print("MAE failed:", p.name, "->", e)
                mae_pred = float("nan")
            t1 = time.perf_counter()
            mae_time_ms = (t1 - t0) * 1000.0

            # FairFace
            t0 = time.perf_counter()
            try:
                fair_pred = inference.predict_age_fairface(model_fairface, str(p))
            except Exception as e:
                print("FairFace failed:", p.name, "->", e)
                fair_pred = float("nan")
            t1 = time.perf_counter()
            fair_time_ms = (t1 - t0) * 1000.0

            # Gemini (fetched asynchronously)
            gemini_pred, individual_time_ms = gemini_results[i]
            # We'll log the effective throughput time in the CSV, or individual, but effective reflects the speedup.
            gemini_time_ms = effective_gemini_ms

            writer.writerow(
                [
                    p.name,
                    actual,
                    mae_pred,
                    mae_time_ms,
                    fair_pred,
                    fair_time_ms,
                    gemini_pred,
                    gemini_time_ms,
                ]
            )

            mae_preds.append(float(mae_pred))
            fair_preds.append(float(fair_pred))
            gemini_preds.append(float(gemini_pred))
            actuals.append(float(actual_val))

            mae_times.append(mae_time_ms)
            fair_times.append(fair_time_ms)
            gemini_times.append(gemini_time_ms)

    mae_metrics = compute_metrics(mae_preds, actuals)
    fair_metrics = compute_metrics(fair_preds, actuals)
    gemini_metrics = compute_metrics(gemini_preds, actuals)

    print("\nSummary (labeled images only):")
    print(
        f"MAE model:     count={mae_metrics['count']}  MAE={mae_metrics['mae']:.3f}  RMSE={mae_metrics['rmse']:.3f}  avg_ms={np.nanmean(mae_times):.1f}"
    )
    print(
        f"FairFace model: count={fair_metrics['count']}  MAE={fair_metrics['mae']:.3f}  RMSE={fair_metrics['rmse']:.3f}  avg_ms={np.nanmean(fair_times):.1f}"
    )
    print(
        f"Gemini:        count={gemini_metrics['count']}  MAE={gemini_metrics['mae']:.3f}  RMSE={gemini_metrics['rmse']:.3f}  avg_ms={np.nanmean(gemini_times):.1f}"
    )

    print(f"\nSaved per-image results to: {out_csv}")


if __name__ == "__main__":
    main()
