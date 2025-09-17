#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def compute_f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2.0 * precision * recall / denom


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on a dataset and report metrics incl. F1@0.5")
    parser.add_argument("--weights", type=str, default="app/models/yolo_model.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--data", type=str, default="test_set/data.yaml", help="Path to Ultralytics data YAML")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu', '0', '0,1', etc.")
    parser.add_argument("--verbose", action="store_true", help="Print extra details")

    args = parser.parse_args()

    weights_path = Path(args.weights)
    data_path = Path(args.data)

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"[ERROR] Data YAML not found: {data_path}", file=sys.stderr)
        return 1

    model = YOLO(str(weights_path))

    # Run validation. Ultralytics uses IoU thresholds for metrics; we keep defaults (mAP50-95) and report P/R and F1.
    results = model.val(
        data=str(data_path),
        device=args.device,
        verbose=args.verbose,
    )

    results_dict = results.results_dict or {}

    precision = results_dict.get("metrics/precision(B)")
    recall = results_dict.get("metrics/recall(B)")
    map50 = results_dict.get("metrics/mAP50(B)")
    map5095 = results_dict.get("metrics/mAP50-95(B)")
    fitness = results_dict.get("fitness")

    f1 = compute_f1(precision, recall)

    print("=== Validation Metrics ===")
    if precision is not None:
        print(f"Precision@0.5: {precision:.6f}")
    if recall is not None:
        print(f"Recall@0.5:    {recall:.6f}")
    if f1 is not None:
        print(f"F1@0.5:        {f1:.6f}")
    if map50 is not None:
        print(f"mAP@0.5:       {map50:.6f}")
    if map5095 is not None:
        print(f"mAP@0.5:0.95:  {map5095:.6f}")
    if fitness is not None:
        print(f"fitness:        {fitness:.6f}")

    # Speed summary (per image)
    if hasattr(results, "speed") and isinstance(results.speed, dict):
        speed = results.speed
        print("\n=== Speed (per image) ===")
        for k in ("preprocess", "inference", "postprocess"):
            if k in speed:
                print(f"{k}: {speed[k]:.1f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
