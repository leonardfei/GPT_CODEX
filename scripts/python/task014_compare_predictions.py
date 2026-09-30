#!/usr/bin/env python3
"""Descriptive, label-free comparison of two WSI prediction distributions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CLASSES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]


def read_qc(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data["n_detected"] != data["n_classified"] or set(data["class_counts"]) != set(CLASSES):
        raise ValueError(f"Unexpected counts/classes in {path}")
    if sum(data["class_counts"].values()) != data["n_classified"]:
        raise ValueError(f"Class counts do not sum to total in {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p84-qc", required=True, type=Path)
    ap.add_argument("--p169-qc", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    p84, p169 = read_qc(args.p84_qc), read_qc(args.p169_qc)
    rows = []
    for label in CLASSES:
        a, b = p84["class_counts"][label], p169["class_counts"][label]
        pa, pb = 100 * a / p84["n_classified"], 100 * b / p169["n_classified"]
        rows.append({"class_name": label, "p84_count": a, "p84_percent": f"{pa:.6f}",
                     "p169_count": b, "p169_percent": f"{pb:.6f}",
                     "p169_minus_p84_percentage_points": f"{pb - pa:+.6f}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote descriptive comparison of {len(rows)} classes to {args.output}")


if __name__ == "__main__":
    main()
