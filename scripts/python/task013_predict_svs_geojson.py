#!/usr/bin/env python3
"""CellViT binary WSI detection + Task012 Midnight classification + QuPath GeoJSON.

Run in the server's task010_env. All intermediates and QC stay outside output-dir.
The two GeoJSON filenames are derived from Path(svs).stem.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import openslide
import torch
import torch.nn.functional as F
import ujson

CLASS_NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
COLORS = [[76, 120, 168], [242, 207, 91], [178, 121, 162], [114, 183, 178],
          [255, 157, 166], [84, 162, 75], [228, 87, 86]]
DETECTOR = Path("/data/lf_data/CellViT-plus-plus/checkpoints/CellViT-SAM-H-x40-AMP.pth")
CELLVIT_CLI = Path("/data/lf_data/CellViT-plus-plus/cellvit/detect_cells.py")
CHECKPOINT = Path("/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth")
EXPECTED_SHA = "ddb3d21f9492032b5f0509c5dc28de15a29b0074105991c5cd1168055d84f633"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path: Path) -> dict:
    slide = openslide.OpenSlide(str(path))
    try:
        props = slide.properties
        x = float(props.get("openslide.mpp-x", "nan"))
        y = float(props.get("openslide.mpp-y", "nan"))
        if not (0.2 <= x <= 0.75 and 0.2 <= y <= 0.75):
            raise ValueError(f"Missing or implausible slide MPP: {x}, {y}")
        if abs(x - y) / max(x, y) > 0.02:
            raise ValueError(f"Anisotropic slide MPP needs explicit review: {x}, {y}")
        return {"width": slide.dimensions[0], "height": slide.dimensions[1],
                "levels": slide.level_count, "mpp_x": x, "mpp_y": y,
                "objective_power": props.get("openslide.objective-power"),
                "vendor": props.get("openslide.vendor")}
    finally:
        slide.close()


def detect(args: argparse.Namespace, stem: str, work: Path) -> Path:
    detector_dir = work / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detected = detector_dir / f"{stem}_cells.json"
    if detected.exists():
        print(f"Reusing completed detector output: {detected}", flush=True)
        return detected
    if not DETECTOR.is_file() or not CELLVIT_CLI.is_file():
        raise FileNotFoundError("CellViT checkpoint or CLI absent")
    cmd = [sys.executable, str(CELLVIT_CLI), "--model", str(DETECTOR),
           "--binary", "--gpu", str(args.detector_gpu), "--resolution", "0.25",
           "--batch_size", str(args.detector_batch_size), "--outdir", str(detector_dir),
           "process_wsi", "--wsi_path", str(args.svs)]
    (work / "detector_command.json").write_text(json.dumps(cmd, indent=2))
    log = work / "logs" / "detector.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(cmd, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       cwd=str(CELLVIT_CLI.parent.parent))
    if not detected.is_file():
        raise RuntimeError("CellViT exited without a complete cells.json")
    return detected


def validate_cells(path: Path, meta: dict) -> list[dict]:
    data = ujson.loads(path.read_bytes())
    detector_meta = data.get("wsi_metadata", {})
    if (abs(float(detector_meta.get("base_mpp", -1)) - meta["mpp_x"]) > 1e-6 or
            int(detector_meta.get("patch_size", -1)) != 1024 or
            int(detector_meta.get("patch_overlap", -1)) != 64):
        raise ValueError("Cached CellViT output metadata do not match this slide/recipe")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("No retained CellViT nuclei")
    if set(str(k) for k in data.get("type_map", {}).values()) != {"Background", "Cell"}:
        raise ValueError("Detector was not run in binary-only mode")
    width, height = meta["width"], meta["height"]
    for i, cell in enumerate(cells):
        cx, cy = cell["centroid"]
        if not (math.isfinite(cx) and math.isfinite(cy) and 0 <= cx < width and 0 <= cy < height):
            raise ValueError(f"Out-of-bounds centroid at nucleus {i}: {cx}, {cy}")
        contour = cell["contour"]
        if len(contour) < 3:
            raise ValueError(f"Invalid polygon at nucleus {i}")
        for x, y in contour:
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError(f"Out-of-bounds contour at nucleus {i}: {x}, {y}")
    return cells


def read_tile(slide: openslide.OpenSlide, cells: list[dict], first_index: int,
              pad: int, tile_size: int) -> tuple[np.ndarray, int, int]:
    """Read one padded level-0 tile; reuse it for every classifier batch in that bin."""
    first_x, first_y = cells[first_index]["centroid"]
    tile_x, tile_y = int(first_x) // tile_size, int(first_y) // tile_size
    x0, y0 = tile_x * tile_size - pad, tile_y * tile_size - pad
    side = tile_size + 2 * pad
    rgba = np.asarray(slide.read_region((x0, y0), 0, (side, side)).convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()
    # OpenSlide's transparent out-of-slide pixels should become white H&E background.
    rgb[rgba[:, :, 3] == 0] = 255
    return rgb, x0, y0


def crop_batch(rgb: np.ndarray, x0: int, y0: int, cells: list[dict], indices: list[int],
               meta: dict) -> np.ndarray:
    """Bicubic-resample the exact 12-um centroid grid into a 57x57 RGB crop."""
    offset = (np.arange(57, dtype=np.float32) + 0.5 - 28.5) * (12.0 / 57.0)
    output = []
    for i in indices:
        cx, cy = cells[i]["centroid"]
        xs = np.float32(cx - x0) + offset / meta["mpp_x"]
        ys = np.float32(cy - y0) + offset / meta["mpp_y"]
        map_x, map_y = np.meshgrid(xs, ys)
        output.append(cv2.remap(rgb, map_x, map_y, cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)))
    return np.stack(output)


def classify(cells: list[dict], meta: dict, args: argparse.Namespace, work: Path) -> np.ndarray:
    sys.path.insert(0, str(args.model_code_dir))
    from task012_inference_common import load_model, CLASS_NAMES as MODEL_CLASSES
    if MODEL_CLASSES != CLASS_NAMES:
        raise ValueError("Task012 class order mismatch")
    device = torch.device(f"cuda:{args.classifier_gpu}" if torch.cuda.is_available() else "cpu")
    model, head = load_model(CHECKPOINT, device)
    slide = openslide.OpenSlide(str(args.svs))
    try:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, cell in enumerate(cells):
            x, y = cell["centroid"]
            groups[(int(x) // args.tile_size, int(y) // args.tile_size)].append(i)
        probabilities = np.empty((len(cells), 7), dtype=np.float32)
        pad = math.ceil(max(6 / meta["mpp_x"], 6 / meta["mpp_y"])) + 4
        done = 0
        with torch.inference_mode():
            for group_id, key in enumerate(sorted(groups)):
                indices = groups[key]
                rgb, x0, y0 = read_tile(slide, cells, indices[0], pad, args.tile_size)
                for start in range(0, len(indices), args.classifier_batch_size):
                    selected = indices[start:start + args.classifier_batch_size]
                    native = crop_batch(rgb, x0, y0, cells, selected, meta)
                    x = torch.from_numpy(native).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
                    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
                    x = (x - 0.5) / 0.5
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        hidden = model(x).last_hidden_state
                        z = torch.cat([hidden[:, 0, :], hidden[:, 1:, :].mean(dim=1)], dim=1)
                        logits = head(z)
                    p = torch.softmax(logits, dim=1).float().cpu().numpy()
                    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=2e-3, rtol=0):
                        raise ValueError(f"Invalid classifier probabilities at tile {key}")
                    p /= p.sum(axis=1, keepdims=True)
                    probabilities[selected] = p
                    done += len(selected)
                if (group_id + 1) % 100 == 0:
                    print(f"Classified {done}/{len(cells)} nuclei; tiles {group_id+1}/{len(groups)}", flush=True)
        if done != len(cells):
            raise AssertionError("Not every nucleus was classified")
        return probabilities
    finally:
        slide.close()


def feature(stem: str, class_id: int, geometry_type: str, coordinates: list) -> dict:
    return {"type": "Feature", "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{stem}:{geometry_type}:{class_id}")),
            "geometry": {"type": geometry_type, "coordinates": coordinates},
            "properties": {"objectType": "annotation",
                           "classification": {"name": CLASS_NAMES[class_id], "color": COLORS[class_id]}}}


def export(cells: list[dict], probs: np.ndarray, stem: str, work: Path,
           output_dir: Path) -> dict:
    predictions = probs.argmax(axis=1)
    if probs.shape != (len(cells), 7) or not np.isfinite(probs).all():
        raise ValueError("Prediction count/shape mismatch")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Probabilities are not normalized")
    polygons = [[] for _ in CLASS_NAMES]
    points = [[] for _ in CLASS_NAMES]
    for i, cell in enumerate(cells):
        class_id = int(predictions[i])
        ring = [[float(x), float(y)] for x, y in cell["contour"]]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        polygons[class_id].append([ring])
        points[class_id].append([float(v) for v in cell["centroid"]])
    segmentation = [feature(stem, i, "MultiPolygon", polygons[i]) for i in range(7) if polygons[i]]
    detection = [feature(stem, i, "MultiPoint", points[i]) for i in range(7) if points[i]]
    staging = work / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    staged = {"cells": staging / f"{stem}_cells.geojson",
              "cell_detection": staging / f"{stem}_cell_detection.geojson"}
    staged["cells"].write_text(ujson.dumps(segmentation))
    staged["cell_detection"].write_text(ujson.dumps(detection))
    for name, geometry, expected in [("cells", "MultiPolygon", len(cells)),
                                     ("cell_detection", "MultiPoint", len(cells))]:
        parsed = ujson.loads(staged[name].read_bytes())
        if not isinstance(parsed, list) or any(f["geometry"]["type"] != geometry for f in parsed):
            raise ValueError(f"Invalid {name} GeoJSON structure")
        count = sum(len(f["geometry"]["coordinates"]) for f in parsed)
        names = {f["properties"]["classification"]["name"] for f in parsed}
        if count != expected or names != {CLASS_NAMES[i] for i in range(7) if points[i]}:
            raise ValueError(f"Invalid {name} GeoJSON feature counts/classes")
        for f in parsed:
            cid = CLASS_NAMES.index(f["properties"]["classification"]["name"])
            if f["properties"]["classification"]["color"] != COLORS[cid]:
                raise ValueError("Classification color mismatch")
            if geometry == "MultiPolygon":
                for poly in f["geometry"]["coordinates"]:
                    if len(poly) != 1 or poly[0][0] != poly[0][-1]:
                        raise ValueError("Open/invalid contour ring")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {path.name for path in staged.values()}
    # Allow slide-prefixed GeoJSON outputs from other WSIs to coexist here.
    # Refuse non-GeoJSON sidecars, but never delete previous slide results.
    existing_files = [p for p in output_dir.iterdir() if p.is_file()]
    bad_sidecars = [p.name for p in existing_files if p.suffix.lower() != ".geojson"]
    if bad_sidecars:
        raise RuntimeError(f"Output directory contains non-GeoJSON sidecars; refusing to proceed: {bad_sidecars}")
    for name, source in staged.items():
        os.replace(source, output_dir / source.name)
    final_names = {p.name for p in output_dir.iterdir() if p.is_file()}
    if not expected_names.issubset(final_names):
        raise AssertionError("Current slide's two GeoJSON outputs are missing after export")
    counts = Counter(int(x) for x in predictions)
    confidence = probs.max(axis=1)
    return {"n_detected": len(cells), "n_classified": len(probs),
            "class_counts": {CLASS_NAMES[i]: int(counts[i]) for i in range(7)},
            "class_proportions": {CLASS_NAMES[i]: float(counts[i] / len(cells)) for i in range(7)},
            "confidence": {"mean": float(confidence.mean()),
                           "p10": float(np.quantile(confidence, 0.1)),
                           "median": float(np.median(confidence)),
                           "p90": float(np.quantile(confidence, 0.9))},
            "geojson_qc": "PASS", "output_files": [str(output_dir / x) for x in sorted(expected_names)]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path,
                    help="Intermediate directory (default: result/task013_<slide>_wsi_inference)")
    ap.add_argument("--model-code-dir", type=Path, default=Path("/data/lf_data/result/final_model/code"))
    ap.add_argument("--detector-gpu", type=int, default=0)
    ap.add_argument("--classifier-gpu", type=int, default=1)
    ap.add_argument("--detector-batch-size", type=int, default=8)
    ap.add_argument("--classifier-batch-size", type=int, default=64)
    ap.add_argument("--tile-size", type=int, default=512)
    args = ap.parse_args()
    start = time.monotonic()
    args.svs = args.svs.resolve()
    if args.work_dir is None:
        args.work_dir = Path("/data/lf_data/result") / f"task013_{args.svs.stem.lower()}_wsi_inference"
    if not args.svs.is_file() or args.svs.suffix.lower() != ".svs":
        raise FileNotFoundError(f"Expected SVS input: {args.svs}")
    if args.classifier_batch_size < 1 or args.detector_batch_size < 1 or args.tile_size < 64:
        raise ValueError("Invalid batch/tile size")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata(args.svs)
    if sha256(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("Task012 classifier checkpoint hash mismatch")
    if not DETECTOR.is_file():
        raise FileNotFoundError(DETECTOR)
    print(f"Slide metadata: {meta}; classifier SHA256 verified", flush=True)
    detected = detect(args, args.svs.stem, args.work_dir)
    cells = validate_cells(detected, meta)
    print(f"Retained CellViT nuclei: {len(cells)}", flush=True)
    probs = classify(cells, meta, args, args.work_dir)
    qc = export(cells, probs, args.svs.stem, args.work_dir, args.output_dir)
    qc.update({"status": "COMPLETED", "svs": str(args.svs), "slide": meta,
               "detector_checkpoint": str(DETECTOR), "classifier_checkpoint": str(CHECKPOINT),
               "classifier_sha256": EXPECTED_SHA, "detector_resolution_mpp": 0.25,
               "crop_fov_um": 12, "resampled_crop_px": 57, "seed": None,
               "elapsed_seconds": time.monotonic() - start,
               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
               "independent_accuracy_estimable": False})
    (args.work_dir / "task013_qc.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2), flush=True)


if __name__ == "__main__":
    main()
