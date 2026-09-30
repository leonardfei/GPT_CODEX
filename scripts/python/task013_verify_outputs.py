#!/usr/bin/env python3
"""Independent final GeoJSON geometry/count check and spatial overlay QA for Task013."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw
import ujson

NAMES = ["Endothelial", "Mesenchymal", "Myeloid", "Neutrophil", "Plasma cell", "T and B", "Tumor"]
COLORS = [[76, 120, 168], [242, 207, 91], [178, 121, 162], [114, 183, 178],
          [255, 157, 166], [84, 162, 75], [228, 87, 86]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svs", required=True, type=Path)
    ap.add_argument("--geojson-dir", required=True, type=Path)
    ap.add_argument("--qc-json", required=True, type=Path)
    ap.add_argument("--overlay", required=True, type=Path)
    args = ap.parse_args()
    stem = args.svs.stem
    expected_files = {f"{stem}_cells.geojson", f"{stem}_cell_detection.geojson"}
    actual_files = {p.name for p in args.geojson_dir.iterdir()}
    if not expected_files.issubset(actual_files) or any(not name.endswith(".geojson") for name in actual_files):
        raise ValueError(f"Current slide GeoJSON pair missing or sidecar present: {actual_files}")
    source_qc = json.loads(args.qc_json.read_text())
    expected_counts = source_qc["class_counts"]
    slide = openslide.OpenSlide(str(args.svs))
    width, height = slide.dimensions
    try:
        shapes = {}
        for suffix, geometry in [("cells", "MultiPolygon"), ("cell_detection", "MultiPoint")]:
            features = ujson.loads((args.geojson_dir / f"{stem}_{suffix}.geojson").read_bytes())
            if not isinstance(features, list):
                raise ValueError(f"{suffix} is not a CellViT-style feature list")
            per_class = {}
            for f in features:
                if f["type"] != "Feature" or f["geometry"]["type"] != geometry:
                    raise ValueError(f"Invalid feature type in {suffix}")
                if f["properties"]["objectType"] != "annotation":
                    raise ValueError(f"Invalid QuPath annotation in {suffix}")
                classification = f["properties"]["classification"]
                name = classification["name"]
                if name not in NAMES or classification["color"] != COLORS[NAMES.index(name)]:
                    raise ValueError(f"Invalid class/color: {name}")
                if name in per_class:
                    raise ValueError(f"Duplicate grouped feature for {name}")
                per_class[name] = f["geometry"]["coordinates"]
            if {name: len(coordinates) for name, coordinates in per_class.items()} != {
                name: count for name, count in expected_counts.items() if count > 0
            }:
                raise ValueError(f"{suffix} per-class counts do not match classifier QC")
            shapes[suffix] = per_class
        n = 0
        invalid = Counter()
        all_points = []
        all_colors = []
        for name in NAMES:
            polygons = shapes["cells"].get(name, [])
            points = shapes["cell_detection"].get(name, [])
            for poly, pt in zip(polygons, points):
                if len(poly) != 1 or len(poly[0]) < 4 or poly[0][0] != poly[0][-1]:
                    invalid["open_or_short_ring"] += 1
                    continue
                ring = np.asarray(poly[0], dtype=np.float64)
                x, y = pt
                if not (np.isfinite(ring).all() and np.isfinite(pt).all()):
                    invalid["nonfinite"] += 1
                elif not (0 <= ring[:, 0].min() and ring[:, 0].max() < width and
                          0 <= ring[:, 1].min() and ring[:, 1].max() < height and
                          0 <= x < width and 0 <= y < height):
                    invalid["out_of_bounds"] += 1
                elif not (ring[:, 0].min() <= x <= ring[:, 0].max() and
                          ring[:, 1].min() <= y <= ring[:, 1].max()):
                    invalid["point_outside_contour_bbox"] += 1
                all_points.append((x, y))
                all_colors.append(COLORS[NAMES.index(name)])
                n += 1
        if invalid or n != source_qc["n_classified"]:
            raise ValueError(f"Independent geometry check failed: n={n}, errors={dict(invalid)}")
        thumbnail = slide.get_thumbnail((1200, 1200)).convert("RGB")
        tw, th = thumbnail.size
        overlay = Image.new("RGBA", thumbnail.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rng = np.random.default_rng(20260930)
        selected = rng.choice(n, size=min(20000, n), replace=False)
        for i in selected:
            x, y = all_points[i]
            color = tuple(all_colors[i]) + (180,)
            u, v = x * tw / width, y * th / height
            draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill=color)
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(thumbnail.convert("RGBA"), overlay).convert("RGB").save(args.overlay)
        result = {"status": "PASS", "n": n, "slide_dimensions": [width, height],
                  "feature_classes": list(shapes["cells"]), "per_class_counts": expected_counts,
                  "all_rings_closed": True, "all_geometry_in_bounds": True,
                  "centroids_within_paired_contour_bboxes": True,
                  "current_slide_pair_present_and_no_sidecars": True,
                  "overlay_sample_n": len(selected), "overlay_seed": 20260930}
        path = args.qc_json.with_name(f"{stem}_independent_geojson_qc.json")
        path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        slide.close()


if __name__ == "__main__":
    main()
