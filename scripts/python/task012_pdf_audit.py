#!/usr/bin/env python3
"""Inspect the Task012 publication PDF package for editable vector content."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

EXPECTED = [
    "Fig1_model_ladder_macroF1.pdf", "Fig2_model_ladder_macroAUPRC.pdf",
    "Fig3_neutrophil_F1.pdf", "Fig4_neutrophil_AUPRC.pdf",
    "Fig5_midnight_FOV_sweep.pdf", "Fig6_musk_FOV_sweep.pdf",
    "Fig7_true_binary_AUROC.pdf", "Fig8_true_binary_AUPRC.pdf",
    "Fig9_per_class_F1.pdf", "Fig10_confusion_matrix.pdf",
    "Fig11_fold_pairing.pdf", "Fig12_final_strategy_schematic.pdf",
    "Fig13_final_recipe_table.pdf",
]


def embedded_font_count(page):
    font_objects = page["/Resources"]["/Font"].get_object()
    embedded = 0
    for obj in font_objects.values():
        font = obj.get_object()
        if "/DescendantFonts" in font:
            font = font["/DescendantFonts"][0].get_object()
        descriptor = font.get("/FontDescriptor")
        if descriptor is not None:
            descriptor = descriptor.get_object()
            if any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
                embedded += 1
    return len(font_objects), embedded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    actual = {p.name for p in args.figure_dir.glob("Fig*.pdf")}
    if actual != set(EXPECTED):
        raise RuntimeError(f"Figure set mismatch: missing={set(EXPECTED)-actual}; extra={actual-set(EXPECTED)}")
    for name in EXPECTED:
        path = args.figure_dir / name
        reader = PdfReader(str(path))
        if len(reader.pages) != 1: raise RuntimeError(f"Expected one page: {name}")
        page = reader.pages[0]
        text = page.extract_text() or ""
        images = len(page.images)
        fonts, embedded_fonts = embedded_font_count(page)
        if len(text.strip()) < 40 or images != 0 or fonts == 0 or embedded_fonts != fonts:
            raise RuntimeError(f"Vector/text QC failed: {name}: text={len(text)}, images={images}, fonts={fonts}, embedded={embedded_fonts}")
        rows.append({"file": name, "bytes": path.stat().st_size, "pages": 1,
                     "extractable_text_chars": len(text), "image_xobjects": images,
                     "font_resources": fonts, "embedded_font_resources": embedded_fonts})
    outcome = {"status": "PASS", "n_figures": len(rows),
               "all_one_page": True, "all_text_extractable": True, "all_fonts_embedded": True,
               "no_raster_image_xobjects": True, "figures": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps({k: outcome[k] for k in ("status", "n_figures", "all_one_page", "all_text_extractable", "no_raster_image_xobjects")}, indent=2))


if __name__ == "__main__": main()
