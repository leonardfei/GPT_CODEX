# Task010 H&E pixel-scale audit

- Source: `/data/lf_data/xenium_data/ID0060276.ome.tif`; audited native shape 50,000 × 23,451 × 3, uint8 RGB.
- Validated project scale: **0.2125 µm/px**, from the historical project notebook.
- OME PhysicalSize metadata was inconsistent with registration and was not used.
- SMALL: 16 µm FOV → 75 × 75 native px.
- CONTEXT: 56 µm FOV → 263 × 263 native px.
- Shared-cohort crops were centered on matched CellViT H&E nucleus centroids; GT-centered crops used registered target centers.
- Both crop sizes were extracted directly from the original OME-TIFF and then passed through the local Phikon-v2 processor.
