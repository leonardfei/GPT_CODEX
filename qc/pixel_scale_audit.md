# Task010 H&E pixel-scale audit

Status: `BLOCKED_PUBLIC_MODEL_DOWNLOAD`; metadata audit completed, crop generation not performed.

- Source: `/data/lf_data/xenium_data/ID0060276.ome.tif`
- TIFF array metadata: 50,000 × 23,451 × 3, uint8 RGB
- OME metadata: `PhysicalSizeX = PhysicalSizeY = 352.77777777777777 µm`
- OME-derived scale: approximately 0.00706 µm/px, inconsistent with the validated registration workflow and therefore rejected
- Historical validated source: `/data/lf_data/xenium_data/Prepare_allcelltype_batch8_train8_test.ipynb`
- Historical validated scale: `PIXEL_SIZE = 0.2125` µm/px
- SMALL target: 16 × 16 µm, approximately 75.29 native pixels per side
- CONTEXT target: 56 × 56 µm, approximately 263.53 native pixels per side

The validated 0.2125 µm/px scale is recorded for the future rerun. No crop was generated while the official public encoders were unreachable.
