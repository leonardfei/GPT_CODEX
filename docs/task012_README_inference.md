# Task012 H&E cell classifier inference

The exported model expects an RGB H&E crop centered on the CellViT nucleus centroid. Each input file must be **57 × 57 native pixels**, equivalent to a 12 μm field of view at the validated 0.2125 μm/px slide scale. Crop generation and detector recall are upstream responsibilities. Do not pass a 224 × 224 pre-resized crop; the scripts reproduce the Task011 bicubic resize and normalization.

## Model paths on the analysis server

- Single full-data model: `/data/lf_data/result/final_model/midnight_fov12_finalblock_7class.pth`
- Five fold-specific models: `/data/lf_data/result/final_model/ensemble/fold{0..4}_midnight_fov12_finalblock_7class.pth`

The full-data model was trained on all 96,044 canonical development cells for exactly three epochs. Its training cells are not an unbiased performance test. The reported performance remains the corrected Task011 grouped five-fold cross-validation estimate. Independent slide/patient validation is required before production replacement.

## Commands

Run in the server environment `/data/lf_data/task010_env/bin/python`:

```bash
cd /data/lf_data/result/final_model/code
/data/lf_data/task010_env/bin/python inference_single_model.py --input /path/to/crop.png --output /path/to/single.csv
/data/lf_data/task010_env/bin/python inference_single_model.py --input /path/to/crop_directory --output /path/to/single_directory.csv
/data/lf_data/task010_env/bin/python inference_single_model.py --input /path/to/crops_manifest.csv --output /path/to/single_manifest.csv
/data/lf_data/task010_env/bin/python inference_ensemble.py --input /path/to/crops_manifest.csv --output /path/to/ensemble.csv
```

The manifest must contain `crop_path`; optional `sample_id` or `cell_id` supplies the output identifier. Relative crop paths are resolved relative to the manifest file. Identifiers must be unique.

Output columns are `sample_id`, `predicted_class_id`, `predicted_class_name`, `p_Endothelial`, `p_Mesenchymal`, `p_Myeloid`, `p_Neutrophil`, `p_Plasma`, `p_T_and_B`, and `p_Tumor`.

The single model is simpler and faster for routine inference. The ensemble averages probability vectors from five independently refit fold models and is suitable when a larger runtime and storage budget is acceptable. Neither output should be interpreted as an externally validated clinical performance estimate.

The scripts load only trusted local checkpoints and the local Midnight source checkpoint. The final model checkpoint and five fold checkpoints are intentionally kept on the analysis server and are not stored in Git.
