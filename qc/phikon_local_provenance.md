# Local Phikon-v2 provenance

- Path: `/data/lf_data/models/phikon-v2`
- Fully offline loading: `AutoImageProcessor.from_pretrained(..., local_files_only=True)` and `AutoModel.from_pretrained(..., local_files_only=True)`
- `model.safetensors` SHA256: `261ae680fa699b3b951597fd57aa19c02ef735805acb104b93af69b36d928569`
- Architecture: `Dinov2Model`, hidden size 1024, patch size 16; runtime CLS feature dimension verified as 1024.
- Preprocessing: RGB, resize shortest edge 224, center crop 224×224, rescale 1/255, ImageNet mean/std, bicubic resampling.
- No alternate model, mirror, network download, credential, or token was used.
- Prior MUSK/public-download blocked provenance remains on the remote output root and was not overwritten.
