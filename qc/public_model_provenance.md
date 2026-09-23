# Task010 public model provenance audit

Status: `BLOCKED_PUBLIC_MODEL_DOWNLOAD`

## Official sources

- Phikon-v2: https://huggingface.co/owkin/phikon-v2
- Midnight-12k: https://huggingface.co/kaiko-ai/midnight
- Midnight official code: https://github.com/kaiko-ai/Midnight

The official model cards document Phikon-v2 as a ViT-L/16 DINOv2 pathology encoder with CLS features and an expected 1024-dimensional output. They document Midnight-12k as the public 12k-WSI variant with 224×224 input, `(0.5, 0.5, 0.5)` normalization, and classification features formed by concatenating the CLS token and mean patch-token embedding. No Midnight-92k variant was considered.

## Access findings

No official model files or cache were found on the remote server. The remote server's requests to the official Hugging Face config URLs timed out. A local request to the same official host also timed out. Because the official files could not be downloaded, no revision, checkpoint path, SHA256, runtime dimension, or feature extraction was verified.

No mirror, token, credential, random checkpoint, or alternative encoder was used.

The previous MUSK access-block files are retained separately under the Task010 output root and are not overwritten as provenance.
