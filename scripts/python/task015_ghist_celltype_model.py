#!/usr/bin/env python3
"""Classification-only adapter around the unmodified official GHIST modules.

The module deliberately does not import or instantiate GHIST Framework because
that class unconditionally creates gene-expression heads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

GHIST_ROOT = Path("/data/lf_data/models/GHIST")
if str(GHIST_ROOT) not in sys.path:
    sys.path.insert(0, str(GHIST_ROOT))

from model.backbone import Backbone
from model.modules import Embed, MLP


class GHISTCellType(nn.Module):
    """H&E + positive patch-local nucleus IDs -> morphology and cell logits."""

    def __init__(self, n_classes: int = 7, emb_dim: int = 256):
        super().__init__()
        if n_classes != 7 or emb_dim != 256:
            raise ValueError("Task015 fixes seven classes and a 256-d embedding")
        self.cnn = Backbone(
            n_channels=3, bilinear=True, is_deconv=True,
            is_batchnorm=True, n_classes=n_classes + 1,
        )
        # Official Framework uses 2 * (hd1_channels + h1_channels) = 768.
        self.embed_hist = Embed(768, emb_dim)
        self.mlp_hist = MLP(emb_dim, emb_dim, n_classes)

    def forward(self, x_hist: torch.Tensor, nuclei_mask: torch.Tensor):
        if x_hist.ndim != 4 or x_hist.shape[1] != 3:
            raise ValueError("x_hist must be [batch,3,height,width]")
        if nuclei_mask.shape != (x_hist.shape[0], *x_hist.shape[-2:]):
            raise ValueError("nuclei_mask must match batch and spatial dimensions")
        out_map, hd1, h1 = self.cnn(x_hist)
        if hd1.shape[-2:] != nuclei_mask.shape[-2:] or h1.shape[-2:] != nuclei_mask.shape[-2:]:
            raise RuntimeError("Official GHIST feature volumes do not match nucleus-mask resolution")
        patch_hd1 = hd1.mean(dim=(-2, -1))
        patch_h1 = h1.mean(dim=(-2, -1))
        features = []
        ordered_ids = []
        areas = []
        for batch_idx in range(x_hist.shape[0]):
            ids = torch.unique(nuclei_mask[batch_idx], sorted=True)
            ids = ids[ids > 0]
            for cell_id in ids.tolist():
                cell_mask = nuclei_mask[batch_idx] == cell_id
                area = cell_mask.sum()
                if not bool(area > 0):
                    continue
                cell_hd1 = hd1[batch_idx, :, cell_mask].mean(dim=1)
                cell_h1 = h1[batch_idx, :, cell_mask].mean(dim=1)
                features.append(torch.cat((cell_hd1, cell_h1,
                                           patch_hd1[batch_idx], patch_h1[batch_idx])))
                ordered_ids.append((batch_idx, int(cell_id)))
                areas.append(area)
        if features:
            feature_matrix = torch.stack(features)
            if feature_matrix.shape[1] != 768:
                raise RuntimeError(f"Unexpected official GHIST feature size: {feature_matrix.shape}")
            embeddings = self.embed_hist(feature_matrix)
            cell_logits, _ = self.mlp_hist(embeddings)
            area_tensor = torch.stack(areas)
        else:
            embeddings = hd1.new_zeros((0, 256))
            cell_logits = hd1.new_zeros((0, 7))
            area_tensor = hd1.new_zeros((0,), dtype=torch.long)
        return {
            "pixel_logits": out_map,
            "cell_logits": cell_logits,
            "ordered_ids": ordered_ids,
            "embeddings": embeddings,
            "instance_areas": area_tensor,
        }
