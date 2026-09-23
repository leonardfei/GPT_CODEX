# Task010 pre-alignment-fix archive manifest

The previous Task010 comparison outputs were archived under `/data/lf_data/result/task010_representation_benchmark/archive/pre_alignment_fix/` before corrected metrics were generated.

Archived categories include the previous report, seven-class linear metrics, old probability-ratio diagnostics, confusion flows, paired deltas, geometry, MLP metrics, and decision summary. The old CellViT baseline is superseded because its token tensor order was not reindexed to the canonical cell order.

The Phikon feature extraction itself was not identified as misaligned; it was nevertheless revalidated against canonical cell IDs. No raw data, labels, production checkpoint, or large tensor was deleted.
