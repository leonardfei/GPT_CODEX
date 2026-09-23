# Task010 fold equivalence audit

The exact Task009 V3_CORE split manifest was reused; no new partition was generated.

| fold | train batches | validation batches |
|---:|---|---|
| 0 | s01A, s01B, s04B, s11, s22, s93 | s02A, s06A |
| 1 | s01A, s01B, s02A, s04B, s06A, s11, s93 | s22 |
| 2 | s01B, s02A, s04B, s06A, s22, s93 | s01A, s11 |
| 3 | s01A, s02A, s04B, s06A, s11, s22, s93 | s01B |
| 4 | s01A, s01B, s02A, s06A, s11, s22 | s04B, s93 |

No train/validation batch overlap was detected in any fold. The same fold membership was used for CellViT_TOKEN, PHIKON_V2_SMALL, PHIKON_V2_CONTEXT, the MLP probe, and the GT-centered upper bound.
