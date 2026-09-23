# Task010 fold equivalence audit

Task010 did not run model evaluation because the official public encoders were unreachable. The Task009 V3_CORE split manifest was nevertheless inspected and the exact batch folds reserved for the future paired benchmark are:

| fold | train batches | validation batches |
|---:|---|---|
| 0 | s01A, s01B, s04B, s11, s22, s93 | s02A, s06A |
| 1 | s01A, s01B, s02A, s04B, s06A, s11, s93 | s22 |
| 2 | s01B, s02A, s04B, s06A, s22, s93 | s01A, s11 |
| 3 | s01A, s02A, s04B, s06A, s11, s22, s93 | s01B |
| 4 | s01A, s01B, s02A, s06A, s11, s22 | s04B, s93 |

No new partition was generated. No batch overlap was present in the source manifest.
