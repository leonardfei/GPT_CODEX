# Task011 fold equivalence audit

Exact Task009 CORE split_manifest.csv was reused.

| fold | n_train | n_val | train batches | validation batches |
|---:|---:|---:|---|---|
| 0 | 76,812 | 19,232 | s01A;s01B;s04B;s11;s22;s93 | s02A;s06A |
| 1 | 79,560 | 16,484 | s01A;s01B;s02A;s04B;s06A;s11;s93 | s22 |
| 2 | 71,161 | 24,883 | s01B;s02A;s04B;s06A;s22;s93 | s01A;s11 |
| 3 | 89,330 | 6,714 | s01A;s02A;s04B;s06A;s11;s22;s93 | s01B |
| 4 | 67,313 | 28,731 | s01A;s01B;s02A;s06A;s11;s22 | s04B;s93 |
