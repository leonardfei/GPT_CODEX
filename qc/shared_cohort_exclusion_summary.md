# Task010 shared cohort exclusion summary

The Task009 V3_CORE training dataset contained 156,300 eligible training cells. The frozen `SHARED_DETECTED_CORE` contains 96,044 cells after requiring a Task009 CellViT-detected/matched nucleus and valid 75×75 and 263×263 native crops. The cohort was frozen before Phikon feature extraction.

No model-dependent Phikon exclusion, Midnight-dependent exclusion, label revision, or historical-dataset dependency was applied. The GT-centered diagnostic separately used all 156,300 registered centers with valid crop bounds.
