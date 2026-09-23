# Corrected nested MLP validation audit

Outer validation batches were never used for epoch selection. For each outer fold, the last sorted outer-training batch was a prespecified inner validation group; the selected epoch was then refit from scratch on all outer-training cells and evaluated once on the untouched outer validation batches. The same rule and architecture were used for all five representations.
