# Project Context

## Project

HBV-related hepatocellular carcinoma research project

## Major research themes

- tumor microenvironment heterogeneity
- single-cell transcriptomics
- spatial transcriptomics
- spatial proteomics
- neutrophil heterogeneity
- macrophage biology
- CAF biology
- MVI-associated spatial niches
- extracellular vesicle biomarkers
- recurrence prediction

## Data modalities

- scRNA-seq / snRNA-seq
- Xenium
- Visium
- spatial proteomics
- H&E pathology
- extracellular vesicle data
- clinical data

## Main scientific objective

To characterize HBV-HCC tumor microenvironment heterogeneity and spatial niches and identify clinically translatable biomarkers.

## Data locations

Initial repository inspection on 2026-09-21 found no `data/` directory and no scientific input files. The repository contained only the workflow setup plan before initialization. Add confirmed paths here after data are added or mounted; do not invent paths.

## Repository structure

The workflow framework now contains:

- root-level project context, decisions, status, and workflow documentation;
- `tasks/` for task specifications and templates;
- `reports/` for task reports and templates;
- `scripts/R/`, `scripts/python/`, and `scripts/shell/` for implementation code;
- `results/`, `figures/`, and `logs/` for reviewable outputs and execution records;
- `config/` for documented configuration;
- `docs/` for supporting documentation.

No pre-existing R project, Python package, analysis directory, metadata file, Xenium file, or other data directory was identified during the initial inspection. Task 000 confirmed that the workflow framework is present and ready for a future data-backed task.

## Important notes

This file contains relatively stable project context. Temporary findings belong in `PROJECT_STATUS.md` or task reports.
