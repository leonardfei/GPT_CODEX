# GPT-Codex Research Workflow

This repository separates scientific planning from computational implementation while keeping the shared state versioned in Git.

## Web GPT workflow

1. Web GPT defines the scientific question, analysis design, inputs, parameters, QC, and completion criteria.
2. Save the task as `tasks/task_XXX.md` using the three-digit identifier.
3. Commit and push the task to GitHub when a remote is configured.
4. Codex pulls the repository changes.
5. Codex executes the task and records the implementation and QC.
6. Codex generates outputs and `reports/task_XXX_report.md`.
7. Codex updates `PROJECT_STATUS.md`, commits, and pushes when possible.
8. Web GPT reads the report from the shared repository.
9. Web GPT performs scientific interpretation and makes decisions.
10. Web GPT generates the next task.

## Codex workflow

When Codex receives `Execute task_XXX.`, it should run:

```text
git pull
↓
read AGENTS.md, PROJECT_CONTEXT.md, SCIENTIFIC_DECISIONS.md, PROJECT_STATUS.md, and the task
↓
inspect inputs
↓
execute the requested implementation
↓
QC
↓
generate outputs
↓
write the report
↓
update PROJECT_STATUS.md
↓
git commit
↓
git push when a configured remote and permissions are available
```

If there is no remote, Codex should continue the local workflow, record the limitation, and avoid repeated push attempts.

## Web GPT should primarily read

```text
PROJECT_CONTEXT.md
SCIENTIFIC_DECISIONS.md
PROJECT_STATUS.md
reports/task_XXX_report.md
```

Web GPT should not need to read every implementation file for routine status review. It should inspect scripts and raw outputs when a report identifies a QC issue or when the scientific decision requires it.

## Directory conventions

- `tasks/`: task specifications from Web GPT;
- `scripts/`: executable analysis code;
- `results/`: small, reviewable tabular outputs;
- `figures/`: reviewable figures;
- `reports/`: structured computational reports;
- `logs/`: execution logs and diagnostics;
- `config/`: documented configuration;
- `docs/`: supporting documentation.

Do not commit credentials or large raw scientific data. Keep paths, parameters, commands, versions, anomalies, and deviations auditable in the report.
