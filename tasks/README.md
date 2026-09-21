# Tasks

Web GPT creates one Markdown task specification per requested computational unit.

Use three-digit identifiers: `task_001.md`, `task_002.md`, and so on. Start with `PENDING`, change to `IN_PROGRESS` when execution begins, and finish with `COMPLETED`, `PARTIAL`, or `BLOCKED`.

Copy `templates/task_template.md` for new tasks. The special `task_000.md` validates the workflow and does not perform scientific analysis.
