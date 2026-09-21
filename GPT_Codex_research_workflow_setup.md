# GPT–Codex 双代理科研工作流搭建指令

你现在是本科研项目的计算实施代理。请在当前 GitHub 项目中搭建一个标准化的“双代理科研工作流”，使：

- Web ChatGPT 负责研究设计、结果解释、分析决策和下一步任务制定；
- Codex 负责具体代码实现、数据处理、统计分析、QC、图表生成和结果整理；
- GitHub 作为两者之间的共享状态仓库、版本控制系统和任务记录系统。

请直接完成框架搭建，不要只给建议。

---

# 一、总体目标

建立如下闭环：

```text
Web GPT
→ 生成 tasks/task_XXX.md
→ Codex读取任务并执行
→ 生成 scripts / results / figures
→ 写 reports/task_XXX_report.md
→ 更新 PROJECT_STATUS.md
→ git commit
→ git push
→ Web GPT读取 GitHub 中的 report
→ 判断结果并生成下一轮 task
```

所有分析必须具有：

1. 可重复性
2. 参数可追踪性
3. 输入输出可追踪性
4. 异常可追踪性
5. 科学决策与代码实现分离
6. Git版本控制

不要改变已有科研数据和已有代码；如果已有同名目录或文件，请在保留原内容的基础上整合。

---

# 二、创建目录结构

在当前 Git repository 根目录中创建：

```text
.
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── SCIENTIFIC_DECISIONS.md
├── PROJECT_STATUS.md
├── README_WORKFLOW.md
│
├── tasks/
│   ├── README.md
│   └── templates/
│       └── task_template.md
│
├── reports/
│   ├── README.md
│   └── templates/
│       └── report_template.md
│
├── scripts/
│   ├── README.md
│   ├── R/
│   ├── python/
│   └── shell/
│
├── results/
│   └── README.md
│
├── figures/
│   └── README.md
│
├── logs/
│   └── README.md
│
├── config/
│   └── README.md
│
└── docs/
    └── README.md
```

如果以下目录已经存在：

```text
scripts/
results/
figures/
data/
```

不要删除、移动或覆盖已有内容，只补充缺失部分。

不要提交大型原始数据到 GitHub。

---

# 三、创建 AGENTS.md

AGENTS.md 是 Codex 的永久行为规范。

写入以下核心规则：

## Role

You are the computational implementation agent for a biomedical research project.

Scientific reasoning, interpretation and high-level research decisions are made by the supervising Web ChatGPT agent.

Your responsibilities are:

- inspect available data;
- implement requested analyses;
- write reproducible R/Python/shell code;
- execute analyses;
- perform QC;
- generate numerical results;
- generate figures;
- report errors, anomalies and limitations;
- preserve a complete audit trail.

You must not silently change the scientific design.

## Required workflow

For every task:

1. Read:
   - AGENTS.md
   - PROJECT_CONTEXT.md
   - SCIENTIFIC_DECISIONS.md
   - PROJECT_STATUS.md
   - the assigned `tasks/task_XXX.md`

2. Inspect relevant input files before analysis.

3. Verify:
   - input files exist;
   - expected columns exist;
   - sample numbers;
   - cell numbers where relevant;
   - missing values;
   - coordinate units;
   - factor/group definitions.

4. Implement the requested analysis.

5. Do not silently change:
   - thresholds;
   - groups;
   - exclusion criteria;
   - statistical tests;
   - distance cutoffs;
   - cell annotations;
   - biological definitions.

6. If implementation must deviate from the requested design:
   - stop that part of the analysis if necessary;
   - document the issue;
   - explain why;
   - propose alternatives;
   - do not silently substitute another method.

7. Save outputs to standardized locations.

8. Perform QC.

9. Create a structured report in:

`reports/task_XXX_report.md`

10. Update:

`PROJECT_STATUS.md`

11. Commit all appropriate code, reports, small results and figures to Git.

12. Push to the configured remote repository if authentication and permissions allow.

## Separation of evidence and interpretation

Codex must distinguish:

- observed result;
- quantitative evidence;
- QC finding;
- computational interpretation;
- biological hypothesis.

Do not present biological hypotheses as established conclusions.

## Reproducibility

Every completed task must record:

- software;
- environment;
- package versions;
- script name;
- command used;
- input paths;
- output paths;
- important parameters;
- random seed where applicable.

## Safety rules

Never:

- delete raw data;
- overwrite source data;
- change original annotations without explicit instruction;
- force-push Git history;
- commit credentials;
- commit API keys;
- commit passwords;
- commit private tokens;
- commit very large raw datasets.

---

# 四、创建 PROJECT_CONTEXT.md

建立科研项目长期背景文件。

内容先使用占位结构：

```md
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

Fill this section only after inspecting the repository.

Do not invent paths.

## Repository structure

Automatically summarize relevant existing repository directories.

## Important notes

This file contains relatively stable project context.

Temporary findings belong in PROJECT_STATUS.md or task reports.
```

然后检查当前 repository，自动补充可以确定的信息，例如已有：

- data目录；
- R项目；
- Python代码；
- Xenium文件；
- metadata；
- analysis目录。

不要凭空猜测数据。

---

# 五、创建 SCIENTIFIC_DECISIONS.md

这个文件只保存已经确定的科研规则和长期分析约定。

初始化：

```md
# Scientific Decisions

This file contains stable scientific and analytical decisions that should remain consistent across tasks.

Do not modify an existing decision unless explicitly instructed.

## Control definitions

To be defined.

## Cell-type definitions

To be defined.

## Statistical thresholds

To be defined.

## Spatial analysis conventions

To be defined.

## Survival analysis conventions

To be defined.

## Multiple-testing correction

Default:
Benjamini-Hochberg correction unless otherwise specified.

## Reproducibility

All stochastic analyses should use a fixed seed when possible.

## Decision log

Each future modification must record:

- date
- decision
- rationale
- task ID responsible for the change
```

不要自行填入未经确认的科学阈值。

---

# 六、创建 PROJECT_STATUS.md

这个文件用于 Web GPT 快速了解当前整个项目的执行状态。

结构：

```md
# Project Status

## Current task
None

## Last completed task
None

## Repository status
Initialized workflow framework.

## Recent completed analyses

None.

## Important findings

None.

## Outstanding QC issues

None.

## Open scientific questions

None.

## Pending tasks

None.

## Latest generated files

None.

## Last update

Automatically record date/time.
```

之后每完成一个 task 都必须更新它。

要求保持简洁，不要把完整分析结果全部复制进去。

---

# 七、创建 task_template.md

路径：

`tasks/templates/task_template.md`

内容：

```md
# Task XXX

## Status
PENDING

## Scientific question

Describe the biological or analytical question.

## Background

Relevant scientific context.

## Hypothesis

Optional.

## Input data

List required files.

## Required analysis

1.
2.
3.

## Statistical methods

Specify required tests.

## Parameters

List thresholds and analysis parameters.

## Required QC

Define QC checks.

## Required outputs

### Results
- results/taskXXX_*.csv

### Figures
- figures/taskXXX_*.pdf

### Scripts
- scripts/R/taskXXX_*.R
or
- scripts/python/taskXXX_*.py

### Report
- reports/task_XXX_report.md

## Do not

List operations that must not be performed.

## Scientific decisions required

Questions Codex should return to Web GPT rather than decide independently.

## Completion criteria

Define what constitutes completion.
```

---

# 八、创建 report_template.md

路径：

`reports/templates/report_template.md`

内容：

```md
# Task XXX Report

## Task
Brief summary of the requested analysis.

## Status
COMPLETED / PARTIAL / BLOCKED

## Input data

Files actually used.

## Data inspection

Report:
- sample count
- observation/cell count
- required columns
- missingness
- relevant QC

## Implementation

Describe exactly what was implemented.

## Parameters

List all important settings.

## Statistical methods

List tests and correction methods.

## QC

Report QC findings.

## Main results

Only report quantitative and directly observed results here.

Examples:

- effect sizes;
- medians;
- OR;
- confidence intervals;
- P values;
- adjusted P values;
- cell proportions;
- distances;
- model metrics.

## Figures generated

List figure paths.

## Result files generated

List result paths.

## Scripts generated

List script paths.

## Unexpected findings

Report:
- anomalies;
- outliers;
- warnings;
- unexpected distributions;
- suspicious samples.

## Deviations from requested task

State explicitly.

If none:

None.

## Scientific interpretation candidates

Clearly label these as hypotheses, not established conclusions.

## Scientific decisions required

Questions that should be decided by Web GPT.

## Recommended next analyses

Suggestions only.

Do not automatically execute unless included in the current task.

## Reproducibility

### Environment

Record environment.

### Package versions

Record important package versions.

### Command

Exact command used.

### Random seed

Record seed if relevant.

## Git

Commit hash:

Branch:

Push status:
```

---

# 九、任务状态规范

Task必须使用以下状态：

```text
PENDING
IN_PROGRESS
COMPLETED
PARTIAL
BLOCKED
```

开始执行时，把：

```text
Status: PENDING
```

修改成：

```text
Status: IN_PROGRESS
```

完成后修改为：

```text
COMPLETED
```

如果因为数据、依赖、科学决策等问题无法完成：

```text
BLOCKED
```

部分完成：

```text
PARTIAL
```

---

# 十、任务编号规范

所有任务统一使用三位编号：

```text
task_001.md
task_002.md
task_003.md
...
```

对应：

```text
reports/task_001_report.md

scripts/R/task001_*.R

scripts/python/task001_*.py

results/task001_*.csv

figures/task001_*.pdf
```

不要混用编号方式。

---

# 十一、Git规范

检查当前 repository 的：

```bash
git status
git branch --show-current
git remote -v
```

如果已经有 Git remote，不要改变 remote。

创建合理的 `.gitignore`，至少排除：

```gitignore
.env
*.key
*.pem
credentials*
secrets*
__pycache__/
.ipynb_checkpoints/
.Rhistory
.RData
.DS_Store

# large scientific data
*.h5ad
*.h5
*.hdf5
*.loom
*.rds
*.RDS
*.mtx
*.bam
*.bai
*.fastq
*.fastq.gz
*.fq
*.fq.gz
*.svs
*.tif
*.tiff
*.ome.tif
*.ome.tiff
```

但是：

不要覆盖用户已有 `.gitignore`。

只追加缺失且合理的规则。

如果某些上述格式已经被用户有意纳入Git管理，不要自动删除或取消跟踪。

---

# 十二、创建 README_WORKFLOW.md

向用户说明这个系统如何使用。

必须包括：

## Web GPT workflow

1. Web GPT generates task.
2. Save task as `tasks/task_XXX.md`.
3. Push to GitHub.
4. Codex pulls repository changes.
5. Codex executes task.
6. Codex generates report.
7. Codex commits and pushes.
8. Web GPT reads report.
9. Web GPT performs scientific interpretation.
10. Web GPT generates next task.

## Codex workflow

Codex收到类似：

```text
Execute task_001.
```

时应该：

```text
git pull
↓
read context files
↓
read task
↓
inspect data
↓
execute
↓
QC
↓
generate outputs
↓
write report
↓
update PROJECT_STATUS
↓
git commit
↓
git push
```

## Web GPT should primarily read

```text
PROJECT_CONTEXT.md
SCIENTIFIC_DECISIONS.md
PROJECT_STATUS.md
reports/task_XXX_report.md
```

而不是每次读取全部代码。

---

# 十三、建立一个 Codex 快捷执行规则

在 AGENTS.md 中加入：

当用户输入：

```text
Execute task_XXX.
```

Codex必须自动完成完整生命周期：

```text
git pull
→ read project state
→ execute requested task
→ QC
→ report
→ PROJECT_STATUS update
→ git commit
→ git push
```

除非发生真正阻塞的问题，否则不要要求用户逐步确认。

如果分析任务可以安全执行，则直接执行。

---

# 十四、建立 Web GPT 反馈字段

在每份 report 结尾加入：

```md
# Web GPT Review

## Questions for Web GPT

1.
2.
3.

## Decision required before next task

- [ ]

## Suggested next task

Provide a short suggestion only.
```

该区域用于 Web GPT 阅读和决策。

Codex不要替 Web GPT 做最终科学判断。

---

# 十五、创建第一个测试任务

创建：

`tasks/task_000.md`

用于验证框架本身，不执行真正科研分析。

内容：

```md
# Task 000

## Status
PENDING

## Scientific question

Workflow validation.

## Required analysis

1. Inspect repository structure.
2. Identify major existing analysis directories.
3. Identify coding languages currently used.
4. Identify likely data directories.
5. Do not open or modify large raw datasets.
6. Do not perform scientific analysis.

## Required output

Create:

reports/task_000_report.md

Update:

PROJECT_CONTEXT.md
PROJECT_STATUS.md

## Completion criteria

The workflow structure is functional and Git status is clean after commit.
```

然后立即执行 Task 000。

---

# 十六、Git commit

框架搭建完成后：

检查：

```bash
git status
```

确保没有：

- credentials；
- token；
- private key；
- 大型原始数据；
- 非预期删除。

然后创建 commit：

```text
Initialize GPT-Codex research workflow
```

如果当前 Git remote 可正常push，则push。

如果不能push：

不要反复尝试。

在最终报告中告诉我：

- branch；
- commit hash；
- push失败原因。

---

# 十七、最终向我汇报

全部完成后，请输出：

1. 创建了哪些文件和目录；
2. 当前Git branch；
3. remote；
4. commit hash；
5. 是否成功push；
6. Task 000执行结果；
7. 当前 PROJECT_STATUS；
8. 是否发现潜在数据安全或Git问题；
9. 下一步我应该在 Web GPT 做什么。

不要只回复“done”。

请实际创建、检查、执行并验证这个框架。
