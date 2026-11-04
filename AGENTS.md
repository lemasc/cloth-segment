# Agent instructions

Clothing-centric human parsing: 3 segmentation models (U-Net, DeepLabV3+,
SegFormer) compared on LIP + CIHP + DeepFashion-MultiModal with one shared label
set. Python 3.12, `uv`, PyTorch.

**Read first:** `docs/PLAN.md` (scope, decisions D1–D9, task status), then
`CONTRIBUTING.md` and `docs/conventions.md`. Work on exactly one task ID.

## Commands

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest
uv run python -m cloth_segment.train configs/<experiment>.yaml
```

## Hard rules

- Never push to or commit on `main`. Work on a branch `<task-id>/<slug>`.
- Never use `--no-verify` or force-push.
- Shared contracts (label set, converters, splits/manifest, eval protocol,
  shared comparison fields) are listed in CONTRIBUTING section 4 with a status.
  Do not change one unless your task requires it; if you do, declare it in the
  PR, bump its version, and list what it affects. If its status is `frozen`,
  do not change it at all unless the task is an explicit `frozen-change`.
- Do not edit D-rows or the scope table in `PLAN.md`. You may set your own
  task's status and nothing else in that table.
- Do not write to `datasets/`, `data/` or `runs/` except through the project's
  scripts, and never delete anything there: they may be shared storage.
- Do not start a GPU job unless the user asked. If the GPU is shared, take the
  GPU lock first and release it afterwards.
- Do not add dependencies without asking; use `uv add` and commit `uv.lock`.
- Never invent or estimate metrics. Report a failing test or run as failing,
  with its output.
- Never commit data, checkpoints, credentials or files over 5 MB.

## Conventions in brief

- Reusable code in `src/cloth_segment/`; `scripts/` and notebooks stay thin.
- Everything configurable lives in YAML, no hard-coded paths or seeds.
- New logic gets a test that runs without the real datasets or a GPU.
- Conventional commits (`feat:`, `fix:`, `docs:` ...), small focused diffs.
- Ruff is enforced by pre-commit; fix what it reports, do not silence it.
- Jupyter notebooks keep their outputs. Do not clear them.

## Language

Reply and write handoffs, PR text and comments in the language the user writes
in (Thai or English). Keep identifiers, config keys, branch names and commit
subjects in English. Use short plain sentences and explain a term the first
time; the readers may not be fluent in the language used.

## Starting a task

Branch `<task-id>/<slug>`. Make the first commit `plan: start <task-id>`
(task to `doing` in `PLAN.md`, nothing else).

## Finishing a task

Everything below goes in the same PR as the work.

1. Run ruff and pytest; paste the result in the PR.
2. Add `docs/handoffs/<date>-<task>-<name>.md` from `docs/handoffs/TEMPLATE.md`.
3. Set the task to `done` in `PLAN.md` (`plan: finish <task-id>` commit).
4. Open the PR with the template filled in (tick the agent box). Stay within the
   task: list scope creep in the PR as follow-ups instead of doing it.
