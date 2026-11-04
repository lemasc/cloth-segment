# Contributing

Rules for everyone working in this repo, human or agent. Scope, decisions
(D1–D9) and task status live in [`docs/PLAN.md`](docs/PLAN.md); read it first.
Coding details are in [`docs/conventions.md`](docs/conventions.md).

## 1. Setup

```bash
uv sync
uv run pre-commit install        # once per clone: ruff + notebook + safety hooks
scripts/setup_links.sh           # datasets/, data/, runs/ -> shared mount
uv run pytest
```

Never copy datasets, checkpoints or run outputs into the repo. They live on the
shared mount (D9) and are gitignored.

## 2. Workflow

1. **Claim a task.** Pick a `todo` row in `docs/PLAN.md` whose dependencies are
   done, set it to `doing`, and merge that one-line change first (small PR, or
   ask the reviewer to merge it directly). Do not claim a task that is already
   `doing`.
2. **Branch** from an up-to-date `main`: `<task-id>/<slug>`, e.g.
   `T4/dataloader`. One task per branch. Never commit to `main`.
3. **Commit** early and often, using conventional prefixes: `feat:`, `fix:`,
   `refactor:`, `test:`, `docs:`, `plan:`, `chore:`. Write the subject in the
   imperative; explain *why* in the body when it is not obvious.
4. **Open a PR** into `main` using the template. Keep it small and focused. Run
   the checks in section 3 first: there is no CI, so the author is the CI.
5. **Review.** The maintainer reviews every PR. Address comments with new
   commits (do not force-push over a review in progress). PRs are
   **squash-merged**.
6. **Finish.** After merge: set the task to `done` in `PLAN.md` (own small
   `plan:` commit) and add a handoff note (section 5).

Each member works in their own clone (D9). Do not edit files in someone else's
clone or in the shared mount by hand.

## 3. Before you open a PR

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The pre-commit hooks run ruff on every commit, so these normally pass already.
Do not bypass hooks (`--no-verify`).

A PR is ready when:

- checks pass and new logic has tests (see conventions);
- any experiment has its YAML config committed (D6);
- README / docs are updated where behaviour changed;
- any shared contract you touched is declared in the PR (section 4).

## 4. Shared contracts and freezing

Some parts define the experiment itself, so changing them silently
invalidates results. We call them **shared contracts**. Each has a status,
set only by the maintainer in this table:

| Contract | Where | Status | Freeze when |
|---|---|---|---|
| Label set and converters | `data/labels.py`, `prepare_masks.py` | open | T4 loader merged and reviewed |
| Splits and manifest | `data/splits.py`, `dedup.py`, `build_splits.py` | open | T4 merged; duplicate thresholds reviewed |
| Evaluation protocol and metrics | `evaluate.py` | open | T5 merged; metrics tested |
| Shared comparison fields (D5) | `configs/base.yaml` | open | T5 merged; first baseline run agreed |

**While a contract is `open`:** changes are allowed through a normal PR, but the
PR must (a) tick the shared-contract box in the template, (b) say what other
tasks or results it affects, and (c) bump the version where one exists (e.g.
`SPLIT_VERSION`) so nothing silently reuses stale data. Tell the team when one
merges. Do not start long training runs on an open contract you expect to
change.

**Once a contract is `frozen`:** the maintainer edits its row to `frozen`.
From then on a change needs its own PR labelled `frozen-change`, with the
reason and a version bump (D7/D8), and the results that depended on it are
re-run or marked stale. Never alter the protocol or test sets to improve a
number, frozen or not.

Changing a decision (a D row) or the scope table is the maintainer's call. Open
an issue or a PR that proposes it; do not just edit the row.

## 5. Handoff notes

When you finish a task (or stop mid-way), add
`docs/handoffs/<yyyy-mm-dd>-<task>-<name>.md` from
[`docs/handoffs/TEMPLATE.md`](docs/handoffs/TEMPLATE.md). Put findings, results,
gotchas and open questions there, not in `PLAN.md`.

## 6. Experiments and the shared machine

- One YAML config per experiment. The resolved config hash is the cache key
  (D7): never retrain a finished run with the same hash, never edit a finished
  run's folder.
- Commit before launching any run you may keep; the run name includes the
  commit.
- Only one GPU. Take the GPU lock (see `runs.py`) and release it when done. Do
  not hold it idle.
- Launch long runs from a terminal (`tmux`), not a notebook kernel (D6).
- Do not delete other members' runs or data. Clean up only your own.

## 7. Notebooks

- EDA: marimo (`notebooks/eda/*.py`). Analysis: Jupyter (`notebooks/analysis/`),
  **outputs committed**; the hooks keep them and only strip volatile metadata.
- Keep outputs small (no giant image dumps, no widget state). A single
  notebook should stay well under 5 MB; the hook blocks bigger files.
- Reusable logic belongs in `src/`, not in a notebook.

## 8. Never commit

Datasets, images, masks, checkpoints, run outputs, credentials, `.env` files,
or anything over 5 MB. Check dataset licences before sharing any sample
outside the team.

## 9. Working with coding agents

Agents are welcome and may open PRs on your behalf. You stay responsible for
what is merged.

- Point the agent at `AGENTS.md` (Claude Code reads `CLAUDE.md`, which imports
  it). Give it a single task ID.
- Read the diff before you let it commit or open the PR. Run the section 3
  checks yourself.
- State in the PR if an agent wrote most of it (the template has a box).
- An agent must not claim results it did not produce. Numbers in a PR come from
  a run folder you can name.
