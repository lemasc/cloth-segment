# Contributing

Rules for everyone working in this repo, human or agent. Scope, decisions
(D1–D9) and task status live in [`docs/PLAN.md`](docs/PLAN.md); read it first.
Coding details are in [`docs/conventions.md`](docs/conventions.md).

## 1. Setup

```bash
uv sync
uv run pre-commit install        # once per clone: ruff + notebook + safety hooks
uv run pytest
```

Then point `datasets/`, `data/` and `runs/` at storage with enough space. They
are gitignored and the code only uses these three paths, so either works:

- **Shared storage** (e.g. the team machine): `scripts/setup_links.sh [ROOT]`
  symlinks them to a shared root.
- **Your own machine**: create the three folders in the clone, or symlink them
  to a disk of your choice. Datasets and their sources are described in
  `docs/handoffs/2026-11-03-kickoff.md` (a proper data-sources section comes
  with T11).

Never commit datasets, checkpoints or run outputs.

## 2. Workflow

1. **Pick a task.** Choose a `todo` row in `docs/PLAN.md` whose dependencies
   are done. Tell the team (chat, not Git) that you take it, so nobody else
   starts the same task. Do not take a task that is already `doing`.
   `M` rows (maintenance) are decisions or data fixes that are not
   deliverables, for example a change to a D row or a split filter. They work
   like any task: one ID per branch and PR, a handoff, and a status in
   `PLAN.md`.
   `X` rows (experiments) are training or evaluation runs: one run, its
   config, its run folder and its report. The T row that implements the model
   only covers code, config and tests. The run is a separate X row, on branch
   `X3/<slug>`, and its PR holds the config, the report and the handoff. A
   retrain or a seed rerun gets a new X row; do not put it on a `docs/` branch
   or reopen a `done` T row. Take the GPU lock and follow section 7. Fill the
   row's run folder and report in the same PR, and set it to `done` only when
   both exist.
2. **Branch** from an up-to-date `main`: `<task-id>/<slug>`, e.g.
   `T4/dataloader`. One task per branch, one PR per task. Never commit to
   `main`. Make your first commit `plan: start T4`, which only sets the task
   to `doing` in `PLAN.md`.
3. **Commit** early and often, using conventional prefixes: `feat:`, `fix:`,
   `refactor:`, `test:`, `docs:`, `plan:`, `chore:`. Write the subject in the
   imperative; explain *why* in the body when it is not obvious.
4. **Open a PR** into `main` using the template. Keep it small and focused. Run
   the checks in section 3 first: there is no CI, so the author is the CI.
   Before you ask for review, add the handoff note (section 6) and set the
   task to `done` in `PLAN.md` (a `plan: finish T4` commit). If the review
   asks for big changes, set it back to `doing` until they are done.
5. **Review.** The maintainer reviews every PR. Address comments with new
   commits (do not force-push over a review in progress). PRs are merged with
   a **merge commit**, so every commit on your branch stays in `main`'s history
   (this keeps the commit hash in a run's folder name, D7, reachable). Because
   all commits are kept, make them meaningful: tidy "wip" or "oops" commits
   before asking for review (see `docs/conventions.md`).

Each member works in their own clone. Do not edit files in someone else's
clone, or in shared storage by hand.

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
| Label set and converters | `data/labels.py`, `prepare_masks.py` | frozen | T4 loader merged and reviewed |
| Splits and manifest | `data/splits.py`, `dedup.py`, `build_splits.py` | frozen | T4 merged; duplicate thresholds reviewed |
| Evaluation protocol and metrics | `evaluate.py` | frozen | T5 merged; metrics tested |
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

## 5. Language and writing

Write in whichever of **Thai or English** you understand best. Handoffs, PR
descriptions, review comments and issues may be in either, and you may mix
them. Being understood matters more than polished English.

- **Always English:** code identifiers, config keys, file and branch names, and
  commit subjects. Tools, agents and `git log` searches depend on it.
- Code comments and docstrings may be in either language; stay consistent
  within a file.
- Write plainly: short sentences, define a term the first time you use it, put
  numbers in tables, and say what you did before why.
- If you cannot follow a document, say so in the PR or an issue. That is a bug
  in the document, not in you.
- Reviewer and agent replies follow the language of the person they are
  answering.

## 6. Handoff notes

Add the handoff note in the same PR as the work, on your task branch (also
when you stop mid-way: push the branch with the note). Create
`docs/handoffs/<yyyy-mm-dd>-<task>-<name>.md` from
[`docs/handoffs/TEMPLATE.md`](docs/handoffs/TEMPLATE.md). Put findings, results,
gotchas and open questions there, not in `PLAN.md`. Start with the three-line
"In plain words" section: a teammate should understand the outcome from it
alone. Reports in `docs/reports/` follow the same rules and are written for
readers who have not seen the code.

## 7. Experiments and hardware

- One YAML config per experiment. The resolved config hash is the cache key
  (D7): never retrain a finished run with the same hash, never edit a finished
  run's folder.
- Commit before launching any run you may keep; the run name includes the
  commit.
- Launch long runs from a terminal (`tmux` or similar), not a notebook kernel
  (D6).
- **If the GPU is shared** (the team machine): take the GPU lock (see
  `runs.py`), release it when done, and do not hold it idle. Do not delete
  other members' runs or data; clean up only your own.
- **If you use your own GPU:** no lock is needed. Results from different
  hardware are comparable only when the config hash and data version match, so
  quote both. Small numeric differences across GPUs are expected. The
  maintainer decides which runs are reported.

### Long-running jobs (draft)

> Draft. Written after one full run (T6). Revise it after more runs. It applies
> to anything that runs long, such as training or a full evaluation, and is
> written for people and for coding agents. Tool names are examples; use your
> harness's equivalent.

1. **Do not assume anyone waits.** The person may watch, may leave, or the
   machine may shut down. Every run has a run folder (D7). Give its full path
   as soon as it exists, so a later session can find it. The run folder is the
   record: `meta.json` (status), `log.jsonl` (loss and validation) and
   `error.txt` (a Python failure). Do not depend on terminal output.
2. **Detach where the system allows it.** Start the job so that closing the
   session does not kill it (`setsid nohup ... &`, or `tmux`). The run folder
   does not exist before launch, so send the launch output to a throwaway file.
   Do not say the run started until its folder appears. If it does not appear,
   read the throwaway file: that is a startup failure. If detaching is not
   possible, tell the person they must stay in the foreground, or ask them to
   run it themselves.
3. **Watch with the harness's event tools,** not blocking loops such as
   `until ...; do sleep; done`, which hold the session and are often blocked.
   In Claude Code, use the Monitor tool (30 minutes at most per arm) or a
   background shell task. A watcher should:
   - read the run folder (`meta.json`, `log.jsonl`), not printed text;
   - prefix every event with the time since `started_at`;
   - report a finish, a failure and a silent death. Silence is not success. A
     silent death is a process that is gone while `meta.json` still says
     `running`, for example after a hard kill or a shutdown;
   - test that the process is alive with an exact pattern or a PID. A loose
     `pgrep -f` pattern also matches the watcher's own command line, so the
     check never fails.
4. **Check the first iterations closely.** Look for errors, loss going down,
   seconds per iteration, and the first validation. Choose the print interval
   from the measured pace: about one event every few minutes, not every minute.
   Each event costs context and gives little.
5. **When the run looks healthy, stop the progress watch and hand control
   back.** Start a watch for errors and completion only, without asking first:
   a question leaves nothing watching until it is answered. Then end your turn
   with a message that gives the run path and an ETA (remaining iterations
   times the measured seconds per iteration, plus validation time), says what
   is being watched, and says that closing the session ends the watch but not
   the run. The person can then stop the watch, close the session, or carry on
   with other work. Anyone can check a run later in `meta.json` and
   `log.jsonl`.
6. **Keep that watch quiet.** Do not stream progress. At each expiry, read the
   status and the last log lines before re-arming. Stop re-arming when the run
   has ended or the person says to stop. Each expiry costs one turn, so say so
   if the run will last many hours.
7. **When the run completes and your session is still alive,** write the report
   file in `docs/reports/` yourself, without asking, then reply (section 6
   rules). Include the run folder, config hash and data version, results,
   interpretation, what is still unknown, and next tasks. Do not make up or
   estimate numbers.
8. **On a failure:** fix, test and commit first, and put the findings in the
   report. Then decide on a rerun. Rerun if it is cheap. Ask first if it is
   costly, for example when the last of four comparison models fails and the
   earlier runs may need redoing.

## 8. Notebooks

These are the recommended tools per folder, not hard rules. Pick another tool
if it fits the task better, and say why in the PR.

- EDA: marimo (`notebooks/eda/*.py`).
- Analysis: Jupyter (`notebooks/analysis/`), **outputs committed**; the hooks
  keep them and only strip volatile metadata.
- Visual review: marimo (`notebooks/review/*.py`), for looking at dataset images
  next to labels and model predictions. Dataset images must not be committed
  (licences), and a marimo notebook stores no outputs, so they stay on the
  shared mount. The notebook reads heavy inputs (predicted masks, counts) from
  files that a script wrote to `data/`. What a reviewer enters (verdicts,
  notes, decisions) goes to a small text file (CSV or JSON) in `docs/reports/`
  and is committed with the report.
- Keep outputs small (no giant image dumps, no widget state). A single
  notebook should stay well under 5 MB; the hook blocks bigger files.
- Reusable logic belongs in `src/`, not in a notebook.

## 9. Never commit

Datasets, images, masks, checkpoints, run outputs, credentials, `.env` files,
or anything over 5 MB. Check dataset licences before sharing any sample
outside the team.

## 10. Working with coding agents

Agents are welcome and may open PRs on your behalf. You stay responsible for
what is merged.

- Point the agent at `AGENTS.md` (Claude Code reads `CLAUDE.md`, which imports
  it). Give it a single task ID.
- Read the diff before you let it commit or open the PR. Run the section 3
  checks yourself.
- State in the PR if an agent wrote most of it (the template has a box).
- An agent must not claim results it did not produce. Numbers in a PR come from
  a run folder you can name.
