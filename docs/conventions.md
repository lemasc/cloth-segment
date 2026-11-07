# Conventions

Details behind `CONTRIBUTING.md`. Keep this short; if a rule is not here, follow
the surrounding code.

## Code

- Python 3.12. Format and lint with ruff (config in `pyproject.toml`, line length
  100). Pre-commit runs it; do not add `# noqa` without a reason in the comment.
- Type hints on public functions. A one-line docstring saying *what and why* is
  enough; do not restate the signature.
- Library code in `src/cloth_segment/`. `scripts/` parse arguments and call the
  library. Notebooks present, they do not define logic.
- Use `cloth_segment.paths` for locations; no absolute paths or user names in
  code. Seeds come from config.
- Prefer small pure functions over classes with hidden state. Fail loudly on
  unexpected input (unknown label ids raise, as in `labels.py`).

## Tests

- Test: label mapping, converters, split and dedup logic, metrics, config
  hashing, the data loader's shapes and dtypes.
- Do not test: training loops end to end, or model accuracy.
- Tests use tiny synthetic arrays or fixtures and must run in seconds without
  the real data or a GPU.

## Configs and runs

- `configs/base.yaml` plus one YAML per experiment, overriding only what
  differs. A model's config must not change the shared fields (D5).
- Run folder: `<datetime>-<commit>-<slug>` (D7). Never edit a finished run.
- Anything that changes the run hash is an experiment; anything that changes the
  dataset or evaluation protocol is a versioned shared-contract change (D8;
  see CONTRIBUTING section 4 for open vs frozen).

## Notebooks

- marimo: `uv run marimo edit notebooks/<eda|review>/<name>.py`; keep cells
  cheap and load heavy results from files. A review notebook saves reviewer
  input to a committed text file in `docs/reports/` (CONTRIBUTING section 8).
- Jupyter: commit with outputs, restart-and-run-all before committing, and keep
  figures at a sensible resolution.

## Commits and PRs

- Conventional prefixes: `feat fix refactor test docs plan chore`.
- One task per PR; reformatting-only changes go in their own commit.
- PRs are merged with a merge commit; the branch commits stay in history, so
  each one should be a coherent step with a clear subject. Before review, fold
  throwaway commits ("wip", "fix typo") into the commit they belong to
  (`git commit --amend`, or `git rebase -i` if you are comfortable with it).
  After review starts, add commits instead of rewriting.
- The PR title becomes the merge commit subject, so write it well.

## Dependencies

- Add with `uv add` (dev tools with `uv add --dev`) and commit `uv.lock`.
- Justify any new dependency in the PR. Torch is pinned; do not bump it casually.
