"""Evaluation entry point: `python -m cloth_segment.evaluate <run> <test_set>`.

Cached per (run, test set, protocol version); writes `eval/<test_set>/metrics.json`.
Tasks T5 and T9.
"""


def main() -> None:
    raise NotImplementedError("T5/T9: evaluation pipeline")


if __name__ == "__main__":
    main()
