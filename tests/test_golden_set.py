"""End-to-end golden-set evaluation for the compliance pipeline.

Run directly:   python tests/test_golden_set.py
Run via pytest: pytest tests/test_golden_set.py -s
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, "")  # ensure project root is on path

from src.evaluation.runner import GoldenSetRunner
from src.evaluation.report import ConsoleReporter, JsonReporter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_golden_set() -> None:
    """Run pipeline against all 38 golden cases and print evaluation report."""
    runner = GoldenSetRunner()
    result = runner.run()

    ConsoleReporter().report(result)

    json_report = JsonReporter().report_to_string(result)
    print(json_report)

    # pytest-friendly assertions — baseline placeholders
    assert result.total_cases == 38
    assert result.exact_match_accuracy >= 0.0


if __name__ == "__main__":
    test_golden_set()
