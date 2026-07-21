"""Harness evaluation package for IELTS Writing Task 2 quality control."""

__all__ = [
    "MetricsCalculator",
    "Evaluator",
    "TestCaseManager",
    "Comparator",
    "Reporter",
]


def __getattr__(name):
    """Load harness classes lazily so CLI modules can run cleanly."""
    if name == "MetricsCalculator":
        from .metrics import MetricsCalculator

        return MetricsCalculator
    if name == "Evaluator":
        from .evaluator import Evaluator

        return Evaluator
    if name == "TestCaseManager":
        from .test_cases import TestCaseManager

        return TestCaseManager
    if name == "Comparator":
        from .comparator import Comparator

        return Comparator
    if name == "Reporter":
        from .reporter import Reporter

        return Reporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
