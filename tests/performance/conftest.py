"""Performance test configuration.

Skips all performance tests unless --run-performance is passed to pytest.
"""

import os

import pytest

# CI environment multiplier for wall-clock thresholds
CI_THRESHOLD_MULTIPLIER = 3.0
_is_ci = os.environ.get("CI", "").lower() == "true"


def pytest_collection_modifyitems(config, items):
    """Skip performance tests unless --run-performance is passed."""
    if config.getoption("--run-performance", default=False):
        return
    skip_perf = pytest.mark.skip(reason="need --run-performance option to run")
    for item in items:
        if "performance" in str(item.fspath):
            item.add_marker(skip_perf)


@pytest.fixture
def ci_threshold_multiplier():
    """Return a multiplier for wall-clock thresholds (3x in CI, 1x locally)."""
    return CI_THRESHOLD_MULTIPLIER if _is_ci else 1.0
