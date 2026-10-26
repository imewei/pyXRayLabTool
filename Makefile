# Makefile for XRayLabTool
# Development, test, docs, build and release commands. Run `make help` for the list.
#
# Commands run against ./.venv when it exists (created by `make install` / `uv sync`),
# so no activation is needed. Override any tool, e.g. `make test PYTHON=python3.13`.

# Pytest runner and optional xdist parallelization
PYTHON ?= $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python)
ifneq ($(wildcard .venv/bin),)
export PATH := $(CURDIR)/.venv/bin:$(PATH)
endif
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
SPHINXBUILD ?= $(PYTHON) -m sphinx
PYTEST_PARALLEL ?= -n auto
PYTEST_PARALLEL_ENABLED ?= 1

# Coverage floor for `make claude`; measured 28.7% on 2026-10-23 (CI floor: COV_THRESHOLD in ci.yml).
COV_MIN ?= 27

.PHONY: help install dev-setup version-check \
	test test-fast test-unit test-integration test-characterization test-gui \
	test-performance test-memory test-stability test-benchmarks test-regression \
	test-optimization test-coverage test-parallel test-smoke test-edge test-ci \
	test-nightly test-all test-all-log \
	cli-test cli-examples cli-help cli-demo \
	lint format check-format type-check type-check-all type-check-strict \
	type-check-cache-info claude \
	docs docs-log docs-serve docs-autobuild docs-clean docs-linkcheck docs-pdf \
	docs-test docs-test-all docs-doctest \
	clean clean-all dev verify verify-fast validate ci-test release-check \
	perf-baseline perf-compare perf-report \
	test-install-local test-install-testpypi test-install-pypi \
	build upload-test upload status info quick-test \
	install-jax-gpu install-jax-gpu-cuda12 install-jax-gpu-cuda13 gpu-check gpu-diagnose

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# When running in parallel, disable pytest-benchmark to avoid it erroring under xdist.
PYTEST_BENCHMARK_DISABLE := $(shell $(PYTHON) -c "import pytest_benchmark" >/dev/null 2>&1 && printf '%s' "--benchmark-disable")

# Only enable xdist args when the plugin is installed and parallel is enabled.
PYTEST_XDIST_ARGS := $(shell \
	if [ "$(PYTEST_PARALLEL_ENABLED)" = "1" ]; then \
		$(PYTHON) -c "import xdist" >/dev/null 2>&1 && printf '%s %s' "$(PYTEST_PARALLEL)" "$(PYTEST_BENCHMARK_DISABLE)"; \
	fi)

# pyproject's addopts always collects coverage; the category targets switch it off.
NO_COV := --no-cov

# GUI/Qt tests segfault when split across xdist workers (native Qt state + worker
# threads from the threaded smoke test corrupt the process running another GUI
# test). Like CI, exclude them from the parallel run and execute them serially.
GUI_TESTS := tests/test_gui_smoke.py tests/test_gui_progress.py tests/unit/test_gui_widgets.py tests/unit/test_calculation_worker.py tests/unit/test_kk_plot.py tests/test_gui_kk_tab.py tests/test_gui_kk_worker_end_to_end.py
GUI_IGNORE := $(addprefix --ignore=,$(GUI_TESTS))

# Extra coverage options for the serial GUI run, which closes the combined report.
PYTEST_COV_ARGS ?=

# Performance/memory/stability tests are deselected unless --run-performance is given.
# They measure wall time and RSS, so they always run serially.
PERF := --run-performance $(NO_COV) --benchmark-disable

# Default target
help:
	@echo "$(BLUE)XRayLabTool Development Commands$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(YELLOW)📦 Installation & Setup:$(NC)"
	@echo "  install          Install package with all dependencies (Python >=3.12, prefers uv)"
	@echo "  dev-setup        Complete development environment setup"
	@echo ""
	@echo "$(YELLOW)🖥️  GPU Acceleration:$(NC)"
	@echo "  install-jax-gpu         Auto-detect system CUDA and install GPU support"
	@echo "  install-jax-gpu-cuda13  Install with system CUDA 13 (SM >= 7.5)"
	@echo "  install-jax-gpu-cuda12  Install with system CUDA 12 (SM >= 5.2)"
	@echo "  gpu-check               Verify GPU backend, devices, SVD"
	@echo "  gpu-diagnose            Diagnose common GPU issues"
	@echo ""
	@echo "$(YELLOW)🧪 Testing:$(NC)"
	@echo "  test             Run all tests with coverage (GUI tests run serially)"
	@echo "  test-fast        Run all tests without coverage (faster)"
	@echo "  test-unit        Run tests/unit"
	@echo "  test-integration Run tests/integration"
	@echo "  test-characterization  Run golden-output regression tests"
	@echo "  test-gui         Run the PySide6 GUI tests serially"
	@echo "  test-performance Run tests/performance (needs --run-performance, serial)"
	@echo "  test-memory      Run memory management tests only"
	@echo "  test-stability   Run numerical stability tests only"
	@echo "  test-benchmarks  Run pytest-benchmark benchmarks only"
	@echo "  test-regression  Run regression-named tests (-k regression)"
	@echo "  test-optimization Run optimization validation tests only"
	@echo "  test-coverage    Run tests and generate HTML/XML coverage reports"
	@echo "  test-parallel    Run tests in parallel for faster execution"
	@echo "  test-all         Run tests, performance tests and CLI checks"
	@echo "  test-all-log     Same as test-all, output logged to test_results.log"
	@echo "  test-smoke       Run the CI smoke subset (quick validation)"
	@echo "  test-edge        Run edge case tests"
	@echo "  test-ci          Run the non-GUI suite the way CI does"
	@echo "  test-nightly     Run the extended suite (performance + slow tests)"
	@echo "  cli-test         Test CLI functionality"
	@echo ""
	@echo "$(YELLOW)🔧 Code Quality:$(NC)"
	@echo "  lint             Run linting with ruff"
	@echo "  format           Format code with ruff"
	@echo "  check-format     Check if code needs formatting"
	@echo "  type-check       Run enhanced type checking on core modules"
	@echo "  type-check-all   Run comprehensive type checking on all modules"
	@echo "  type-check-strict Run strict type checking with enhanced rules"
	@echo "  type-check-cache-info Show MyPy cache information and performance"
	@echo "  claude           Format, lint, mypy --strict, tests with coverage floor, perf checks"
	@echo ""
	@echo "$(YELLOW)📚 Documentation:$(NC)"
	@echo "  docs             Build Sphinx documentation"
	@echo "  docs-log         Build Sphinx documentation with output logged to docs_build.log"
	@echo "  docs-serve       Build and serve documentation locally"
	@echo "  docs-autobuild   Live server with auto-rebuild on changes"
	@echo "  docs-clean       Clean documentation build files"
	@echo "  docs-linkcheck   Check documentation links"
	@echo "  docs-pdf         Build PDF documentation (requires LaTeX)"
	@echo "  docs-test        Test documentation build with warnings as errors"
	@echo "  docs-test-all    Comprehensive documentation testing (examples, links, coverage)"
	@echo "  docs-doctest     Test code examples in docstrings and RST files"
	@echo ""
	@echo "$(YELLOW)⚡ CLI Tools:$(NC)"
	@echo "  cli-help         Show CLI help and available commands"
	@echo "  cli-examples     Run CLI examples to verify functionality"
	@echo "  cli-demo         Interactive CLI demonstration"
	@echo "  quick-test       Smoke-test the Python API and CLI"
	@echo ""
	@echo "$(YELLOW)🏗️  Building & Release:$(NC)"
	@echo "  build               Build sdist + wheel with uv and run twine check"
	@echo "  test-install-local  Test local wheel installation in clean environment"
	@echo "  test-install-testpypi  Test TestPyPI installation"
	@echo "  test-install-pypi   Test PyPI installation"
	@echo "  upload-test         Upload to TestPyPI"
	@echo "  upload              Upload to PyPI (releases normally go through the publish workflow)"
	@echo "  version-check       Check that every file recording the version agrees"
	@echo "  perf-baseline       Save a pytest-benchmark baseline"
	@echo "  perf-compare        Compare benchmarks against the saved baseline"
	@echo "  perf-report         Write benchmark_report.json"
	@echo ""
	@echo "$(YELLOW)🧹 Cleanup:$(NC)"
	@echo "  clean            Clean build artifacts and caches (preserves virtual environments)"
	@echo "  clean-all        Deep clean including virtual environments (.venv, venv, env, ENV)"
	@echo ""
	@echo "$(YELLOW)🚀 Development Workflows:$(NC)"
	@echo "  dev              Quick development cycle (check-format, lint, type-check, test-fast)"
	@echo "  verify           Local CI mirror (format + lint + mypy + unit + characterization) - run before push"
	@echo "  verify-fast      Quick verification (format + lint + mypy only, no tests)"
	@echo "  validate         Full validation (format, lint, strict types, coverage, benchmarks, docs)"
	@echo "  ci-test          Simulate CI environment"
	@echo "  release-check    Pre-release validation checklist"
	@echo "  status / info    Project status / package information"

# Installation & Setup
install:
	@echo "$(YELLOW)Installing XRayLabTool with all dependencies (Python >=3.12)...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(BLUE)Using uv (preferred)...$(NC)"; \
		uv sync; \
	else \
		echo "$(BLUE)uv not found; falling back to pip...$(NC)"; \
		pip install -e .; \
	fi
	@echo "$(GREEN)✅ Installation complete$(NC)"

dev-setup: install
	@echo "$(GREEN)🚀 Development environment set up successfully!$(NC)"
	@echo "$(BLUE)📋 Quick commands:$(NC)"
	@echo "  make verify          # Mirror CI locally before pushing"
	@echo "  make claude          # 🤖 Comprehensive code quality analysis"
	@echo "  make cli-help        # Show CLI help"
	@echo "  make test-fast       # Run tests quickly"
	@echo "  make docs-serve      # Build and serve docs"
	@echo "  make cli-examples    # Test CLI functionality"

version-check:
	@echo "$(YELLOW)Checking version consistency...$(NC)"
	@$(PYTHON) scripts/check_version.py --installed
	@echo "$(GREEN)✅ Version check complete$(NC)"

# Testing targets
test:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(GUI_IGNORE) tests/
	@echo "$(YELLOW)Running GUI tests serially...$(NC)"
	$(PYTEST) $(GUI_TESTS) --cov-append $(PYTEST_COV_ARGS)
	@echo "$(GREEN)✅ Tests completed$(NC)"

test-fast:
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) $(GUI_IGNORE) tests/
	@echo "$(YELLOW)Running GUI tests serially...$(NC)"
	$(PYTEST) $(NO_COV) $(GUI_TESTS)
	@echo "$(GREEN)✅ Fast tests completed$(NC)"

# Core test categories
test-unit:
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) $(GUI_IGNORE) tests/unit/
	@echo "$(GREEN)✅ Unit tests completed$(NC)"

test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) tests/integration/
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-characterization:
	@echo "$(YELLOW)Running characterization (golden-output) tests...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) tests/characterization/
	@echo "$(GREEN)✅ Characterization tests completed$(NC)"

test-gui:
	@echo "$(YELLOW)Running GUI tests serially...$(NC)"
	$(PYTEST) $(NO_COV) $(GUI_TESTS) --timeout=60
	@echo "$(GREEN)✅ GUI tests completed$(NC)"

# Performance and optimization tests
test-performance:
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(PYTEST) $(PERF) tests/performance/ --tb=short
	@echo "$(GREEN)✅ Performance tests completed$(NC)"

test-memory:
	@echo "$(YELLOW)Running memory management tests...$(NC)"
	$(PYTEST) $(PERF) tests/ -m "memory" --tb=short
	@echo "$(GREEN)✅ Memory tests completed$(NC)"

test-stability:
	@echo "$(YELLOW)Running numerical stability tests...$(NC)"
	$(PYTEST) $(PERF) tests/ -m "stability" --tb=short
	@echo "$(GREEN)✅ Stability tests completed$(NC)"

test-benchmarks:
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	$(PYTEST) --run-performance $(NO_COV) tests/ -m "benchmark" --benchmark-only
	@echo "$(GREEN)✅ Benchmarks completed$(NC)"

test-regression:
	@echo "$(YELLOW)Running regression tests...$(NC)"
	$(PYTEST) $(PERF) tests/ -k "regression" --tb=short
	@echo "$(GREEN)✅ Regression tests completed$(NC)"

test-optimization:
	@echo "$(YELLOW)Running optimization validation tests...$(NC)"
	$(PYTEST) $(PERF) tests/performance/test_optimization_validation.py --tb=short
	@echo "$(GREEN)✅ Optimization tests completed$(NC)"

# Test execution modes
test-coverage:
	@echo "$(YELLOW)Running tests with detailed coverage...$(NC)"
	$(MAKE) test PYTEST_COV_ARGS="--cov-report=html --cov-report=xml"
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/ and coverage.xml$(NC)"

test-parallel:
	@echo "$(YELLOW)Running tests in parallel...$(NC)"
	@$(MAKE) PYTEST_PARALLEL_ENABLED=1 test-fast
	@echo "$(GREEN)✅ Parallel tests completed$(NC)"

test-smoke:
	@echo "$(YELLOW)Running smoke tests...$(NC)"
	$(PYTEST) $(NO_COV) tests/unit/test_core.py tests/integration/test_integration.py::TestBasicSetupAndInitialization --tb=line
	@echo "$(GREEN)✅ Smoke tests completed$(NC)"

test-edge:
	@echo "$(YELLOW)Running edge case tests...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) $(GUI_IGNORE) tests/ -k "edge" --tb=short
	@echo "$(GREEN)✅ Edge case tests completed$(NC)"

test-ci:
	@echo "$(YELLOW)Running CI test suite...$(NC)"
	$(PYTEST) $(PYTEST_XDIST_ARGS) $(NO_COV) $(GUI_IGNORE) tests/unit/ tests/integration/ tests/characterization/ --tb=short --maxfail=5
	@echo "$(GREEN)✅ CI tests completed$(NC)"

test-nightly:
	@echo "$(YELLOW)Running nightly test suite...$(NC)"
	$(PYTEST) --run-performance $(NO_COV) tests/ -m "performance or slow" --tb=short
	@echo "$(GREEN)✅ Nightly tests completed$(NC)"

test-all: test test-performance cli-test
	@echo "$(GREEN)✅ Comprehensive test suite completed$(NC)"

test-all-log:
	@echo "$(BLUE)📝 Output will be saved to test_results.log$(NC)"
	@echo "🕒 Started: $$(date)" | tee test_results.log
	@if $(MAKE) test-all 2>&1 | tee -a test_results.log; then \
		echo "🕒 Completed: $$(date)" | tee -a test_results.log; \
	else \
		echo "❌ Failed: $$(date)" | tee -a test_results.log; \
		exit 1; \
	fi

# CLI Testing
cli-test:
	@echo "$(YELLOW)Testing CLI functionality...$(NC)"
	@echo "$(BLUE)🔍 Testing CLI installation...$(NC)"
	xraylabtool --version
	@echo "$(BLUE)🔍 Testing basic commands...$(NC)"
	xraylabtool --help > /dev/null
	xraylabtool list constants > /dev/null
	xraylabtool list fields > /dev/null
	xraylabtool calc SiO2 -e 10.0 -d 2.2 > /dev/null
	@echo "$(GREEN)✅ CLI tests passed$(NC)"

cli-examples:
	@echo "$(YELLOW)Running CLI examples...$(NC)"
	@echo "$(BLUE)Single material calculation:$(NC)"
	xraylabtool calc SiO2 -e 10.0 -d 2.2
	@echo ""
	@echo "$(BLUE)Unit conversion:$(NC)"
	xraylabtool convert energy 10.0 --to wavelength
	@echo ""
	@echo "$(BLUE)Formula analysis:$(NC)"
	xraylabtool formula SiO2
	@echo "$(GREEN)✅ CLI examples completed$(NC)"

cli-help:
	@echo "$(YELLOW)XRayLabTool CLI Help:$(NC)"
	xraylabtool --help
	@echo ""
	@echo "$(BLUE)Available subcommands:$(NC)"
	xraylabtool list examples

cli-demo:
	@echo "$(YELLOW)🎆 XRayLabTool CLI Interactive Demo$(NC)"
	@echo "$(BLUE)This demo shows the main CLI capabilities$(NC)"
	@echo ""
	@echo "$(GREEN)1. Basic calculation for quartz:$(NC)"
	xraylabtool calc SiO2 -e 10.0 -d 2.2
	@echo ""
	@echo "$(GREEN)2. Energy range scan:$(NC)"
	xraylabtool calc Si -e 8,10,12 -d 2.33 --fields formula,energy_kev,critical_angle_degrees
	@echo ""
	@echo "$(GREEN)3. Unit conversions:$(NC)"
	xraylabtool convert energy 8.048,10.0,12.4 --to wavelength
	@echo ""
	@echo "$(GREEN)4. Chemical analysis:$(NC)"
	xraylabtool formula Al2O3
	xraylabtool atomic Si,Al,O
	@echo ""
	@echo "$(GREEN)5. Bragg diffraction:$(NC)"
	xraylabtool bragg -d 3.14,2.45 -e 8.048
	@echo ""
	@echo "$(BLUE)🎆 Demo complete! Try 'make cli-help' for more options$(NC)"

# Code Quality
lint:
	@echo "$(YELLOW)Running linting checks...$(NC)"
	$(RUFF) check xraylabtool tests
	@echo "$(GREEN)✅ Linting completed$(NC)"

format:
	@echo "$(YELLOW)Formatting code with ruff...$(NC)"
	$(RUFF) format xraylabtool tests
	$(RUFF) check --fix xraylabtool tests
	@echo "$(GREEN)✅ Code formatting completed$(NC)"

check-format:
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	$(RUFF) format --check xraylabtool tests
	@echo "$(GREEN)✅ Format check passed$(NC)"

type-check:
	@echo "$(YELLOW)Running enhanced type checks...$(NC)"
	@$(PYTHON) scripts/run_type_check.py --target core
	@echo "$(GREEN)✅ Enhanced type checking completed$(NC)"

type-check-all:
	@echo "$(YELLOW)Running comprehensive type checks...$(NC)"
	@$(PYTHON) scripts/run_type_check.py --target all
	@echo "$(GREEN)✅ Comprehensive type checking completed$(NC)"

type-check-strict:
	@echo "$(YELLOW)Running strict type checks...$(NC)"
	@$(PYTHON) scripts/run_type_check.py --target core --strict
	@echo "$(GREEN)✅ Strict type checking completed$(NC)"

type-check-cache-info:
	@echo "$(YELLOW)Checking type cache information...$(NC)"
	@$(PYTHON) scripts/run_type_check.py --target core --cache-info
	@echo "$(GREEN)✅ Cache information displayed$(NC)"

claude:
	@echo "$(BLUE)🤖 Claude Code Comprehensive Quality Analysis$(NC)"
	@echo "$(BLUE)=============================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Phase 1: Code Formatting & Style$(NC)"
	@echo "$(BLUE)→ Running Ruff formatter...$(NC)"
	@$(RUFF) format xraylabtool/ tests/
	@echo "$(BLUE)→ Running Ruff import sorting + linting with auto-fixes...$(NC)"
	@$(RUFF) check xraylabtool/ tests/ --fix --show-fixes
	@echo "$(GREEN)✅ Phase 1 Complete: Formatting & linting$(NC)"
	@echo ""
	@echo "$(YELLOW)Phase 2: Type Safety Validation$(NC)"
	@echo "$(BLUE)→ Running MyPy strict type checking...$(NC)"
	@$(MYPY) xraylabtool/ --strict --show-error-codes
	@echo "$(GREEN)✅ Phase 2 Complete: Type safety$(NC)"
	@echo ""
	@echo "$(YELLOW)Phase 3: Test Coverage Validation$(NC)"
	@echo "$(BLUE)→ Running comprehensive test suite...$(NC)"
	@$(MAKE) --no-print-directory test PYTEST_COV_ARGS="--cov-fail-under=$(COV_MIN) --cov-report=json:coverage-claude.json -q"
	@echo "$(GREEN)✅ Phase 3 Complete: Test coverage (≥$(COV_MIN)%)$(NC)"
	@echo ""
	@echo "$(YELLOW)Phase 4: Performance Regression Tests$(NC)"
	@echo "$(BLUE)→ Running optimization validation...$(NC)"
	@$(PYTEST) $(PERF) tests/performance/test_optimization_validation.py --tb=short -x
	@echo "$(BLUE)→ Running numerical stability checks...$(NC)"
	@$(PYTEST) $(NO_COV) tests/unit/test_numerical_stability.py::TestNumericalStabilityChecks --tb=short
	@echo "$(GREEN)✅ Phase 4 Complete: Performance validation$(NC)"
	@echo ""
	@echo "$(GREEN)🎉 Claude Code Quality Analysis Complete!$(NC)"
	@echo "$(BLUE)📁 Artifacts Generated:$(NC)"
	@echo "   • coverage-claude.json (test coverage data)"
	@echo "   • Test output above for quality summary"
	@echo ""
	@echo "$(YELLOW)Ready for commit! 🚀$(NC)"

# Documentation
docs:
	@echo "$(YELLOW)Building Sphinx documentation...$(NC)"
	$(SPHINXBUILD) -b html docs docs/_build/html
	@echo "$(GREEN)✅ Documentation built successfully in docs/_build/html/$(NC)"
	@echo "$(BLUE)📖 View at: file://$(CURDIR)/docs/_build/html/index.html$(NC)"

docs-log:
	@echo "$(YELLOW)Building Sphinx documentation with logging...$(NC)"
	@echo "$(BLUE)📝 Output will be saved to docs_build.log$(NC)"
	@echo "$(BLUE)🕒 Build started at: $$(date)$(NC)" | tee docs_build.log
	@echo "$(BLUE)📁 Working directory: $$(pwd)$(NC)" | tee -a docs_build.log
	@echo "" >> docs_build.log
	@if $(SPHINXBUILD) -b html docs docs/_build/html 2>&1 | tee -a docs_build.log; then \
		echo "" >> docs_build.log; \
		echo "$(BLUE)🕒 Build completed successfully at: $$(date)$(NC)" | tee -a docs_build.log; \
		echo "$(GREEN)✅ Documentation built successfully with full log in docs_build.log$(NC)"; \
		echo "$(BLUE)📖 View at: file://$(CURDIR)/docs/_build/html/index.html$(NC)"; \
	else \
		echo "" >> docs_build.log; \
		echo "$(RED)❌ Build failed at: $$(date)$(NC)" | tee -a docs_build.log; \
		echo "$(RED)❌ Documentation build failed - check docs_build.log for details$(NC)"; \
		exit 1; \
	fi

docs-serve: docs
	@echo "$(YELLOW)Serving documentation locally...$(NC)"
	@echo "$(BLUE)Documentation server starting at http://localhost:8000$(NC)"
	@echo "$(BLUE)Press Ctrl+C to stop the server$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	@echo "$(YELLOW)Cleaning documentation build files...$(NC)"
	rm -rf docs/_build/
	rm -rf docs/api/generated/
	@echo "$(GREEN)✅ Documentation cleaned$(NC)"

docs-linkcheck:
	@echo "$(YELLOW)Checking documentation links...$(NC)"
	$(SPHINXBUILD) -b linkcheck docs docs/_build/linkcheck
	@echo "$(GREEN)✅ Link check completed$(NC)"

docs-pdf:
	@echo "$(YELLOW)Building PDF documentation...$(NC)"
	$(SPHINXBUILD) -b latex docs docs/_build/latex
	cd docs/_build/latex && pdflatex XRayLabTool.tex || echo "$(BLUE)LaTeX not available, PDF build skipped$(NC)"
	@echo "$(GREEN)✅ PDF documentation build attempted$(NC)"

docs-test:
	@echo "$(YELLOW)Testing documentation build with warnings as errors...$(NC)"
	$(SPHINXBUILD) -W -b html docs docs/_build/test
	@echo "$(GREEN)✅ Documentation test build completed$(NC)"

docs-doctest:
	@echo "$(YELLOW)Testing code examples in docstrings and documentation...$(NC)"
	$(SPHINXBUILD) -b doctest docs docs/_build/doctest
	@echo "$(GREEN)✅ Documentation code examples tested$(NC)"

docs-test-all:
	@echo "$(YELLOW)Running comprehensive documentation testing...$(NC)"
	$(PYTHON) scripts/test_docs.py
	@echo "$(GREEN)✅ Comprehensive documentation testing completed$(NC)"

docs-autobuild:
	@echo "$(YELLOW)Starting live documentation server with auto-rebuild...$(NC)"
	@if $(PYTHON) -c "import sphinx_autobuild" >/dev/null 2>&1; then \
		echo "$(BLUE)Live server at http://localhost:8000 (auto-reloads on changes)$(NC)"; \
		$(PYTHON) -m sphinx_autobuild docs docs/_build/html --host 0.0.0.0 --port 8000; \
	else \
		echo "$(BLUE)sphinx-autobuild not available. Install with: uv pip install sphinx-autobuild$(NC)"; \
		echo "$(BLUE)Using regular build and serve instead...$(NC)"; \
		$(MAKE) docs-serve; \
	fi

# Cleanup
clean:
	@echo "$(YELLOW)Cleaning build artifacts and cache files...$(NC)"
	@echo "$(BLUE)Note: Virtual environments (.venv/, venv/, env/) are preserved$(NC)"
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage .coverage.* coverage.xml coverage.json coverage-claude.json
	rm -rf .pytest_cache/ .benchmarks/ benchmark_report.json pytest-results.xml
	rm -rf .mypy_cache/ .ruff_cache/
	rm -rf docs_build.log test_results.log
	find . \( -path ./.venv -o -path ./.git \) -prune -o -type d -name __pycache__ -exec rm -rf {} +
	@echo "$(GREEN)✅ Cleanup completed (virtual environments preserved)$(NC)"

clean-all: clean docs-clean
	@echo "$(YELLOW)Deep cleaning ALL artifacts including virtual environments...$(NC)"
	@echo "$(RED)WARNING: This will delete virtual environments (.venv/, venv/, env/, ENV/)$(NC)"
	rm -rf .venv/ venv/ env/ ENV/
	@echo "$(GREEN)✅ Deep cleanup completed (run 'make install' to rebuild .venv)$(NC)"

# Development Workflows
dev: check-format lint type-check test-fast
	@echo "$(GREEN)Quick development cycle completed$(NC)"

# ===================
# Pre-push verification (run before pushing to ensure CI will pass)
# ===================
verify:
	@echo "$(BLUE)======================================$(NC)"
	@echo "$(BLUE)  FULL LOCAL CI VERIFICATION$(NC)"
	@echo "$(BLUE)======================================$(NC)"
	@echo ""
	@echo "Step 1/4: Format + lint"
	@$(RUFF) format --check xraylabtool/ tests/ || (echo "$(RED)Format check failed! Run 'make format'$(NC)" && exit 1)
	@$(RUFF) check xraylabtool/ tests/ || (echo "$(RED)Lint check failed!$(NC)" && exit 1)
	@echo ""
	@echo "Step 2/4: Type checking"
	@$(MYPY) xraylabtool/ --no-error-summary || (echo "$(RED)Type check failed!$(NC)" && exit 1)
	@echo ""
	@echo "Step 3/4: Unit tests"
	@$(PYTEST) tests/unit/ $(GUI_IGNORE) $(PYTEST_XDIST_ARGS) $(NO_COV) -q --tb=line || (echo "$(RED)Unit tests failed!$(NC)" && exit 1)
	@echo ""
	@echo "Step 4/4: Characterization tests"
	@$(PYTEST) tests/characterization/ $(PYTEST_XDIST_ARGS) $(NO_COV) -q --tb=line || (echo "$(RED)Characterization tests failed!$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)======================================$(NC)"
	@echo "$(GREEN)  ALL CHECKS PASSED - SAFE TO PUSH$(NC)"
	@echo "$(GREEN)======================================$(NC)"

verify-fast:
	@echo "$(BLUE)======================================$(NC)"
	@echo "$(BLUE)  QUICK LOCAL CI VERIFICATION$(NC)"
	@echo "$(BLUE)======================================$(NC)"
	@echo ""
	@echo "Step 1/2: Format + lint"
	@$(RUFF) format --check xraylabtool/ tests/ || (echo "$(RED)Format check failed! Run 'make format'$(NC)" && exit 1)
	@$(RUFF) check xraylabtool/ tests/ || (echo "$(RED)Lint check failed!$(NC)" && exit 1)
	@echo ""
	@echo "Step 2/2: Type checking"
	@$(MYPY) xraylabtool/ --no-error-summary || (echo "$(RED)Type check failed!$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)======================================$(NC)"
	@echo "$(GREEN)  QUICK CHECKS PASSED$(NC)"
	@echo "$(GREEN)======================================$(NC)"

validate: format lint type-check-strict test-coverage test-benchmarks cli-test docs-test-all
	@echo "$(GREEN)✅ Full validation completed - ready for commit!$(NC)"
	@echo "$(BLUE)💡 Tip: For comprehensive pre-commit analysis, run 'make claude'$(NC)"

ci-test: clean install version-check lint type-check test-coverage test-benchmarks cli-test docs-test
	@echo "$(GREEN)✅ CI simulation completed successfully$(NC)"

release-check: clean dev-setup version-check validate docs-doctest docs docs-linkcheck build test-install-local
	@echo "$(YELLOW)Pre-release validation checklist:$(NC)"
	@echo "$(BLUE)✓ Code formatted and linted$(NC)"
	@echo "$(BLUE)✓ All tests passing with coverage$(NC)"
	@echo "$(BLUE)✓ CLI functionality verified$(NC)"
	@echo "$(BLUE)✓ Documentation built, doctested and links checked$(NC)"
	@echo "$(BLUE)✓ Package built, twine-checked and installed into a clean venv$(NC)"
	@echo "$(BLUE)✓ Version consistency verified$(NC)"
	@echo "$(GREEN)✅ Ready for release!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Commit the version bump and changelog, push, and wait for CI"
	@echo "  2. Tag release: git tag -a v$$($(PYTHON) scripts/check_version.py --print) -m 'Release v$$($(PYTHON) scripts/check_version.py --print)'"
	@echo "  3. Push tag: git push origin --tags"
	@echo "  4. Create the GitHub Release, then run the 'Publish to PyPI' workflow"

# Performance Monitoring
perf-baseline:
	@echo "$(YELLOW)Creating performance baseline...$(NC)"
	$(PYTEST) --run-performance $(NO_COV) tests/integration/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-save=baseline
	@echo "$(GREEN)✅ Baseline saved$(NC)"

perf-compare:
	@echo "$(YELLOW)Comparing against performance baseline...$(NC)"
	$(PYTEST) --run-performance $(NO_COV) tests/integration/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-compare=baseline
	@echo "$(GREEN)✅ Performance comparison completed$(NC)"

perf-report:
	@echo "$(YELLOW)Generating performance report...$(NC)"
	$(PYTEST) --run-performance $(NO_COV) tests/integration/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-json=benchmark_report.json
	@echo "$(GREEN)✅ Performance report saved to benchmark_report.json$(NC)"

# Installation Testing (scripts/test_install.py: fresh venv, one SiO2 calculation)
test-install-local: build
	@echo "$(YELLOW)Testing local wheel installation in clean environment...$(NC)"
	@$(PYTHON) scripts/test_install.py wheel
	@echo "$(GREEN)✅ Local installation test passed$(NC)"

test-install-testpypi:
	@echo "$(YELLOW)Testing TestPyPI installation in clean environment...$(NC)"
	@$(PYTHON) scripts/test_install.py testpypi
	@echo "$(GREEN)✅ TestPyPI installation test passed$(NC)"

test-install-pypi:
	@echo "$(YELLOW)Testing PyPI installation in clean environment...$(NC)"
	@$(PYTHON) scripts/test_install.py pypi
	@echo "$(GREEN)✅ PyPI installation test passed$(NC)"

# Package Building & Release
build: clean
	@echo "$(YELLOW)Building distribution packages...$(NC)"
	uv build
	$(PYTHON) -m twine check dist/*
	@echo "$(GREEN)✅ Packages built in dist/$(NC)"
	@ls -la dist/

upload-test: build
	@echo "$(YELLOW)Uploading to TestPyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✅ Uploaded to TestPyPI$(NC)"
	@echo "$(BLUE)Test installation: make test-install-testpypi$(NC)"

upload: build
	@echo "$(YELLOW)Uploading to PyPI...$(NC)"
	@echo "$(RED)WARNING: This will upload to production PyPI!$(NC)"
	@printf "Are you sure? (y/N) "; read confirm; [ "$$confirm" = "y" ] || exit 1
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)✅ Uploaded to PyPI$(NC)"

# Utility Targets
status:
	@echo "$(BLUE)XRayLabTool Project Status:$(NC)"
	@echo "$(YELLOW)Version:$(NC) $$($(PYTHON) -c 'import xraylabtool; print(xraylabtool.__version__)')"
	@echo "$(YELLOW)Python:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Location:$(NC) $$(pwd)"
	@echo "$(YELLOW)Git branch:$(NC) $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "$(YELLOW)Git status:$(NC)"
	@git status --porcelain 2>/dev/null || echo "Not a git repository"

info:
	@echo "$(BLUE)XRayLabTool Package Information:$(NC)"
	@$(PYTHON) -c "import sys, xraylabtool as xlt; r = xlt.calculate_single_material_properties('SiO2', 10.0, 2.2); print(f'Package: XRayLabTool v{xlt.__version__}'); print(f'Python: {sys.version}'); print(f'API Test: ✓ Critical angle = {r.critical_angle_degrees[0]:.3f}°')"
	@echo "$(YELLOW)CLI Test:$(NC)"
	@xraylabtool --version

quick-test:
	@echo "$(YELLOW)Quick functionality test...$(NC)"
	@$(PYTHON) -c "import xraylabtool as xlt; result = xlt.calculate_single_material_properties('SiO2', 10.0, 2.2); print(f'✓ Python API: {result.critical_angle_degrees[0]:.3f}°')"
	@xraylabtool calc SiO2 -e 10.0 -d 2.2 --fields critical_angle_degrees | grep -E "Critical|SiO2" | head -2
	@echo "$(GREEN)✅ Quick test passed$(NC)"

# GPU Acceleration Targets
# ========================
# System CUDA detection and JAX GPU installation. The gpu_cuda12 / gpu_cuda13 extras
# are declared conflicting in pyproject.toml, so `uv sync --extra ...` swaps one for the
# other and removes the plugin that is no longer selected.
# The extras are dropped again by a plain `uv sync`; re-run the target after one.

UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

install-jax-gpu:
	@echo "$(YELLOW)Installing JAX with GPU support (auto-detect)...$(NC)"
ifeq ($(PLATFORM),linux)
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found — install CUDA toolkit first$(NC)"; \
		exit 1; \
	fi; \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	echo "System CUDA: $$CUDA_FULL"; \
	echo "GPU: $$GPU_NAME (SM $$SM_DISPLAY)"; \
	case "$$SM_VERSION" in \
		""|*[!0-9]*) echo "$(RED)Error: nvidia-smi could not report the GPU compute capability$(NC)"; exit 1;; \
	esac; \
	if [ "$$CUDA_VERSION" = "13" ]; then \
		if [ "$$SM_VERSION" -ge 75 ]; then \
			$(MAKE) install-jax-gpu-cuda13; \
		else \
			echo "$(RED)Error: SM $$SM_DISPLAY < 7.5 — cannot use CUDA 13$(NC)"; \
			exit 1; \
		fi; \
	elif [ "$$CUDA_VERSION" = "12" ]; then \
		if [ "$$SM_VERSION" -ge 52 ]; then \
			$(MAKE) install-jax-gpu-cuda12; \
		else \
			echo "$(RED)Error: SM $$SM_DISPLAY < 5.2 — GPU unsupported$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)Error: CUDA $$CUDA_VERSION unsupported (need 12 or 13)$(NC)"; \
		exit 1; \
	fi
else
	@echo "$(RED)Error: GPU acceleration requires Linux$(NC)"
	@echo "  Current platform: $(PLATFORM)"
	@exit 1
endif

install-jax-gpu-cuda13:
	@echo "$(YELLOW)Installing JAX with system CUDA 13...$(NC)"
ifeq ($(PLATFORM),linux)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra gpu_cuda13; \
	else \
		$(PYTHON) -m pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true; \
		$(PYTHON) -m pip install -e ".[gpu_cuda13]"; \
	fi
	@$(MAKE) gpu-check
	@echo "$(GREEN)✅ GPU acceleration installed (CUDA 13)$(NC)"
else
	@echo "$(RED)Error: CUDA 13 GPU acceleration requires Linux$(NC)"
	@exit 1
endif

install-jax-gpu-cuda12:
	@echo "$(YELLOW)Installing JAX with system CUDA 12...$(NC)"
ifeq ($(PLATFORM),linux)
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --extra gpu_cuda12; \
	else \
		$(PYTHON) -m pip uninstall -y jax-cuda13-plugin jax-cuda13-pjrt 2>/dev/null || true; \
		$(PYTHON) -m pip install -e ".[gpu_cuda12]"; \
	fi
	@$(MAKE) gpu-check
	@echo "$(GREEN)✅ GPU acceleration installed (CUDA 12)$(NC)"
else
	@echo "$(RED)Error: CUDA 12 GPU acceleration requires Linux$(NC)"
	@exit 1
endif

gpu-check:
	@echo "$(YELLOW)GPU Verification$(NC)"
	@$(PYTHON) scripts/gpu_check.py

gpu-diagnose:
	@echo "$(YELLOW)GPU Diagnostics$(NC)"
	@echo ""
	@echo "1. System CUDA:"
	@nvcc --version 2>/dev/null | grep "release" || echo "  nvcc not found"
	@echo ""
	@echo "2. GPU hardware:"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
		|| echo "  nvidia-smi not found (no NVIDIA GPU or driver not installed)"
	@echo ""
	@echo "3. Installed JAX/CUDA packages and plugin conflicts:"
	@$(PYTHON) scripts/gpu_check.py --diagnose
