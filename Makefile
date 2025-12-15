# Makefile for XRayLabTool Python package
# Optimized and Consolidated

.PHONY: help install install-docs dev-setup version-check test test-% cli-test cli-examples cli-help cli-demo lint format check-format type-check type-check-% docs docs-% clean clean-% dev validate ci-test release-check perf-% upload upload-% status info

# -----------------------------------------------------------------------------
# Configuration & Variables
# -----------------------------------------------------------------------------
SHELL := /bin/bash
PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
PYTEST_PARALLEL ?= -n auto
# Check for xdist/benchmark availability
PYTEST_XDIST_ARGS := $(shell $(PYTHON) -c "import xdist" >/dev/null 2>&1 && echo "$(PYTEST_PARALLEL)" || echo "")
PYTEST_BENCHMARK_ARGS := $(shell $(PYTHON) -c "import pytest_benchmark" >/dev/null 2>&1 && echo "" || echo "--benchmark-disable")

# Colors
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

# -----------------------------------------------------------------------------
# Help System
# -----------------------------------------------------------------------------
## @ Help: Show available commands
help:
	@echo "$(BLUE)XRayLabTool Development Commands$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@grep -E '^## @' $(MAKEFILE_LIST) | sed -e 's/## @ //g' | column -t -s ':' | sed -e 's/^/  /'

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
## @ install: Install package with dev dependencies (tries uv first)
install:
	@echo "$(YELLOW)Installing XRayLabTool...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(BLUE)Using uv...$(NC)"; uv sync --dev; \
	else \
		echo "$(BLUE)Using pip...$(NC)"; pip install -e .[dev]; \
	fi
	@echo "$(GREEN)✅ Installation complete$(NC)"

## @ install-docs: Install documentation dependencies
install-docs:
	@echo "$(YELLOW)Installing docs dependencies...$(NC)"
	pip install -r docs/requirements.txt

## @ dev-setup: Complete development setup
dev-setup: install install-docs
	@echo "$(GREEN)🚀 Dev environment ready$(NC)"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
# Pattern rule for specific test markers: make test-unit, test-integration, etc.
test-%:
	@echo "$(YELLOW)Running $* tests...$(NC)"
	@if [ "$*" = "all" ]; then \
		python run_tests.py; \
	elif [ "$*" = "fast" ]; then \
		$(PYTEST) $(PYTEST_XDIST_ARGS) tests/ -v; \
	elif [ "$*" = "coverage" ]; then \
		$(PYTEST) $(PYTEST_XDIST_ARGS) tests/ --cov=xraylabtool --cov-report=html --cov-report=term-missing; \
	elif [ "$*" = "ci" ]; then \
		$(PYTEST) $(PYTEST_XDIST_ARGS) tests/ -m "ci or (unit and not slow)" -v --tb=short --maxfail=5; \
	else \
		$(PYTEST) $(PYTEST_XDIST_ARGS) tests/ -m "$*" -v; \
	fi
	@echo "$(GREEN)✅ $* tests completed$(NC)"

## @ test: Run all tests with coverage
test: test-coverage

# -----------------------------------------------------------------------------
# Code Quality
# -----------------------------------------------------------------------------
## @ lint: Run flake8 and ruff checks
lint:
	@echo "$(YELLOW)Linting...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	@command -v ruff >/dev/null 2>&1 && ruff check . || echo "Ruff not found, skipping"
	@echo "$(GREEN)✅ Linting passed$(NC)"

## @ format: Format code (black, ruff, isort)
format:
	@echo "$(YELLOW)Formatting...$(NC)"
	black xraylabtool tests *.py
	@command -v ruff >/dev/null 2>&1 && ruff format . || true
	@command -v isort >/dev/null 2>&1 && isort . || true
	@echo "$(GREEN)✅ Formatting complete$(NC)"

## @ type-check: Run static type checks (core)
type-check:
	@echo "$(YELLOW)Type checking...$(NC)"
	@command -v mypy >/dev/null 2>&1 && python scripts/run_type_check.py --target core || echo "mypy not found"

## @ type-check-all: Run comprehensive type checks
type-check-all:
	@command -v mypy >/dev/null 2>&1 && python scripts/run_type_check.py --target all

## @ claude: Comprehensive code quality (lint, format, type, secure)
claude: format lint type-check-all
	@echo "$(BLUE)Running security scan...$(NC)"
	@command -v bandit >/dev/null 2>&1 && bandit -r xraylabtool/ -ll -ii || echo "bandit skipped"
	@echo "$(GREEN)🎉 Quality analysis complete$(NC)"

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
## @ docs: Build HTML documentation
docs:
	sphinx-build -b html docs docs/_build/html
	@echo "$(BLUE)View at: file://$(shell pwd)/docs/_build/html/index.html$(NC)"

## @ docs-serve: Serve documentation locally
docs-serve: docs
	cd docs/_build/html && python -m http.server 8000

## @ docs-clean: Clean docs build
docs-clean:
	rm -rf docs/_build/ docs/api/generated/

# -----------------------------------------------------------------------------
# CLI & Examples
# -----------------------------------------------------------------------------
## @ cli-test: Verify CLI installation and basic commands
cli-test:
	@echo "$(YELLOW)Testing CLI...$(NC)"
	xraylabtool --version
	xraylabtool --help > /dev/null
	xraylabtool list constants | head -5

## @ cli-demo: Run interactive CLI demo
cli-demo:
	@echo "$(GREEN)Running CLI Demo sequence...$(NC)"
	xraylabtool calc SiO2 -e 10.0 -d 2.2
	xraylabtool convert energy 10.0 --to wavelength

# -----------------------------------------------------------------------------
# Deployment & Cleanup
# -----------------------------------------------------------------------------
## @ build: Build distribution packages
build: clean
	python3 -m build

## @ upload: Upload to PyPI (requires confirmation)
upload: build
	@echo "$(RED)WARNING: Uploading to PyPI...$(NC)"
	@read -p "Are you sure? (y/N) " confirm && [ "$$confirm" = "y" ]
	python -m twine upload dist/*

## @ clean: Clean build artifacts (preserves venv)
clean:
	@echo "$(YELLOW)Cleaning...$(NC)"
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage .pytest_cache/ .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

## @ clean-all: Deep clean including venvs
clean-all: clean docs-clean
	rm -rf venv/ env/ .env/ node_modules/

# -----------------------------------------------------------------------------
# Workflows
# -----------------------------------------------------------------------------
## @ dev: Quick dev cycle (format, lint, test-fast)
dev: check-format lint type-check test-fast

## @ validate: Full pre-push validation
validate: format lint type-check test-coverage cli-test docs

## @ ci-test: CI simulation
ci-test: clean install lint type-check test-ci
