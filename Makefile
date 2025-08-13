# Makefile for XRayLabTool Python package
# Provides convenient commands for testing, development, and CI

.PHONY: help install test test-fast test-integration test-benchmarks test-coverage test-all clean lint format check-format docs

# Default target
help:
	@echo "XRayLabTool Development Commands"
	@echo "================================"
	@echo "install          Install package with development dependencies"
	@echo "test             Run all tests with coverage"
	@echo "test-fast        Run tests without coverage (faster)"
	@echo "test-integration Run integration tests only"
	@echo "test-benchmarks  Run performance benchmarks only"
	@echo "test-coverage    Run tests and generate HTML coverage report"
	@echo "test-all         Run comprehensive test suite using run_tests.py"
	@echo "lint             Run linting with flake8"
	@echo "format           Format code with black"
	@echo "check-format     Check if code needs formatting"
	@echo "clean            Clean up build artifacts and cache files"
	@echo "docs             Build documentation"

# Installation
install:
	pip install -e .[dev]

# Testing targets
test:
	pytest tests/ -v --cov=xraylabtool --cov-report=term-missing

test-fast:
	pytest tests/ -v

test-integration:
	pytest tests/test_integration.py -v

test-benchmarks:
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only -v

test-coverage:
	pytest tests/ --cov=xraylabtool --cov-report=html --cov-report=xml --cov-report=term-missing

test-all:
	python run_tests.py

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black xraylabtool tests *.py

check-format:
	black --check xraylabtool tests *.py

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf benchmark.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	@echo "Documentation build not configured yet"

# CI simulation
ci-test:
	@echo "Simulating CI test environment..."
	$(MAKE) clean
	$(MAKE) install
	$(MAKE) lint
	$(MAKE) test-coverage
	$(MAKE) test-benchmarks

# Development setup
dev-setup: install
	@echo "Development environment set up successfully!"
	@echo "Try: make test-fast"

# Quick development cycle
dev: check-format lint test-fast

# Full validation (use before pushing)
validate: format lint test-coverage test-benchmarks
	@echo "✅ All validation steps passed!"

# Performance monitoring
perf-baseline:
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-save=baseline

perf-compare:
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-compare=baseline

# Package building
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*
