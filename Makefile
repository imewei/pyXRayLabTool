# Makefile for XRayLabTool Python package
# Provides convenient commands for testing, development, and CI
# Supports both Python API and CLI functionality

.PHONY: help install install-docs dev-setup version-check test test-fast test-integration test-benchmarks test-coverage test-all cli-test cli-examples cli-help cli-demo lint format check-format type-check docs docs-serve docs-clean docs-linkcheck docs-pdf clean clean-all dev validate ci-test release-check perf-baseline perf-compare perf-report test-install-local test-install-testpypi test-install-pypi build upload-test upload status info quick-test

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[0;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

# Default target
help:
	@echo "$(BLUE)XRayLabTool Development Commands$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(YELLOW)📦 Installation & Setup:$(NC)"
	@echo "  install          Install package with development dependencies"
	@echo "  dev-setup        Complete development environment setup"
	@echo "  install-docs     Install documentation dependencies"
	@echo ""
	@echo "$(YELLOW)🧪 Testing:$(NC)"
	@echo "  test             Run all tests with coverage"
	@echo "  test-fast        Run tests without coverage (faster)"
	@echo "  test-integration Run integration tests only"
	@echo "  test-benchmarks  Run performance benchmarks only"
	@echo "  test-coverage    Run tests and generate HTML coverage report"
	@echo "  test-all         Run comprehensive test suite using run_tests.py"
	@echo "  cli-test         Test CLI functionality"
	@echo ""
	@echo "$(YELLOW)🔧 Code Quality:$(NC)"
	@echo "  lint             Run linting with flake8"
	@echo "  format           Format code with black"
	@echo "  check-format     Check if code needs formatting"
	@echo "  type-check       Run type checking with mypy (if available)"
	@echo ""
	@echo "$(YELLOW)📚 Documentation:$(NC)"
	@echo "  docs             Build Sphinx documentation"
	@echo "  docs-serve       Build and serve documentation locally"
	@echo "  docs-clean       Clean documentation build files"
	@echo "  docs-linkcheck   Check documentation links"
	@echo ""
	@echo "$(YELLOW)⚡ CLI Tools:$(NC)"
	@echo "  cli-help         Show CLI help and available commands"
	@echo "  cli-examples     Run CLI examples to verify functionality"
	@echo "  cli-demo         Interactive CLI demonstration"
	@echo ""
	@echo "$(YELLOW)🏗️  Building & Release:$(NC)"
	@echo "  build               Build distribution packages"
	@echo "  test-install-local  Test local wheel installation in clean environment"
	@echo "  test-install-testpypi  Test TestPyPI installation"
	@echo "  test-install-pypi   Test PyPI installation"
	@echo "  upload-test         Upload to TestPyPI"
	@echo "  upload              Upload to PyPI"
	@echo "  version-check       Check version consistency"
	@echo ""
	@echo "$(YELLOW)🧹 Cleanup:$(NC)"
	@echo "  clean            Clean build artifacts and cache (preserves virtual environments)"
	@echo "  clean-all        Deep clean including virtual environments and all unrelated files"
	@echo ""
	@echo "$(YELLOW)🚀 Development Workflows:$(NC)"
	@echo "  dev              Quick development cycle (format, lint, test-fast)"
	@echo "  validate         Full validation (use before pushing)"
	@echo "  ci-test          Simulate CI environment"
	@echo "  release-check    Pre-release validation checklist"

# Installation & Setup
install:
	@echo "$(YELLOW)Installing XRayLabTool with development dependencies...$(NC)"
	pip install -e .[dev]
	@echo "$(GREEN)✅ Installation complete$(NC)"

install-docs:
	@echo "$(YELLOW)Installing documentation dependencies...$(NC)"
	pip install sphinx sphinx-rtd-theme
	@echo "$(GREEN)✅ Documentation dependencies installed$(NC)"

dev-setup: install install-docs
	@echo "$(GREEN)🚀 Development environment set up successfully!$(NC)"
	@echo "$(BLUE)📋 Quick commands:$(NC)"
	@echo "  make cli-help        # Show CLI help"
	@echo "  make test-fast       # Run tests quickly"
	@echo "  make docs-serve      # Build and serve docs"
	@echo "  make cli-examples    # Test CLI functionality"

version-check:
	@echo "$(YELLOW)Checking version consistency...$(NC)"
	@python -c "import xraylabtool; print(f'Package version: {xraylabtool.__version__}')"
	@grep -n "version =" pyproject.toml || echo "Version not found in pyproject.toml"
	@grep -n "release =" docs/source/conf.py || echo "Release not found in docs/source/conf.py"
	@echo "$(GREEN)✅ Version check complete$(NC)"

# Testing targets
test:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=xraylabtool --cov-report=term-missing
	@echo "$(GREEN)✅ Tests completed$(NC)"

test-fast:
	@echo "$(YELLOW)Running fast tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✅ Fast tests completed$(NC)"

test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	pytest tests/test_integration.py -v
	@echo "$(GREEN)✅ Integration tests completed$(NC)"

test-benchmarks:
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only -v
	@echo "$(GREEN)✅ Benchmarks completed$(NC)"

test-coverage:
	@echo "$(YELLOW)Running tests with detailed coverage...$(NC)"
	pytest tests/ --cov=xraylabtool --cov-report=html --cov-report=xml --cov-report=term-missing
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

test-all:
	@echo "$(YELLOW)Running comprehensive test suite...$(NC)"
	python run_tests.py
	@echo "$(GREEN)✅ All tests completed$(NC)"

# CLI Testing
cli-test:
	@echo "$(YELLOW)Testing CLI functionality...$(NC)"
	@echo "$(BLUE)🔍 Testing CLI installation...$(NC)"
	xraylabtool --version
	@echo "$(BLUE)🔍 Testing basic commands...$(NC)"
	xraylabtool --help > /dev/null
	xraylabtool list constants | head -5
	xraylabtool list fields | head -5
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
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	@echo "$(GREEN)✅ Linting completed$(NC)"

format:
	@echo "$(YELLOW)Formatting code with black...$(NC)"
	black xraylabtool tests *.py
	@echo "$(GREEN)✅ Code formatting completed$(NC)"

check-format:
	@echo "$(YELLOW)Checking code formatting...$(NC)"
	black --check xraylabtool tests *.py
	@echo "$(GREEN)✅ Format check passed$(NC)"

type-check:
	@echo "$(YELLOW)Running type checks...$(NC)"
	@command -v mypy >/dev/null 2>&1 && mypy xraylabtool/ || echo "$(BLUE)mypy not available, skipping type checks$(NC)"
	@echo "$(GREEN)✅ Type checking completed$(NC)"

# Documentation
docs:
	@echo "$(YELLOW)Building Sphinx documentation...$(NC)"
	sphinx-build -b html docs/source docs/build/html
	@echo "$(GREEN)✅ Documentation built successfully in docs/build/html/$(NC)"

docs-serve: docs
	@echo "$(YELLOW)Serving documentation locally...$(NC)"
	@echo "$(BLUE)Documentation server starting at http://localhost:8000$(NC)"
	@echo "$(BLUE)Press Ctrl+C to stop the server$(NC)"
	cd docs/build/html && python -m http.server 8000

docs-clean:
	@echo "$(YELLOW)Cleaning documentation build files...$(NC)"
	rm -rf docs/build/
	@echo "$(GREEN)✅ Documentation cleaned$(NC)"

docs-linkcheck:
	@echo "$(YELLOW)Checking documentation links...$(NC)"
	sphinx-build -b linkcheck docs/source docs/build/linkcheck
	@echo "$(GREEN)✅ Link check completed$(NC)"

docs-pdf:
	@echo "$(YELLOW)Building PDF documentation...$(NC)"
	sphinx-build -b latex docs/source docs/build/latex
	cd docs/build/latex && pdflatex XRayLabTool.tex
	@echo "$(GREEN)✅ PDF documentation built$(NC)"

# Cleanup
clean:
	@echo "$(YELLOW)Cleaning build artifacts and cache files...$(NC)"
	@echo "$(BLUE)Note: Virtual environments (venv/, env/, .env/) are preserved$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .pytest_cache/
	rm -rf benchmark.json
	rm -rf .benchmarks
	rm -rf .tox/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✅ Cleanup completed (virtual environments preserved)$(NC)"

clean-all: clean docs-clean
	@echo "$(YELLOW)Deep cleaning ALL artifacts including virtual environments...$(NC)"
	@echo "$(RED)WARNING: This will delete virtual environments (venv/, env/, .env/)$(NC)"
	rm -rf venv/ env/ .env/
	rm -rf node_modules/
	rm -rf .DS_Store
	find . -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Deep cleanup completed (all files removed)$(NC)"

# Development Workflows
dev: check-format lint test-fast
	@echo "$(GREEN)✅ Quick development cycle completed$(NC)"

validate: format lint test-coverage test-benchmarks cli-test
	@echo "$(GREEN)✅ Full validation completed - ready for commit!$(NC)"

ci-test: clean install version-check lint type-check test-coverage test-benchmarks cli-test docs
	@echo "$(GREEN)✅ CI simulation completed successfully$(NC)"

release-check: clean dev-setup version-check validate docs docs-linkcheck build
	@echo "$(YELLOW)Pre-release validation checklist:$(NC)"
	@echo "$(BLUE)✓ Code formatted and linted$(NC)"
	@echo "$(BLUE)✓ All tests passing with coverage$(NC)"
	@echo "$(BLUE)✓ CLI functionality verified$(NC)"
	@echo "$(BLUE)✓ Documentation built and links checked$(NC)"
	@echo "$(BLUE)✓ Package built successfully$(NC)"
	@echo "$(BLUE)✓ Version consistency verified$(NC)"
	@echo "$(GREEN)✅ Ready for release!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Update CHANGELOG.md with release notes"
	@echo "  2. Tag release: git tag v$$(python -c 'import xraylabtool; print(xraylabtool.__version__)')"
	@echo "  3. Push tag: git push origin --tags"
	@echo "  4. Upload to PyPI: make upload"

# Performance Monitoring
perf-baseline:
	@echo "$(YELLOW)Creating performance baseline...$(NC)"
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-save=baseline
	@echo "$(GREEN)✅ Baseline saved$(NC)"

perf-compare:
	@echo "$(YELLOW)Comparing against performance baseline...$(NC)"
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-compare=baseline
	@echo "$(GREEN)✅ Performance comparison completed$(NC)"

perf-report:
	@echo "$(YELLOW)Generating performance report...$(NC)"
	pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only --benchmark-json=benchmark_report.json
	@echo "$(GREEN)✅ Performance report saved to benchmark_report.json$(NC)"

# Installation Testing
test-install-local: build
	@echo "$(YELLOW)Testing local wheel installation in clean environment...$(NC)"
	@python -c "
	import sys, subprocess, tempfile, shutil;
	from pathlib import Path;
	wheel_files = list(Path('dist').glob('*.whl'));
	if not wheel_files: print('No wheel files found'); sys.exit(1);
	wheel = wheel_files[0];
	with tempfile.TemporaryDirectory() as tmpdir:
		venv_path = Path(tmpdir) / 'test_venv';
		subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True);
		python_exe = venv_path / 'bin' / 'python' if sys.platform != 'win32' else venv_path / 'Scripts' / 'python.exe';
		subprocess.run([str(python_exe), '-m', 'pip', 'install', str(wheel.absolute())], check=True);
		result = subprocess.run([str(python_exe), '-c', 'import xraylabtool as xlt; result = xlt.calculate_single_material_properties(\"SiO2\", 10.0, 2.2); print(f\"✓ Local install test: {result.critical_angle_degrees[0]:.3f}°\")'], capture_output=True, text=True, check=True);
		print(result.stdout.strip());
	"
	@echo "$(GREEN)✅ Local installation test passed$(NC)"

test-install-testpypi:
	@echo "$(YELLOW)Testing TestPyPI installation in clean environment...$(NC)"
	@python -c "
	import sys, subprocess, tempfile;
	from pathlib import Path;
	with tempfile.TemporaryDirectory() as tmpdir:
		venv_path = Path(tmpdir) / 'test_venv';
		subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True);
		python_exe = venv_path / 'bin' / 'python' if sys.platform != 'win32' else venv_path / 'Scripts' / 'python.exe';
		subprocess.run([str(python_exe), '-m', 'pip', 'install', '--index-url', 'https://test.pypi.org/simple/', '--extra-index-url', 'https://pypi.org/simple/', 'xraylabtool'], check=True);
		result = subprocess.run([str(python_exe), '-c', 'import xraylabtool as xlt; result = xlt.calculate_single_material_properties(\"SiO2\", 10.0, 2.2); print(f\"✓ TestPyPI install test: {result.critical_angle_degrees[0]:.3f}°\")'], capture_output=True, text=True, check=True);
		print(result.stdout.strip());
	"
	@echo "$(GREEN)✅ TestPyPI installation test passed$(NC)"

test-install-pypi:
	@echo "$(YELLOW)Testing PyPI installation in clean environment...$(NC)"
	@python -c "
	import sys, subprocess, tempfile;
	from pathlib import Path;
	with tempfile.TemporaryDirectory() as tmpdir:
		venv_path = Path(tmpdir) / 'test_venv';
		subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True);
		python_exe = venv_path / 'bin' / 'python' if sys.platform != 'win32' else venv_path / 'Scripts' / 'python.exe';
		subprocess.run([str(python_exe), '-m', 'pip', 'install', 'xraylabtool'], check=True);
		result = subprocess.run([str(python_exe), '-c', 'import xraylabtool as xlt; result = xlt.calculate_single_material_properties(\"SiO2\", 10.0, 2.2); print(f\"✓ PyPI install test: {result.critical_angle_degrees[0]:.3f}°\")'], capture_output=True, text=True, check=True);
		print(result.stdout.strip());
	"
	@echo "$(GREEN)✅ PyPI installation test passed$(NC)"

# Package Building & Release
build: clean
	@echo "$(YELLOW)Building distribution packages...$(NC)"
	python3 -m build
	@echo "$(GREEN)✅ Packages built in dist/$(NC)"
	@ls -la dist/

upload-test: build
	@echo "$(YELLOW)Uploading to TestPyPI...$(NC)"
	python -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✅ Uploaded to TestPyPI$(NC)"
	@echo "$(BLUE)Test installation: pip install -i https://test.pypi.org/simple/ xraylabtool$(NC)"

upload: build
	@echo "$(YELLOW)Uploading to PyPI...$(NC)"
	@echo "$(RED)WARNING: This will upload to production PyPI!$(NC)"
	@read -p "Are you sure? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	python -m twine upload dist/*
	@echo "$(GREEN)✅ Uploaded to PyPI$(NC)"

# Utility Targets
status:
	@echo "$(BLUE)XRayLabTool Project Status:$(NC)"
	@echo "$(YELLOW)Version:$(NC) $$(python -c 'import xraylabtool; print(xraylabtool.__version__)')"
	@echo "$(YELLOW)Python:$(NC) $$(python --version)"
	@echo "$(YELLOW)Location:$(NC) $$(pwd)"
	@echo "$(YELLOW)Git branch:$(NC) $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "$(YELLOW)Git status:$(NC)"
	@git status --porcelain 2>/dev/null || echo "Not a git repository"

info:
	@echo "$(BLUE)XRayLabTool Package Information:$(NC)"
	@python -c "
	import xraylabtool as xlt;
	import sys;
	print(f'Package: XRayLabTool v{xlt.__version__}');
	print(f'Python: {sys.version}');
	result = xlt.calculate_single_material_properties('SiO2', 10.0, 2.2);
	print(f'API Test: ✓ Critical angle = {result.critical_angle_degrees[0]:.3f}°');
	"
	@echo "$(YELLOW)CLI Test:$(NC)"
	@xraylabtool --version

quick-test:
	@echo "$(YELLOW)Quick functionality test...$(NC)"
	@python -c "import xraylabtool as xlt; result = xlt.calculate_single_material_properties('SiO2', 10.0, 2.2); print(f'✓ Python API: {result.critical_angle_degrees[0]:.3f}°')"
	@xraylabtool calc SiO2 -e 10.0 -d 2.2 --fields critical_angle_degrees | grep -E "Critical|SiO2" | head -2
	@echo "$(GREEN)✅ Quick test passed$(NC)"
