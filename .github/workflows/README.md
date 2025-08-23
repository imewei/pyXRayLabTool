# GitHub Actions Workflows

This directory contains automated workflows for continuous integration, testing, and publishing.

## Workflows Overview

### `ci.yml` - Main CI/CD Pipeline
**Triggers:** Push to main/develop, pull requests, releases

**Jobs:**
- **test**: Cross-platform testing on Ubuntu, macOS, Windows with Python 3.12 & 3.13
  - Linting with flake8
  - Code formatting check with black
  - Type checking with mypy
  - Unit tests with pytest
  - CLI functionality tests
  - Coverage reporting to Codecov

- **benchmark**: Performance benchmarks with pytest-benchmark
  - Tracks performance regressions
  - Stores benchmark history

- **build**: Package building and validation
  - Creates wheel and source distributions
  - Validates package integrity with twine

- **publish-test**: Automatic publishing to Test PyPI (develop branch)
- **publish**: Automatic publishing to PyPI (releases)

### `publish.yml` - Manual Publishing Workflow
**Triggers:** Manual dispatch, releases

**Features:**
- Manual publishing to Test PyPI or production PyPI
- Full test suite execution before publishing
- Package integrity checks

## Setup Requirements

### Repository Secrets
Configure these secrets in your GitHub repository settings:

- `PYPI_API_TOKEN`: PyPI API token for production publishing
- `TEST_PYPI_API_TOKEN`: Test PyPI API token for testing

### Environments (Optional but Recommended)
Create GitHub environments for additional protection:
- `pypi`: Production publishing environment
- `testpypi`: Test publishing environment

## Workflow Triggers

### Automatic Triggers
- **Push to main/develop**: Full CI pipeline
- **Pull requests**: Full CI pipeline (no publishing)
- **GitHub releases**: Full CI + production PyPI publishing

### Manual Triggers
- Use "Actions" tab → "Publish to PyPI" → "Run workflow"
- Choose "test" or "prod" target

## Development Workflow

1. **Feature development**: Create feature branch, make changes
2. **Pull request**: Triggers full CI pipeline
3. **Merge to develop**: Triggers CI + Test PyPI publishing
4. **Create release**: Triggers CI + production PyPI publishing

## Local Testing

Simulate CI locally using Make commands:
```bash
make validate    # Full validation (lint + format + tests)
make ci-test     # CI simulation
make build       # Package building
```

## Monitoring

- **Test results**: View in Actions tab
- **Coverage**: Check Codecov reports  
- **Benchmarks**: Track performance trends
- **PyPI packages**: Monitor at https://pypi.org/project/xraylabtool/
