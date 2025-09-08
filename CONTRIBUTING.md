# Contributing to XRayLabTool

Thank you for your interest in contributing to XRayLabTool! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- **Python ≥ 3.12**
- **Git** for version control
- **Virtual environment** (recommended)

### Areas for Contribution

We welcome contributions in several areas:

- 🐛 **Bug fixes** - Fix issues or improve error handling
- ✨ **New features** - Add new X-ray analysis capabilities
- 📚 **Documentation** - Improve guides, examples, or API docs
- 🚀 **Performance** - Optimize calculations or memory usage
- 🧪 **Testing** - Add test coverage or improve test quality
- 🔧 **Infrastructure** - CI/CD, build tools, or development workflow

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/pyXRayLabTool.git
cd pyXRayLabTool
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode with all dependencies
pip install -e .[dev,docs,perf]

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests to ensure setup is correct
python -m pytest tests/ -v

# Check CLI works
xraylabtool --version

# Build documentation
sphinx-build -b html docs/source docs/_build/html
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Changes

- Follow the [code standards](#code-standards) below
- Add tests for new functionality
- Update documentation as needed
- Run tests frequently during development

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add feature: brief description

Detailed explanation of changes and why they were made.

Fixes #issue-number"
```

## Code Standards

### Code Formatting

We use automated code formatting tools:

- **[Black](https://black.readthedocs.io/)** for code formatting
- **[isort](https://pycqa.github.io/isort/)** for import sorting
- **[Ruff](https://docs.astral.sh/ruff/)** for fast linting

```bash
# Format code (done automatically by pre-commit)
black xraylabtool/ tests/
isort xraylabtool/ tests/
ruff check xraylabtool/ tests/
```

### Type Hints

- Use type hints for all public APIs
- Use `from typing import ...` for type annotations
- Document complex types in docstrings

```python
from typing import Dict, List, Optional, Union
import numpy as np

def calculate_properties(
    formula: str,
    energies: Union[float, np.ndarray],
    density: float
) -> XRayResult:
    """Calculate X-ray properties for a material."""
    pass
```

### Docstring Standards

Use **NumPy-style docstrings** for consistency:

```python
def example_function(param1: str, param2: float = 1.0) -> bool:
    """
    Brief description of the function.

    Longer description explaining the function's purpose,
    behavior, and any important implementation details.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : float, optional
        Description of param2, by default 1.0

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When invalid input is provided

    Examples
    --------
    >>> example_function("hello", 2.0)
    True

    Notes
    -----
    Additional notes about implementation or usage.
    """
```

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add comments for complex algorithms
- Group related functionality in modules
- Follow existing code patterns and structure

## Testing

### Test Suite Overview

We have comprehensive test coverage with **388 tests** across multiple categories:

- **Unit tests** - Core functionality (atomic data, physics, parsing)
- **Integration tests** - Cross-module functionality and CLI
- **Performance tests** - Benchmarks and optimization validation
- **Edge case tests** - Boundary conditions and error handling

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python run_tests.py --phase unit
python run_tests.py --phase integration
python run_tests.py --phase performance

# Run with coverage
python -m pytest --cov=xraylabtool --cov-report=html

# Run specific test file
python -m pytest tests/test_core.py -v
```

### Writing Tests

- **Add tests for all new functionality**
- Use descriptive test names: `test_calculate_properties_with_invalid_formula`
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use fixtures for common test data

```python
import pytest
import numpy as np
import xraylabtool as xlt

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        result = xlt.your_new_function("SiO2", 10.0)
        assert result is not None
        assert isinstance(result, expected_type)

    def test_error_handling(self):
        """Test that errors are handled appropriately."""
        with pytest.raises(xlt.ValidationError):
            xlt.your_new_function("", -1.0)
```

### Performance Testing

For performance-critical changes:

- Add benchmarks to `tests/test_performance_benchmarks.py`
- Ensure no significant performance regression
- Document performance improvements in PR

## Documentation

### Types of Documentation

1. **API Documentation** - Docstrings in code
2. **User Guides** - Sphinx documentation in `docs/source/`
3. **Examples** - Working code examples
4. **CLI Reference** - Command-line documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation
sphinx-build -b html docs/source docs/_build/html

# Check for broken links
sphinx-build -b linkcheck docs/source docs/_build/linkcheck

# Serve locally for testing
python -m http.server 8000 -d docs/_build/html
```

### Documentation Guidelines

- Update documentation for API changes
- Add examples for new features
- Keep documentation accurate and up-to-date
- Use clear, concise language
- Include code examples that work

## Submitting Changes

### Before Submitting

1. **Run the full test suite** and ensure all tests pass
2. **Check code formatting** with pre-commit hooks
3. **Update documentation** for any API changes
4. **Update CHANGELOG.md** following [Keep a Changelog](https://keepachangelog.com/) format
5. **Ensure type checking passes** with mypy

```bash
# Final checks before submitting
python run_tests.py                # Run all tests
pre-commit run --all-files         # Check formatting
mypy xraylabtool/                   # Type checking
sphinx-build -b html docs/source docs/_build/html  # Build docs
```

### Pull Request Process

1. **Push your branch** to your fork on GitHub
2. **Create Pull Request** with descriptive title and description
3. **Fill out PR template** with:
   - Summary of changes
   - Testing performed
   - Documentation updates
   - Breaking changes (if any)
4. **Respond to reviews** and make requested changes
5. **Ensure CI passes** - all automated checks must pass

### PR Requirements

- ✅ All tests pass
- ✅ Code follows formatting standards
- ✅ Documentation is updated
- ✅ CHANGELOG.md is updated
- ✅ No merge conflicts with main branch

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major** (x.0.0) - Breaking changes
- **Minor** (0.x.0) - New features, backwards compatible
- **Patch** (0.0.x) - Bug fixes, backwards compatible

### Maintainer Release Steps

1. Update version in `pyproject.toml` and `xraylabtool/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create release tag: `git tag v0.x.y`
4. Push tag: `git push origin v0.x.y`
5. GitHub Actions automatically builds and publishes to PyPI
6. Create GitHub release with changelog

## Getting Help

### Communication Channels

- 🐛 **Issues** - Bug reports and feature requests
- 💬 **Discussions** - Questions and general discussion
- 📧 **Email** - Wei Chen (wchen@anl.gov) for sensitive issues

### Resources

- 📖 **Documentation** - https://pyxraylabtool.readthedocs.io/
- 🔬 **Examples** - See `docs/source/examples.rst`
- 🚀 **Performance Guide** - See `docs/source/performance_guide.rst`
- ⚡ **CLI Reference** - See `CLI_REFERENCE.md`

## Recognition

Contributors are recognized in:

- 🎉 **CHANGELOG.md** - All contributors listed for each release
- 👥 **GitHub Contributors** - Automatic recognition on repository
- 📦 **PyPI Acknowledgments** - Major contributors acknowledged

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful in all interactions and follow these principles:

- **Be Respectful** - Treat everyone with respect, regardless of background or experience level
- **Be Collaborative** - Work together constructively and share knowledge openly
- **Be Patient** - Help newcomers learn and grow within our community
- **Be Professional** - Maintain professional standards in all communications

### Scientific Integrity

As a scientific computing tool, we maintain high standards for:

- **Data Accuracy** - All calculations must be physically and mathematically correct
- **Source Attribution** - Properly cite scientific sources and databases (CXRO, NIST)
- **Reproducible Results** - Ensure all features produce consistent, reproducible results
- **Peer Review** - Scientific changes undergo additional review by domain experts

### Community Resources

- 📚 **Documentation** - https://pyxraylabtool.readthedocs.io/
- 💬 **Discussions** - GitHub Discussions for questions and ideas
- 🐛 **Issues** - GitHub Issues for bug reports and feature requests
- 🔬 **Examples** - Interactive notebooks and scientific workflows
- 📧 **Contact** - Wei Chen (wchen@anl.gov) for project leadership

### Contributor Levels

**New Contributors**:
- Start with documentation improvements or small bug fixes
- Join discussions to understand the project goals
- Review existing issues labeled "good first issue"

**Regular Contributors**:
- Work on feature development and significant improvements
- Participate in code reviews and architectural discussions
- Help mentor new contributors

**Core Contributors**:
- Maintain project infrastructure and CI/CD systems
- Make architectural decisions and set project direction
- Coordinate releases and manage project roadmap

### Scientific Workflow Examples

For scientific software, consider these types of contributions:

- **Physics Validation** - Verify calculations against known standards
- **Performance Benchmarks** - Optimize for large-scale scientific computing
- **Use Case Examples** - Real-world synchrotron and laboratory examples
- **Integration Tools** - Connect with other scientific Python packages
- **Educational Content** - Tutorials for students and researchers

## Getting Support

### Before Asking for Help

1. **Check Documentation** - Search our comprehensive documentation
2. **Review Issues** - Look for existing solutions or discussions
3. **Try Examples** - Run through provided examples and tutorials
4. **Check Stack Overflow** - Search for related Python/NumPy questions

### Where to Get Help

- **GitHub Discussions** - Best for usage questions and community discussion
- **GitHub Issues** - For confirmed bugs or feature requests
- **Email** - Direct contact for sensitive issues or collaboration inquiries
- **Documentation** - Comprehensive guides and API reference

### Response Expectations

- **Issues** - We aim to respond within 48-72 hours
- **Pull Requests** - Initial review within 1 week
- **Security Reports** - Acknowledgment within 48 hours
- **General Questions** - Community discussions typically have faster responses

---

Thank you for contributing to XRayLabTool! Your contributions help advance X-ray analysis tools for the scientific community. Together, we're building better tools for synchrotron science, materials research, and X-ray optics development. 🔬✨
