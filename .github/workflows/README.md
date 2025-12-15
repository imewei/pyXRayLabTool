# GitHub Workflows Documentation

This directory contains GitHub Actions workflows for the pyXRayLabTool project.

## Workflows Overview

### 1. Continuous Integration (`ci.yml`)
**Triggers:** Push/PR to main/develop, manual dispatch

Ultra-optimized CI pipeline with intelligent execution:

- **Smart Change Detection**: Only runs full pipeline when necessary
- **Ultra-Fast Linting**: Ruff + MyPy for code quality
- **Intelligent Testing**: Smart test selection based on file changes
- **Build Verification**: Package building and integrity validation

**Key Features:**
- ⚡ **Fast Feedback**: Results in 3-8 minutes
- 🧠 **Smart Execution**: Conditional matrix expansion based on changes
- 🔄 **Advanced Caching**: Multi-layer dependency caching
- 📊 **Comprehensive Reporting**: Detailed status and performance metrics

### 2. Documentation (`docs.yml`)
**Triggers:** Push/PR affecting docs or code, manual dispatch

Comprehensive documentation pipeline:

- **Build Validation**: Sphinx documentation building with error handling
- **Example Testing**: Doctest execution and README code validation
- **Quality Checks**: RST syntax and style validation
- **Link Checking**: External link validation (main branch only)

### 3. Release Automation (`release.yml`)
**Triggers:** Manual workflow dispatch

Fully automated release pipeline:

- **Version Management**: Automated version bumping and validation
- **Asset Building**: Source and wheel distribution with checksums
- **GitHub Release**: Automated release creation with detailed notes
- **PyPI Publishing**: Trusted publishing to PyPI

### 4. Dependency Management (`dependencies.yml`)
**Triggers:** Weekly schedule, manual dispatch

Proactive dependency lifecycle management:

- **Security Auditing**: Vulnerability scanning and reporting
- **Update Detection**: Automated dependency update identification
- **Automated PRs**: Dependency update pull requests

## Workflow Configuration

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided (no setup needed)
- `PYPI_API_TOKEN`: PyPI trusted publishing token for releases

### Branch Protection Rules
Configure branch protection on `main` branch with:
- Require status checks: `📊 Status & Performance Report` from ci.yml
- Require up-to-date branches

## Usage Examples

### Creating a Release
Navigate to Actions → Release Automation → Run workflow, then select version and type.

### Manual Dependency Updates
Navigate to Actions → Dependency Management → Run workflow.

## Typical Pipeline Times
- **Documentation Changes**: < 2 minutes (skipped CI)
- **Code Changes**: 3-8 minutes (optimized CI)
- **Full Matrix**: 8-12 minutes (when needed)
