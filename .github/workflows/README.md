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

### 3. Publish to PyPI (`publish.yml`)
**Triggers:** Manual workflow dispatch (takes an existing git tag)

Builds and publishes an already-tagged release to PyPI:

- **Asset Building**: Source and wheel distribution
- **Verification**: `twine check` on built distributions
- **PyPI Publishing**: Trusted publishing to PyPI

`release.yml` (the previous fully-automated version/changelog/tag/GitHub-release/PyPI
pipeline) was removed after repeated failures: a crashing reusable-workflow
call, a changelog-corrupting step, and a malformed version-format regex.
Bumping the version, updating `CHANGELOG.md`, tagging, and creating the
GitHub Release are now done by hand — see "Creating a Release" below; this
workflow only automates the PyPI half.

### 4. Dependency Management (`dependencies.yml`)
**Triggers:** Weekly schedule, manual dispatch

Proactive dependency lifecycle management:

- **Security Auditing**: Vulnerability scanning and reporting
- **Update Detection**: Automated dependency update identification
- **Automated PRs**: Dependency update pull requests

## Workflow Configuration

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided (no setup needed)
- PyPI publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) via the `pypi` environment — no stored token needed. The trusted publisher on PyPI must be registered for the exact workflow filename (`publish.yml`).

### Branch Protection Rules
Configure branch protection on `main` branch with:
- Require status checks: `📊 Status & Performance Report` from ci.yml
- Require up-to-date branches

## Usage Examples

### Creating a Release
1. Bump the version in `pyproject.toml` and `xraylabtool/__init__.py`, and add a
   dated entry to `CHANGELOG.md` (and `docs/development/changelog.rst`). Commit.
2. Wait for CI to pass on that commit.
3. Tag it (`git tag -a vX.Y.Z -m "Release X.Y.Z"`) and push the tag.
4. Create the GitHub Release from that tag (e.g. `gh release create vX.Y.Z --title "Release X.Y.Z" --notes-from-tag`, or paste the CHANGELOG entry as the notes).
5. Navigate to Actions → Publish to PyPI → Run workflow, enter the tag (e.g. `vX.Y.Z`).

### Manual Dependency Updates
Navigate to Actions → Dependency Management → Run workflow.

## Typical Pipeline Times
- **Documentation Changes**: < 2 minutes (skipped CI)
- **Code Changes**: 3-8 minutes (optimized CI)
- **Full Matrix**: 8-12 minutes (when needed)
