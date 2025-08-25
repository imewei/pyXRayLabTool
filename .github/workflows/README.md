# GitHub Workflows Documentation

This directory contains comprehensive GitHub Actions workflows for the pyXRayLabTool project, providing automated CI/CD, security, quality assurance, and dependency management.

## Workflows Overview

### 1. Continuous Integration (`ci.yml`)
**Triggers:** Push/PR to main/develop, weekly schedule

A comprehensive CI pipeline with multiple parallel jobs:

- **Code Quality & Linting**: Pre-commit hooks, Black formatting, flake8 linting, MyPy type checking, Bandit security, Safety vulnerability scanning
- **Test Suite**: Matrix testing across Ubuntu/macOS/Windows with Python 3.12/3.13
- **Performance Benchmarks**: Automated performance regression testing
- **Build Verification**: Package building and integrity validation
- **Integration Testing**: End-to-end CLI and import testing
- **Status Reporting**: Consolidated pass/fail status for branch protection

**Features:**
- Intelligent caching for pip packages and pre-commit hooks
- Parallel job execution with proper dependencies
- Comprehensive test coverage reporting
- Artifact collection for debugging
- Memory-efficient execution with automatic cleanup

### 2. Security & Code Quality (`security.yml`)
**Triggers:** Push/PR to main/develop, daily schedule

Multi-layered security and quality analysis:

- **Security Scanning**: Bandit SAST, Safety vulnerability checks, Semgrep pattern analysis
- **Dependency Review**: Automated dependency security analysis for PRs
- **CodeQL Analysis**: GitHub's semantic code analysis for vulnerabilities
- **License Compliance**: Automated license compatibility checking
- **Code Quality Metrics**: Complexity analysis, maintainability scoring, dead code detection
- **Secrets Scanning**: TruffleHog for credential leak detection

**Features:**
- SARIF report integration with GitHub Security tab
- Configurable security severity thresholds
- License policy enforcement
- Comprehensive quality metrics collection

### 3. Release Automation (`release.yml`)
**Triggers:** Manual workflow dispatch

Fully automated release pipeline with validation:

- **Version Validation**: Semantic version parsing and prerelease detection
- **Pre-Release Testing**: Complete CI suite execution (optional skip for emergencies)
- **Version Updates**: Automatic version bumping in pyproject.toml and __init__.py
- **Changelog Generation**: Automated changelog entry creation
- **Asset Building**: Source and wheel distribution building with checksums
- **GitHub Release**: Automated GitHub release with comprehensive notes
- **PyPI Publishing**: Trusted publishing to PyPI with verification
- **Post-Release Tasks**: Issue creation for follow-up tasks and notifications

**Features:**
- Support for patch/minor/major/rc/beta release types
- Draft release option for final review
- Emergency release path with test skipping
- Comprehensive release notes with checksums
- Automated post-release checklist creation

### 4. Dependency Management (`dependencies.yml`)
**Triggers:** Weekly schedule, manual workflow dispatch

Proactive dependency lifecycle management:

- **Security Auditing**: Safety and pip-audit vulnerability scanning
- **Update Detection**: Multi-tool outdated package identification
- **Smart Updates**: Configurable update strategies (patch/minor/major)
- **Automated Testing**: Dependency compatibility validation
- **Pull Request Creation**: Automated dependency update PRs
- **Dependency Analysis**: Comprehensive dependency tree and SBOM generation

**Features:**
- Intelligent update filtering based on semantic versioning
- Automated PR creation with detailed change summaries
- Security-first update prioritization
- Comprehensive dependency documentation
- Integration with GitHub's dependency graph

### 5. Legacy Workflows

#### Test Suite (`test.yml`)
Simple matrix testing workflow (superseded by ci.yml but maintained for compatibility)

#### PyPI Publishing (`publish.yml`)
Basic publishing workflow (enhanced by release.yml automation)

## Workflow Configuration

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided (no setup needed)
- `PYPI_API_TOKEN`: PyPI trusted publishing token for releases
- `TEST_PYPI_API_TOKEN`: Test PyPI token for development releases
- `CODECOV_TOKEN`: Code coverage reporting (optional)
- `SEMGREP_APP_TOKEN`: Enhanced Semgrep scanning (optional)

### Branch Protection Rules
Configure branch protection on `main` branch with:
- Require status checks: `Status Check` from ci.yml
- Require up-to-date branches
- Require linear history
- Include administrators

### Security Configuration
- Enable Dependabot alerts and security updates
- Configure CodeQL analysis for Python
- Enable secret scanning and push protection
- Set up SARIF upload permissions

## Usage Examples

### Creating a Release
```bash
# Navigate to Actions → Release Automation → Run workflow
# Select version: 0.2.0
# Choose type: minor
# Create draft: false
```

### Manual Dependency Updates
```bash
# Navigate to Actions → Dependency Management → Run workflow
# Update type: minor
# Create PR: true
```

### Checking Security Status
All security scans run automatically, view results in:
- Actions → Security & Code Quality → Latest run
- Security tab → Code scanning alerts
- Security tab → Dependabot alerts

## Development Workflow

1. **Feature Development**: Create feature branch, implement changes
2. **Pull Request**: Triggers comprehensive CI pipeline with security scans
3. **Code Review**: Automated quality checks assist human reviewers
4. **Merge to Main**: Full pipeline execution with artifact generation
5. **Release Creation**: Use automated release workflow for consistent releases
6. **Dependency Updates**: Weekly automated PRs for dependency maintenance

## Monitoring and Maintenance

### Key Metrics to Monitor
- CI success rate and duration trends
- Security vulnerability count and resolution time
- Dependency update frequency and success rate
- Test coverage percentage and trends
- Build performance and artifact sizes

### Regular Maintenance Tasks
- Review and update workflow dependencies monthly
- Audit security scan results weekly
- Monitor dependency update PRs for breaking changes
- Update workflow documentation after significant changes
- Review and optimize caching strategies quarterly
