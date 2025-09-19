# GitHub Workflows Documentation

This directory contains streamlined GitHub Actions workflows for the pyXRayLabTool project, providing automated CI/CD, security, quality assurance, and dependency management.

## Workflows Overview

### 1. Continuous Integration (`ci.yml`)
**Triggers:** Push/PR to main/develop, manual dispatch

Ultra-optimized CI pipeline with intelligent execution:

- **Smart Change Detection**: Only runs full pipeline when necessary
- **Ultra-Fast Linting**: Ruff, Black, isort, MyPy for code quality
- **Intelligent Testing**: Smart test selection based on file changes
- **Build Verification**: Package building and integrity validation
- **Performance Optimizations**: Uses uv for ultra-fast package installation

**Key Features:**
- ⚡ **Fast Feedback**: Results in 3-8 minutes
- 🧠 **Smart Execution**: Conditional matrix expansion based on changes
- 🔄 **Advanced Caching**: Multi-layer dependency caching
- 🚫 **Fail-Fast**: Immediate feedback on failures
- 📊 **Comprehensive Reporting**: Detailed status and performance metrics

### 2. Security Scanning (`security.yml`)
**Triggers:** Push/PR to main/develop, daily schedule

Comprehensive security analysis separate from CI for focused security coverage:

- **Dependency Security**: Safety and pip-audit vulnerability scanning
- **Static Analysis**: Bandit SAST and Semgrep pattern analysis
- **Supply Chain**: Trivy vulnerability scanner with SARIF upload

**Features:**
- Automated SARIF report integration with GitHub Security tab
- Runs independently from CI for focused security analysis
- Comprehensive security metrics collection
- Can be configured as required check for branch protection

### 3. Documentation (`docs.yml`)
**Triggers:** Push/PR affecting docs or code, manual dispatch

Comprehensive documentation pipeline:

- **Build Validation**: Sphinx documentation building with error handling
- **Example Testing**: Doctest execution and README code validation
- **Quality Checks**: RST syntax and style validation
- **Link Checking**: External link validation (main branch only)

**Features:**
- Enhanced error handling and partial build support
- Comprehensive documentation statistics
- Automated quality checks

### 4. Release Automation (`release.yml`)
**Triggers:** Manual workflow dispatch

Fully automated release pipeline:

- **Version Management**: Automated version bumping and validation
- **Asset Building**: Source and wheel distribution with checksums
- **GitHub Release**: Automated release creation with detailed notes
- **PyPI Publishing**: Trusted publishing to PyPI

### 5. Dependency Management (`dependencies.yml`)
**Triggers:** Weekly schedule, manual dispatch

Proactive dependency lifecycle management:

- **Security Auditing**: Vulnerability scanning and reporting
- **Update Detection**: Automated dependency update identification
- **Smart Updates**: Configurable update strategies
- **Automated PRs**: Dependency update pull requests

### 6. Performance Monitoring (`performance-monitoring.yml`)
**Triggers:** Daily schedule

CI/CD pipeline performance analysis:

- **Metrics Collection**: Success rates, duration trends, cache efficiency
- **Performance Analysis**: Identifies regressions and optimization opportunities
- **Health Monitoring**: Validates workflow effectiveness


## Workflow Configuration

### Required Secrets
- `GITHUB_TOKEN`: Automatically provided (no setup needed)
- `PYPI_API_TOKEN`: PyPI trusted publishing token for releases
- `CODECOV_TOKEN`: Code coverage reporting (optional)

### Branch Protection Rules
Configure branch protection on `main` branch with:
- Require status checks: `Status & Performance Report` from ci.yml
- Require up-to-date branches
- Include administrators

### Security Configuration
- Enable Dependabot alerts and security updates
- Enable secret scanning and push protection
- Set up SARIF upload permissions for security workflows

## Usage Examples

### Creating a Release
```bash
# Navigate to Actions → Release Automation → Run workflow
# Select version and type as needed
```

### Manual Dependency Updates
```bash
# Navigate to Actions → Dependency Management → Run workflow
# Configure update strategy and PR creation
```

### Checking Security Status
Security scans run automatically, view results in:
- Actions → Security Scanning → Latest run
- Security tab → Code scanning alerts
- Security tab → Dependabot alerts

## Development Workflow

1. **Feature Development**: Create feature branch, implement changes
2. **Pull Request**: Triggers optimized CI pipeline with smart change detection
3. **Code Review**: Automated quality checks provide fast feedback
4. **Merge to Main**: Full pipeline execution with comprehensive testing
5. **Release Creation**: Use automated release workflow for consistent releases
6. **Dependency Updates**: Weekly automated dependency maintenance

## Performance Benefits

### Optimization Features
- **Smart Change Detection**: Skips unnecessary work when only docs change
- **Conditional Matrix**: Expands test matrix only when needed
- **Ultra-fast Package Installation**: Uses uv for 40-60% faster builds
- **Advanced Caching**: Multi-layer caching strategy for maximum efficiency
- **Intelligent Test Selection**: Runs only affected tests for faster feedback

### Typical Pipeline Times
- **Documentation Changes**: < 2 minutes (skipped CI)
- **Code Changes**: 3-8 minutes (optimized CI)
- **Full Matrix**: 8-12 minutes (when needed)

## Monitoring and Maintenance

### Performance Monitoring
The `performance-monitoring.yml` workflow runs daily to track:
- Success rates and failure patterns
- Pipeline duration trends
- Cache effectiveness
- Optimization opportunities

### Maintenance Best Practices
- Monitor performance metrics weekly
- Review security scan results promptly
- Keep workflow dependencies updated
- Optimize caching strategies based on metrics
