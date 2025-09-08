# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

The XRayLabTool team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

### How to Report Security Issues

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**Security Contact**: Wei Chen at `wchen@anl.gov`

Please include the following information in your report:

- **Type of issue** (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths of source file(s)** related to the manifestation of the issue
- **The location of the affected source code** (tag/branch/commit or direct URL)
- **Any special configuration required** to reproduce the issue
- **Step-by-step instructions to reproduce the issue**
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the issue**, including how an attacker might exploit the issue

### What to Expect

After you submit a report, we will:

1. **Acknowledge receipt** of your vulnerability report within 48 hours
2. **Provide an initial assessment** of the vulnerability within 5 business days
3. **Work with you to understand and resolve the issue** as quickly as possible
4. **Keep you informed** about the progress toward a fix and full announcement
5. **Credit you** in the security advisory (if you wish)

### Security Update Process

1. **Vulnerability Assessment** - We assess the severity and impact
2. **Fix Development** - We develop and test a fix
3. **Security Advisory** - We prepare a security advisory
4. **Coordinated Release** - We release the fix and advisory simultaneously
5. **Public Disclosure** - We announce the vulnerability after users have had time to update

## Security Best Practices for Users

### Installation Security

- **Use official sources**: Install only from PyPI (`pip install xraylabtool`) or official GitHub releases
- **Verify integrity**: Check package hashes when possible
- **Keep updated**: Regularly update to the latest version to get security patches

### Usage Security

- **Input validation**: Validate chemical formulas and numerical inputs in your applications
- **File permissions**: Be cautious when processing batch files from untrusted sources
- **Resource limits**: Set appropriate limits for energy arrays and batch sizes to prevent resource exhaustion

### Development Security

If you're contributing to XRayLabTool:

- **Dependency scanning**: We use automated tools to scan for vulnerable dependencies
- **Code review**: All changes undergo security-focused code review
- **Static analysis**: Our CI pipeline includes security static analysis
- **Regular updates**: We maintain up-to-date dependencies

## Known Security Considerations

### Scientific Computing Context

XRayLabTool is designed for scientific computing with trusted data. However, be aware of:

- **NumPy array inputs**: Large arrays can consume significant memory
- **File operations**: Batch processing involves file I/O operations
- **Mathematical calculations**: Some calculations may be sensitive to numerical precision

### No Known Vulnerabilities

As of the latest release, there are no known security vulnerabilities in XRayLabTool.

## Security-Related Dependencies

We monitor security advisories for our key dependencies:

- **NumPy** - Mathematical operations
- **SciPy** - Scientific computing functions
- **Pandas** - Data manipulation
- **Matplotlib** - Plotting capabilities

## Contact Information

- **Primary Contact**: Wei Chen (`wchen@anl.gov`)
- **GitHub Issues**: For non-security related bugs only
- **Documentation**: https://pyxraylabtool.readthedocs.io/

## Disclosure Policy

When we receive a security bug report, we will:

- Confirm the problem and determine the affected versions
- Audit code to find any similar problems
- Prepare fixes for all supported versions
- Release new versions with fixes as soon as possible

We aim to disclose security vulnerabilities within 90 days of initial report, or sooner if possible.

---

Thank you for helping keep XRayLabTool and the scientific computing community safe!
