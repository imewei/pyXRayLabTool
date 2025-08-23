#!/usr/bin/env python3
"""
Setup script for XRayLabTool package.

This setup.py provides backward compatibility and additional configuration
options for PyPI publishing. The primary package configuration is in
pyproject.toml, following modern Python packaging standards.
"""

import re
from pathlib import Path
from setuptools import setup, find_packages

# Package metadata
PACKAGE_NAME = "xraylabtool"
VERSION = "0.1.5"
AUTHOR = "Wei Chen"
AUTHOR_EMAIL = "wchen@anl.gov"
DESCRIPTION = (
    "High-Performance X-ray Optical Properties Calculator with CLI - "
    "Python package and command-line tool for calculating X-ray optical "
    "properties of materials based on chemical formulas and densities"
)
URL = "https://github.com/imewei/pyXRayLabTool"
LICENSE = "MIT"
PYTHON_REQUIRES = ">=3.12"


# Read long description from README
def read_long_description():
    """Read the long description from README.md."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Remove badges from description for PyPI
        content = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content)
        return content
    return DESCRIPTION


# Core dependencies (production requirements)
INSTALL_REQUIRES = [
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "mendeleev>=0.10.0",
    "tqdm>=4.60.0",
    "matplotlib>=3.4.0",
]

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "pytest-benchmark>=3.4.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.900",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "sphinxcontrib-napoleon>=0.7",
    ],
    "test": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "pytest-benchmark>=3.4.0",
    ],
    "lint": [
        "black>=21.0.0",
        "flake8>=3.9.0",
        "mypy>=0.900",
    ],
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE["all"] = list(
    set(dep for extra_deps in EXTRAS_REQUIRE.values() for dep in extra_deps)
)

# Package classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Environment :: Console",
    "Topic :: System :: Shells",
    "Topic :: Utilities",
    "Natural Language :: English",
]

# Keywords for discoverability
KEYWORDS = [
    "xray",
    "crystallography",
    "diffraction",
    "scattering",
    "laboratory",
    "synchrotron",
    "optics",
    "materials",
    "cxro",
    "nist",
    "cli",
    "command-line",
    "batch-processing",
    "analysis",
    "physics",
    "chemistry",
]

# Project URLs
PROJECT_URLS = {
    "Homepage": "https://github.com/imewei/pyXRayLabTool",
    "Documentation": "https://pyxraylabtool.readthedocs.io",
    "Repository": "https://github.com/imewei/pyXRayLabTool.git",
    "Bug Reports": "https://github.com/imewei/pyXRayLabTool/issues",
    "Issues": "https://github.com/imewei/pyXRayLabTool/issues",
    "Changelog": "https://github.com/imewei/pyXRayLabTool/blob/main/CHANGELOG.md",
    "CLI Reference": (
        "https://github.com/imewei/pyXRayLabTool/blob/main/CLI_REFERENCE.md"
    ),
    "Source": "https://github.com/imewei/pyXRayLabTool",
}

# Entry points for command-line scripts
ENTRY_POINTS = {
    "console_scripts": [
        "xraylabtool=xraylabtool.cli:main",
    ],
}

# Package data files
PACKAGE_DATA = {
    "xraylabtool": [
        "data/AtomicScatteringFactor/*.nff",
    ],
}

# Data files to include in distribution
INCLUDE_PACKAGE_DATA = True

# Exclude test files from distribution
EXCLUDE = ["tests*", "docs*", "benchmarks*"]


def main():
    """Main setup function."""
    setup(
        # Basic package information
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_long_description(),
        long_description_content_type="text/markdown",
        # URLs and metadata
        url=URL,
        project_urls=PROJECT_URLS,
        license=LICENSE,
        # Package discovery and inclusion
        packages=find_packages(exclude=EXCLUDE),
        package_data=PACKAGE_DATA,
        include_package_data=INCLUDE_PACKAGE_DATA,
        # Requirements and compatibility
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        # Classification and discovery
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        # Command-line interface
        entry_points=ENTRY_POINTS,
        # Build configuration
        zip_safe=False,  # Needed for data files access
        # Additional metadata
        platforms=["any"],
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
    )


if __name__ == "__main__":
    main()
