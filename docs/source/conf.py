"""Configuration file for the Sphinx documentation builder."""

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "XRayLabTool"
copyright = "2025, XRayLabTool Developers"
author = "XRayLabTool Developers"
release = "0.1.10"
version = "0.1.10"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core Sphinx extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",  # For math equations
    "sphinx.ext.githubpages",  # GitHub Pages deployment support
    "sphinx.ext.doctest",  # Test code examples
    "sphinx.ext.coverage",  # Documentation coverage
    "sphinx.ext.todo",  # TODO directive support
    "sphinx.ext.extlinks",  # External link shortcuts
    # Essential enhanced functionality extensions
    "sphinx_copybutton",  # Add copy button to code blocks
    "sphinx_design",  # Modern design components
    "sphinx_autodoc_typehints",  # Better type hint rendering
    "sphinx_togglebutton",  # Collapsible content
    "sphinx_tabs.tabs",  # Multi-tab content support
    # Content format support
    "myst_parser",  # Markdown support
    # "nbsphinx",  # Jupyter notebook support - temporarily disabled
]

# Napoleon settings for numpydoc style
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
autosummary_generate = False  # Disable automatic generation to avoid conflicts
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Suppress duplicate object warnings
suppress_warnings = ["autosummary.import_cycle"]

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# MyST Parser settings (for Markdown support)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Autodoc typehints settings
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
always_document_param_types = True

# NBSphinx settings (disabled)
# nbsphinx_execute = "never"  # Don't execute notebooks during build
# nbsphinx_allow_errors = True  # Allow notebooks with errors


# External link shortcuts
extlinks = {
    "pypi": ("https://pypi.org/project/%s/", "PyPI: %s"),
    "github": ("https://github.com/imewei/pyXRayLabTool/%s", "GitHub: %s"),
    "issue": ("https://github.com/imewei/pyXRayLabTool/issues/%s", "#%s"),
}

# TODO configuration
todo_include_todos = True


templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.ico"
html_logo = "_static/logo.svg"

# HTML meta tags for SEO
html_meta = {
    "description": (
        "High-performance X-ray optical properties calculator for "
        "materials science and synchrotron research"
    ),
    "keywords": (
        "x-ray, crystallography, materials science, synchrotron, "
        "CXRO, NIST, scattering factors, optical constants"
    ),
    "author": "Wei Chen",
    "viewport": "width=device-width, initial-scale=1.0",
    "robots": "index, follow",
    "theme-color": "#2980B9",
}

# Additional HTML configuration
html_title = f"{project} {version} Documentation"
html_short_title = project

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/imewei/pyXRayLabTool",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "top_of_page_buttons": ["edit"],
    "sidebar_hide_name": False,
    "light_css_variables": {
        "color-brand-primary": "#2980B9",
        "color-brand-content": "#2980B9",
        "color-api-name": "#E74C3C",
        "color-api-pre-name": "#E74C3C",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3498DB",
        "color-brand-content": "#3498DB",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/imewei/pyXRayLabTool",
            "html": (
                """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">  # noqa: E501
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>  # noqa: E501
                </svg>
            """
            ),
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xraylabtool/",
            "html": (
                """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 24 24">  # noqa: E501
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"></path>  # noqa: E501
                </svg>
            """
            ),
            "class": "",
        },
    ],
}

html_context = {
    "display_github": True,
    "github_user": "imewei",
    "github_repo": "pyXRayLabTool",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "mendeleev": ("https://mendeleev.readthedocs.io/en/stable/", None),
    # "tqdm": ("https://tqdm.github.io/docs/", None),  # URL not available
}

# -- Extension-specific configuration ----------------------------------------

# Doctest configuration
doctest_default_flags = (
    0
    | __import__("doctest").ELLIPSIS
    | __import__("doctest").IGNORE_EXCEPTION_DETAIL
    | __import__("doctest").DONT_ACCEPT_TRUE_FOR_1
)
