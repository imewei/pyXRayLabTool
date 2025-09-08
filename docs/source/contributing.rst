Contributing to XRayLabTool
============================

We welcome contributions from the community! This page provides information for potential contributors.

.. note::

   For detailed contribution guidelines, please see our main `CONTRIBUTING.md <https://github.com/imewei/pyXRayLabTool/blob/main/CONTRIBUTING.md>`_ file on GitHub.

Quick Start for Contributors
-----------------------------

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a virtual environment** and install in development mode
4. **Make your changes** following our coding standards
5. **Run tests** to ensure everything works
6. **Submit a pull request** with a clear description

Development Setup
-----------------

.. code-block:: bash

   # Clone your fork
   git clone https://github.com/YOUR-USERNAME/pyXRayLabTool.git
   cd pyXRayLabTool

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in development mode with all dependencies
   pip install -e .[dev]

   # Install pre-commit hooks
   pre-commit install

Areas for Contribution
----------------------

.. grid:: 2

    .. grid-item-card:: 🐛 Bug Fixes
        :link: https://github.com/imewei/pyXRayLabTool/labels/bug

        Help us fix issues and improve reliability

    .. grid-item-card:: ✨ New Features
        :link: https://github.com/imewei/pyXRayLabTool/labels/enhancement

        Add new X-ray analysis capabilities

    .. grid-item-card:: 📚 Documentation
        :link: https://github.com/imewei/pyXRayLabTool/labels/documentation

        Improve guides, examples, or API docs

    .. grid-item-card:: 🚀 Performance
        :link: https://github.com/imewei/pyXRayLabTool/labels/performance

        Optimize calculations or memory usage

    .. grid-item-card:: 🧪 Testing
        :link: https://github.com/imewei/pyXRayLabTool/labels/testing

        Add test coverage or improve test quality

    .. grid-item-card:: 🔧 Infrastructure
        :link: https://github.com/imewei/pyXRayLabTool/labels/infrastructure

        CI/CD, build tools, or development workflow

Coding Standards
----------------

We follow strict coding standards to ensure code quality:

**Code Style:**

- **Black** for code formatting (line length 88)
- **Ruff** for fast linting and import sorting
- **MyPy** for type checking in strict mode
- **Pre-commit hooks** to enforce standards automatically

**Documentation:**

- **NumPy-style docstrings** for all public functions
- **Type hints** for all function parameters and returns
- **Examples** in docstrings where helpful
- **Comprehensive API documentation** for new features

**Testing:**

- **pytest** for all testing
- **High test coverage** (aim for >90%)
- **Multiple test categories**: unit, integration, performance
- **Test naming** following ``test_<functionality>.py`` pattern

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   make test

   # Quick tests without coverage
   make test-fast

   # Run specific test file
   pytest tests/test_core.py -v

   # Run with coverage report
   pytest --cov=xraylabtool --cov-report=html

Code Quality Checks
--------------------

.. code-block:: bash

   # Run all quality checks
   make lint
   make format
   make type-check

   # Or run pre-commit on all files
   pre-commit run --all-files

Git Workflow
------------

We use a **feature branch workflow**:

1. **Create a feature branch** from main:

   .. code-block:: bash

      git checkout main
      git pull origin main
      git checkout -b feature/your-feature-name

2. **Make your changes** in logical commits
3. **Push to your fork** and create a pull request
4. **Address review feedback** if needed
5. **Squash and merge** once approved

**Commit Message Format:**

.. code-block:: text

   feat: add energy-dependent critical angle calculation

   - Implement new calculation method for varying energies
   - Add comprehensive tests for edge cases
   - Update documentation with examples

   Fixes #123

Types of Contributions
----------------------

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- **XRayLabTool version**: ``xraylabtool --version``
- **Python version** and platform
- **Minimal code example** that reproduces the issue
- **Expected vs. actual behavior**
- **Full error traceback** if applicable

Feature Requests
~~~~~~~~~~~~~~~~

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** and motivation
- **Provide example usage** of the proposed feature
- **Consider implementation complexity** and backwards compatibility

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation contributions are highly valued:

- **Fix typos** and improve clarity
- **Add examples** for complex features
- **Update tutorials** with new functionality
- **Improve API documentation** with better descriptions

Code Contributions
~~~~~~~~~~~~~~~~~~

For code contributions:

- **Start with small changes** to get familiar with the codebase
- **Follow existing patterns** and conventions
- **Add tests** for all new functionality
- **Update documentation** as needed
- **Consider performance implications** for large-scale usage

Testing Guidelines
------------------

Test Categories
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Category
     - Description
   * - **Unit**
     - Test individual functions and classes in isolation
   * - **Integration**
     - Test complete workflows and module interactions
   * - **Performance**
     - Test speed and memory usage to prevent regressions
   * - **Edge Cases**
     - Test boundary conditions and error handling

Writing Good Tests
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_calculate_single_material_properties_basic():
       """Test basic single material calculation functionality."""
       # Arrange
       formula = "SiO2"
       energy = 10.0
       density = 2.2

       # Act
       result = calculate_single_material_properties(formula, energy, density)

       # Assert
       assert result.formula == formula
       assert result.density_g_cm3 == density
       assert len(result.energy_kev) == 1
       assert result.energy_kev[0] == energy
       assert result.critical_angle_degrees[0] > 0

Performance Considerations
--------------------------

When contributing code that affects performance:

**Measure Before Optimizing:**

.. code-block:: python

   # Use pytest-benchmark for performance tests
   def test_calculation_performance(benchmark):
       """Benchmark single material calculation performance."""
       result = benchmark(
           calculate_single_material_properties,
           "SiO2", 10.0, 2.2
       )
       assert result.formula == "SiO2"

**Consider Memory Usage:**

.. code-block:: python

   # For large datasets, test memory efficiency
   import tracemalloc

   tracemalloc.start()
   # Your code here
   current, peak = tracemalloc.get_traced_memory()
   tracemalloc.stop()

   assert peak < 100_000_000  # Less than 100 MB

**Use Profiling Tools:**

.. code-block:: bash

   # Profile your changes
   python -m cProfile -s cumulative your_script.py

   # Memory profiling
   mprof run your_script.py
   mprof plot

Release Process
---------------

XRayLabTool follows **semantic versioning**:

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.0.1): Bug fixes, backwards compatible

Release steps:

1. **Update version** in ``pyproject.toml``
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite** including performance tests
4. **Build documentation** and verify it renders correctly
5. **Create release tag** and GitHub release
6. **Publish to PyPI** (maintainers only)

Communication
-------------

**GitHub Issues:**
  For bug reports, feature requests, and specific questions

**GitHub Discussions:**
  For general questions, ideas, and community discussion

**Pull Request Reviews:**
  Be respectful, constructive, and thorough in code reviews

**Community Guidelines:**

- **Be respectful** and inclusive
- **Help others** learn and contribute
- **Focus on the code**, not the person
- **Assume good intentions**

Recognition
-----------

We recognize contributions in several ways:

- **Contributors list** in README.md
- **GitHub contributor statistics**
- **Changelog acknowledgments** for significant contributions
- **Special recognition** for exceptional contributions

Getting Help
------------

If you need help with contributing:

1. **Read the full contributing guide**: `CONTRIBUTING.md <https://github.com/imewei/pyXRayLabTool/blob/main/CONTRIBUTING.md>`_
2. **Check existing issues** for similar questions
3. **Ask on GitHub Discussions** for general help
4. **Comment on specific issues** for targeted assistance

Thank you for helping make XRayLabTool better! 🚀
