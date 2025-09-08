Tutorials
=========

Welcome to the XRayLabTool tutorials! These guides will walk you through practical examples and use cases for X-ray optical property calculations.

.. grid:: 2

    .. grid-item-card:: 🚀 Basic Usage
        :link: basic_usage
        :link-type: doc

        Learn the fundamentals of calculating X-ray properties for materials.

    .. grid-item-card:: 🔬 Advanced Examples
        :link: advanced_examples
        :link-type: doc

        Complex calculations for real-world scenarios and research applications.

    .. grid-item-card:: 📊 Data Analysis
        :link: ../howto/data_processing
        :link-type: doc

        Process large datasets and create visualizations of your results.

    .. grid-item-card:: ⚡ Performance Optimization
        :link: ../performance_guide
        :link-type: doc

        Optimize calculations for high-performance computing environments.

Tutorial Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   basic_usage
   advanced_examples


Getting Started
---------------

If you're new to XRayLabTool, we recommend starting with:

1. :doc:`../installation` - Install XRayLabTool on your system
2. :doc:`../quickstart` - Your first calculations in 5 minutes
3. :doc:`basic_usage` - Comprehensive introduction to all features
4. :doc:`advanced_examples` - Real-world applications

Prerequisites
~~~~~~~~~~~~~

These tutorials assume you have:

- Basic Python knowledge
- Familiarity with NumPy arrays
- Understanding of X-ray physics concepts (helpful but not required)

Learning Path by Experience Level
---------------------------------

Beginner
~~~~~~~~

**Learning Path:** Installation → Quickstart → Basic Usage → CLI Guide

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Tutorial
     - Description
   * - :doc:`../installation`
     - Get XRayLabTool installed and working
   * - :doc:`../quickstart`
     - Your first calculation in minutes
   * - :doc:`basic_usage`
     - Complete introduction to the Python API
   * - :doc:`../cli_guide`
     - Master the command-line interface

Intermediate
~~~~~~~~~~~~

**Learning Path:** Basic Usage → Advanced Examples → Data Analysis → Performance

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Tutorial
     - Description
   * - :doc:`advanced_examples`
     - Complex calculations and multi-material analysis
   * - :doc:`../howto/data_processing`
     - Process large datasets and create visualizations
   * - :doc:`../performance_guide`
     - Optimize for speed and memory efficiency

Advanced
~~~~~~~~

**Learning Path:** Performance → Custom Extensions → Integration → Contributing

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Tutorial
     - Description
   * - :doc:`../performance_guide`
     - High-performance computing optimization
   * - :doc:`../howto/data_processing`
     - Define custom materials and compositions
   * - :doc:`../installation`
     - Installation and setup guides
   * - :doc:`../contributing`
     - Contribute to the XRayLabTool project

Common Use Cases
----------------

Research Applications
~~~~~~~~~~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card:: 🔬 Synchrotron Beamlines
        :link: advanced_examples.html#synchrotron-optics

        Design X-ray mirrors, monochromators, and focusing elements

    .. grid-item-card:: 🧪 Materials Characterization
        :link: advanced_examples.html#materials-analysis

        Analyze thin films, multilayers, and bulk materials

    .. grid-item-card:: 📡 X-ray Imaging
        :link: advanced_examples.html#imaging-applications

        Optimize contrast and dose for medical and industrial imaging

    .. grid-item-card:: 💎 Crystallography
        :link: advanced_examples.html#crystallography

        Calculate structure factors and diffraction properties

Engineering Applications
~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card:: 🏭 Quality Control
        :link: data_analysis.html#batch-analysis

        Automated analysis of material batches

    .. grid-item-card:: 🎯 Design Optimization
        :link: advanced_examples.html#optimization

        Optimize material selection for specific applications

    .. grid-item-card:: 📊 Data Processing
        :link: data_analysis.html#data-pipeline

        Build analysis pipelines for large datasets

    .. grid-item-card:: 🔧 Tool Integration
        :link: ../howto/integration

        Integrate with existing analysis workflows

Tutorial Formats
----------------

We provide tutorials in multiple formats to suit your learning style:

.. tabs::

   .. tab:: Written Guides

      Step-by-step instructions with code examples and explanations.
      Perfect for following along at your own pace.

   .. tab:: Jupyter Notebooks

      Interactive notebooks you can download and run locally.
      Includes sample data and visualizations.

   .. tab:: Video Walkthroughs

      Recorded demonstrations of key concepts and workflows.
      Available in future releases.

Need Help?
----------

If you get stuck while following a tutorial:

1. **Check the FAQ**: :doc:`../faq` for common issues
2. **Search Issues**: `GitHub Issues <https://github.com/imewei/pyXRayLabTool/issues>`_
3. **Ask Questions**: `GitHub Discussions <https://github.com/imewei/pyXRayLabTool/discussions>`_
4. **Get Support**: Include your system info and the specific tutorial step

Contributing Tutorials
----------------------

We welcome community contributions to our tutorials! If you have:

- Interesting use cases to share
- Improvements to existing tutorials
- New tutorial ideas

Please see our :doc:`../contributing` guide for how to get involved.

.. tip::

   All tutorial code is tested automatically to ensure it works with the latest version of XRayLabTool. Look for the ✅ badges on tutorial pages.
