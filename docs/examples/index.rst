Examples
========

Real-world examples and use cases for XRayLabTool.

.. note::
   Interactive examples will be added in future releases.

Interactive Examples
--------------------

These examples include downloadable Jupyter notebooks and Python scripts demonstrating practical applications of XRayLabTool.

**Quick Start Examples:**
- Interactive tutorial with comprehensive examples
- Simple calculations and common materials
- Single energy calculations
- Energy range analysis
- Formula parsing and validation

**Application Examples:**
- Beamline design and optimization
- X-ray optics and substrate selection
- Component characterization
- High-throughput property screening

Example Categories
------------------

**By Complexity:**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Level
     - Topics Covered
     - Prerequisites
   * - Beginner
     - Single materials, basic CLI usage
     - Python basics
   * - Intermediate
     - Batch processing, energy analysis
     - NumPy, Matplotlib
   * - Advanced
     - Optimization, integration workflows
     - Scientific computing

**By Application Domain:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Domain
     - Example Applications
   * - Synchrotron Science
     - Beamline design, monochromator selection, filter optimization
   * - X-ray Optics
     - Mirror substrates, multilayer coatings, focusing elements
   * - Materials Research
     - Property screening, comparative analysis, database studies
   * - Education
     - Physics demonstrations, computational exercises

Running the Examples
--------------------

**Download Notebooks:**
All examples are available as Jupyter notebooks:

.. code-block:: bash

   # Clone the repository to get examples
   git clone https://github.com/b80985/pyXRayLabTool.git
   cd pyXRayLabTool/examples

**Required Dependencies:**
Install additional packages for running examples:

.. code-block:: bash

   pip install jupyter matplotlib pandas seaborn

**Interactive Execution:**

.. code-block:: bash

   jupyter notebook examples/

**Standalone Scripts:**
Many examples also work as standalone Python scripts:

.. code-block:: bash

   python examples/basic_calculations.py

Example Data
------------

**Included Datasets:**
- ``materials_database.csv`` - Common materials with densities
- ``synchrotron_energies.csv`` - Typical beamline energy ranges
- ``mirror_substrates.csv`` - X-ray mirror material properties
- ``test_compounds.csv`` - Chemical formulas for validation

**Custom Data:**
Examples show how to:
- Import your own material lists
- Create custom energy ranges
- Export results in various formats
- Integrate with other analysis tools

Performance Demonstrations
--------------------------

**Speed Comparisons:**
- Single vs batch processing efficiency
- Energy array vs individual calculations
- Memory usage optimization techniques
- Cache performance benefits

**Scaling Analysis:**
- Performance vs dataset size
- Memory requirements for large calculations
- Optimization strategies for different use cases

Best Practices
--------------

Examples demonstrate:

1. **Efficient Workflows:**
   - When to use batch processing
   - Energy range optimization
   - Memory management for large datasets

2. **Error Handling:**
   - Input validation strategies
   - Graceful error recovery
   - Results verification methods

3. **Data Management:**
   - Organizing material databases
   - Version control for datasets
   - Documentation and metadata

4. **Integration:**
   - Using XRayLabTool with other tools
   - Automation and scripting
   - Web application development

Contributing Examples
---------------------

We welcome community-contributed examples:

**Submission Guidelines:**
1. **Format**: Jupyter notebook with clear documentation
2. **Scope**: Focused on specific applications or techniques
3. **Testing**: Include data validation and expected results
4. **Documentation**: Explain the scientific context and methods

**Example Template:**

.. code-block:: python

   """
   Example: [Title]

   Description: [Brief description of the example]
   Application: [Target application domain]
   Level: [Beginner/Intermediate/Advanced]
   Dependencies: [Required packages beyond XRayLabTool]

   Author: [Your name]
   Date: [Creation date]
   """

   import xraylabtool as xrt
   # ... example code ...

**Submission Process:**
1. Fork the repository on GitHub
2. Add your example to the ``examples/`` directory
3. Include documentation and test data
4. Submit a pull request with description

Getting Help
------------

**For Example Issues:**
- Check the notebook outputs for expected results
- Verify your XRayLabTool installation and version
- Review the prerequisites and dependencies
- Use GitHub Issues for bug reports

**For Application Questions:**
- Join the discussion on GitHub Discussions
- Reference the scientific literature cited in examples
- Contact domain experts in the community

**For Custom Applications:**
- Start with the closest existing example
- Refer to the API documentation for detailed function information
- Consider contributing your solution back to the community
