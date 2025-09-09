I/O Operations Module
=====================

The I/O module handles file operations, data import/export, and format conversions.

.. currentmodule:: xraylabtool.io

File Operations
---------------

.. automodule:: xraylabtool.io.file_operations
   :members:
   :undoc-members:
   :show-inheritance:

Supported File Formats
~~~~~~~~~~~~~~~~~~~~~~

**Input Formats:**
- **CSV**: Comma-separated values with flexible column mapping
- **JSON**: Structured data format with full feature support
- **Excel**: .xlsx files with multiple sheet support (optional dependency)

**Output Formats:**
- **CSV**: Standard comma-separated format
- **JSON**: Pretty-printed JSON with metadata
- **HDF5**: High-performance binary format for large datasets (optional)

CSV File Handling
~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.load_materials_csv

   Load material specifications from CSV file.

   **Expected CSV Format:**

   .. code-block:: text

      Formula,Density,Energy
      Si,2.33,8000
      SiO2,2.20,8000
      Al,2.70,5000

   **Flexible Column Names:**
   - Formula: "formula", "Formula", "material", "Material"
   - Density: "density", "Density", "rho", "ρ" 
   - Energy: "energy", "Energy", "E", "keV" (with automatic unit conversion)

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import load_materials_csv
      
      materials = load_materials_csv("materials.csv")
      for material in materials:
          print(f"{material['formula']}: ρ = {material['density']} g/cm³")

.. autofunction:: xraylabtool.io.file_operations.save_results_csv

   Save calculation results to CSV format.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import save_results_csv
      
      # Calculate results
      results = calculate_xray_properties(materials, energies)
      
      # Save to CSV
      save_results_csv(results, "output.csv")

JSON File Handling
~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.load_materials_json

   Load materials from JSON file with full parameter support.

   **Expected JSON Format:**

   .. code-block:: json

      [
          {
              "formula": "Si",
              "density": 2.33,
              "energy": 8000,
              "metadata": {
                  "source": "experimental",
                  "temperature": 300
              }
          }
      ]

.. autofunction:: xraylabtool.io.file_operations.save_results_json

   Save results in JSON format with metadata preservation.

Data Export
-----------

.. automodule:: xraylabtool.io.data_export
   :members:
   :undoc-members:
   :show-inheritance:

Export Functions
~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.data_export.export_to_format

   Universal export function supporting multiple formats.

   **Parameters:**
   - ``results``: List of XRayResult objects
   - ``format``: Output format ('csv', 'json', 'excel', 'hdf5')
   - ``filename``: Output file path
   - ``options``: Format-specific options

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import export_to_format
      
      # Export to different formats
      export_to_format(results, 'csv', 'data.csv')
      export_to_format(results, 'json', 'data.json', 
                      options={'pretty': True, 'include_metadata': True})
      export_to_format(results, 'excel', 'data.xlsx', 
                      options={'sheet_name': 'XRay_Properties'})

Excel Integration
~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.data_export.export_to_excel

   Export results to Excel with advanced formatting.

   **Features:**
   - Multiple sheets for different material groups
   - Formatted tables with proper headers
   - Conditional formatting for critical values
   - Charts and plots (optional)

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import export_to_excel
      
      export_to_excel(results, "analysis.xlsx", options={
          'sheet_name': 'Materials_Analysis',
          'include_charts': True,
          'format_numbers': True,
          'freeze_panes': (1, 0)
      })

HDF5 Support
~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.data_export.export_to_hdf5

   Export large datasets to HDF5 format for high-performance access.

   **Use Cases:**
   - Large batch calculations (>10,000 materials)
   - Time-series or parametric studies
   - Integration with scientific computing workflows

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import export_to_hdf5
      
      # Export large dataset
      export_to_hdf5(results, "large_dataset.h5", options={
          'compression': 'gzip',
          'group_by': 'formula',
          'include_metadata': True
      })

Format Conversion
-----------------

.. autofunction:: xraylabtool.io.data_export.convert_format

   Convert between different file formats.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import convert_format
      
      # Convert CSV to JSON
      convert_format("data.csv", "data.json", 
                    source_format="csv", target_format="json")
      
      # Convert JSON to Excel
      convert_format("data.json", "analysis.xlsx",
                    source_format="json", target_format="excel")

Batch File Processing
---------------------

.. autofunction:: xraylabtool.io.file_operations.process_directory

   Process all files in a directory with pattern matching.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import process_directory
      
      # Process all CSV files in a directory
      results = process_directory("/data/materials/", 
                                pattern="*.csv",
                                output_dir="/results/")

Advanced I/O Features
---------------------

Template Generation
~~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.generate_template

   Generate template files for batch processing.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import generate_template
      
      # Generate CSV template
      generate_template("csv", "materials_template.csv")
      
      # Generate JSON template with examples
      generate_template("json", "materials_template.json", 
                       include_examples=True)

Validation and Repair
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.validate_file

   Validate file format and content before processing.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import validate_file
      
      try:
          is_valid, errors = validate_file("materials.csv")
          if not is_valid:
              print("File validation errors:")
              for error in errors:
                  print(f"  - {error}")
      except Exception as e:
          print(f"Cannot validate file: {e}")

Metadata Handling
~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.data_export.add_metadata

   Add metadata to exported files for traceability.

   **Automatic Metadata:**
   - Calculation timestamp
   - XRayLabTool version
   - Input parameters and settings
   - System information

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import add_metadata
      
      metadata = {
          'project': 'Synchrotron Mirror Analysis',
          'operator': 'Research Team',
          'notes': 'Initial screening of candidate materials'
      }
      
      save_results_json(results, "results.json", metadata=metadata)

Error Handling
--------------

The I/O module provides comprehensive error handling:

.. code-block:: python

   from xraylabtool.io.file_operations import load_materials_csv
   from xraylabtool.validation.exceptions import ValidationError
   
   try:
       materials = load_materials_csv("materials.csv")
   except FileNotFoundError:
       print("Input file not found")
   except ValidationError as e:
       print(f"Invalid data in file: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Considerations
-------------------------

**Large Files:**
- Use chunked processing for files >100MB
- Consider HDF5 format for datasets >10,000 entries
- Enable compression for storage efficiency

**Network Files:**
- Cache frequently accessed files locally
- Use progress bars for long downloads
- Implement retry logic for network errors

**Memory Management:**
- Stream processing for very large datasets
- Automatic cleanup of temporary files
- Memory usage monitoring and warnings