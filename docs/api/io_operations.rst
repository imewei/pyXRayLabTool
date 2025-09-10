I/O Operations Module
=====================

The I/O module handles file operations, data import/export, and format conversions.

.. automodule:: xraylabtool.io
   :members:
   :undoc-members:
   :show-inheritance:

File Operations
---------------

.. automodule:: xraylabtool.io.file_operations
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Data Export and Formatting
---------------------------

.. automodule:: xraylabtool.io.data_export
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Available Functions
-------------------

File Loading and Saving
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.load_data_file

   Load data file with error handling.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import load_data_file

      data = load_data_file("atomic_data.nff")
      print(data.head())

.. autofunction:: xraylabtool.io.file_operations.save_calculation_results

   Save calculation results to file.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import save_calculation_results

      # Save as CSV
      save_calculation_results(results, "output.csv", format_type="csv")

      # Save as JSON
      save_calculation_results(results, "output.json", format_type="json")

Export Functions
~~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.file_operations.export_to_csv

   Export data to CSV format.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import export_to_csv

      export_to_csv(calculation_results, "results.csv")

.. autofunction:: xraylabtool.io.file_operations.export_to_json

   Export data to JSON format.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.file_operations import export_to_json

      export_to_json(calculation_results, "results.json")

Data Formatting
~~~~~~~~~~~~~~~

.. autofunction:: xraylabtool.io.data_export.format_xray_result

   Format XRayResult for display or export.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import format_xray_result

      # Format as table
      table_output = format_xray_result(result, format_type="table")
      print(table_output)

      # Format as JSON
      json_output = format_xray_result(result, format_type="json")

      # Format as CSV
      csv_output = format_xray_result(result, format_type="csv")

.. autofunction:: xraylabtool.io.data_export.format_calculation_summary

   Format a summary of multiple calculation results.

   **Example:**

   .. code-block:: python

      from xraylabtool.io.data_export import format_calculation_summary

      summary = format_calculation_summary(results_list, format_type="table")
      print(summary)

Supported File Formats
-----------------------

**Input Formats:**
- **Space-separated (.nff)**: Atomic scattering factor data files
- **CSV**: General comma-separated data files

**Output Formats:**
- **CSV**: Standard comma-separated format
- **JSON**: Pretty-printed JSON with proper numeric precision

Usage Examples
--------------

**Basic File Operations:**

.. code-block:: python

   from xraylabtool.io.file_operations import load_data_file, save_calculation_results
   from xraylabtool.io.data_export import format_xray_result

   # Load atomic data
   atomic_data = load_data_file("element.nff")

   # Save calculation results
   save_calculation_results(results, "output.csv")

   # Format results for display
   formatted_result = format_xray_result(result, format_type="table", precision=4)
   print(formatted_result)

Error Handling
--------------

The I/O module provides comprehensive error handling:

.. code-block:: python

   from xraylabtool.io.file_operations import load_data_file
   from xraylabtool.validation.exceptions import DataFileError

   try:
       data = load_data_file("data.nff")
   except FileNotFoundError:
       print("Input file not found")
   except DataFileError as e:
       print(f"Error loading data file: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Considerations
--------------------------

**File Loading:**
- Automatic detection of .nff format for space-separated data
- Efficient pandas-based CSV parsing
- Proper error handling for malformed files

**Data Export:**
- Configurable numeric precision for output formatting
- Support for both single results and result collections
- Memory-efficient processing for large datasets
