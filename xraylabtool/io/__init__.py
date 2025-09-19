"""
XRayLabTool I/O Module.

This module contains input/output operations, file handling, and data persistence.
"""


# Lazy import heavy modules to improve startup time
def __getattr__(name):
    if name in ["format_calculation_summary", "format_xray_result"]:
        from xraylabtool.io.data_export import (
            format_calculation_summary,
            format_xray_result,
        )

        globals().update(
            {
                "format_calculation_summary": format_calculation_summary,
                "format_xray_result": format_xray_result,
            }
        )
        return globals()[name]
    elif name in [
        "export_to_csv",
        "export_to_json",
        "load_data_file",
        "save_calculation_results",
    ]:
        from xraylabtool.io.file_operations import (
            export_to_csv,
            export_to_json,
            load_data_file,
            save_calculation_results,
        )

        globals().update(
            {
                "export_to_csv": export_to_csv,
                "export_to_json": export_to_json,
                "load_data_file": load_data_file,
                "save_calculation_results": save_calculation_results,
            }
        )
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "export_to_csv",
    "export_to_json",
    "format_calculation_summary",
    # Data formatting
    "format_xray_result",
    # File operations
    "load_data_file",
    "save_calculation_results",
]
