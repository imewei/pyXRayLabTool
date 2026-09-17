"""
Command Line Interface for XRayLabTool.

This package provides a comprehensive CLI for calculating X-ray optical properties
of materials, including single material calculations, batch processing, utility
functions for X-ray analysis, and shell completion installation.

Available Commands:
    calc                Calculate X-ray properties for a single material
    batch               Process multiple materials from CSV file
    compare             Compare X-ray properties between multiple materials
    convert             Convert between energy and wavelength units
    formula             Parse and analyze chemical formulas
    atomic              Look up atomic data for elements
    bragg               Calculate Bragg angles for diffraction
    list                List available data and information
    install-completion  Install shell completion for xraylabtool
    uninstall-completion Remove shell completion for xraylabtool

The CLI supports various output formats (table, CSV, JSON), field filtering,
precision control, and comprehensive shell completion for enhanced usability.

Command parsing lives in ``parser.py``, output formatting in ``formatting.py``,
shared helpers in ``utils.py``, and each subcommand's handler in its own module
under ``commands/``. This module wires them together and re-exports the public
names that used to live in the single ``cli.py`` module, for backward
compatibility.
"""

import sys
from typing import Any

from xraylabtool.interfaces.cli.commands.atomic import cmd_atomic
from xraylabtool.interfaces.cli.commands.batch import (
    _convert_result_to_dict,
    _parse_batch_data,
    _process_batch_materials,
    _save_batch_results,
    _validate_batch_input,
    cmd_batch,
)
from xraylabtool.interfaces.cli.commands.bragg import cmd_bragg
from xraylabtool.interfaces.cli.commands.calc import (
    _determine_output_format,
    _print_calc_verbose_info,
    _save_or_print_output,
    _validate_calc_inputs,
    cmd_calc,
)
from xraylabtool.interfaces.cli.commands.compare import cmd_compare
from xraylabtool.interfaces.cli.commands.completion import (
    cmd_completion,
    cmd_install_completion,
    cmd_uninstall_completion,
)
from xraylabtool.interfaces.cli.commands.convert import cmd_convert
from xraylabtool.interfaces.cli.commands.formula import (
    _get_atomic_data,
    _output_formula_results,
    _print_formula_results,
    _process_formula,
    cmd_formula,
)
from xraylabtool.interfaces.cli.commands.list_cmd import cmd_list
from xraylabtool.interfaces.cli.formatting import (
    _format_as_csv,
    _format_as_json,
    _format_filtered_table,
    _format_material_properties,
    _format_multiple_energies,
    _get_default_fields,
    format_xray_result,
)
from xraylabtool.interfaces.cli.parser import create_parser
from xraylabtool.interfaces.cli.utils import (
    create_batch_progress_tracker,
    parse_energy_string,
)
from xraylabtool.logging_utils import configure_logging, get_logger, log_environment

__all__ = [
    "cmd_atomic",
    "cmd_batch",
    "cmd_bragg",
    "cmd_calc",
    "cmd_compare",
    "cmd_completion",
    "cmd_convert",
    "cmd_formula",
    "cmd_install_completion",
    "cmd_list",
    "cmd_uninstall_completion",
    "create_parser",
    "format_xray_result",
    "main",
    "parse_energy_string",
]


def main() -> int:
    """Execute the main CLI application."""
    configure_logging()
    logger = get_logger("cli")
    log_environment(logger, component="cli")

    parser = create_parser()

    import time

    started = time.perf_counter()

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # Handle argparse sys.exit calls gracefully in tests
        if e.code == 0:  # --help or --version
            raise  # Re-raise for normal help/version behavior
        else:
            # Invalid arguments - return error code instead of exiting
            return 1

    # Show debug mode status if enabled
    if getattr(args, "debug", False):
        print(
            "🔍 Debug mode enabled - detailed error information will be shown",
            file=sys.stderr,
        )

    # Handle --install-completion flag before checking for subcommands
    if hasattr(args, "install_completion") and args.install_completion is not None:
        from xraylabtool.interfaces.completion_v2.integration import (
            install_completion_main,
        )

        # Create a mock args object that matches the install-completion
        # subcommand format
        class MockArgs:
            def __init__(
                self,
                shell_type: str | None,
                test: bool = False,
                system: bool = False,
                uninstall: bool = False,
            ) -> None:
                self.shell = shell_type if shell_type != "auto" else None
                self.system = system
                # user installation is default unless system is specified
                self.user = not system
                self.uninstall = uninstall
                self.test = test

        mock_args = MockArgs(
            args.install_completion,
            test=getattr(args, "test", False),
            system=getattr(args, "system", False),
            uninstall=getattr(args, "uninstall", False),
        )
        return install_completion_main(mock_args)  # type: ignore[arg-type]

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    command_handlers = {
        "calc": cmd_calc,
        "batch": cmd_batch,
        "compare": cmd_compare,
        "convert": cmd_convert,
        "formula": cmd_formula,
        "atomic": cmd_atomic,
        "bragg": cmd_bragg,
        "list": cmd_list,
        "completion": cmd_completion,
        "install-completion": cmd_install_completion,
        "uninstall-completion": cmd_uninstall_completion,
    }

    handler = command_handlers.get(args.command)
    if handler:
        logger.info("Starting command", extra={"command": args.command})
        try:
            rc = handler(args)
        except Exception as exc:
            logger.exception("Command failed", extra={"command": args.command})
            if getattr(args, "debug", False):
                raise
            print(f"Error running {args.command}: {exc}", file=sys.stderr)
            return 1
        duration_s = time.perf_counter() - started
        logger.info(
            "Command finished",
            extra={
                "command": args.command,
                "status": rc,
                "duration_s": round(duration_s, 4),
            },
        )
        return rc
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
