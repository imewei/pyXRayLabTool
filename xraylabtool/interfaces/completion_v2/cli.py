"""Command-line interface for the new completion system.

This module provides CLI commands for managing the virtual environment-centric
shell completion system.
"""

import argparse
import sys
from typing import Any

from .installer import CompletionInstaller
from .shells import CompletionManager


def create_completion_parser() -> argparse.ArgumentParser:
    """Create argument parser for completion commands."""
    parser = argparse.ArgumentParser(
        prog="xraylabtool completion",
        description="Manage virtual environment-centric shell completion",
    )

    subparsers = parser.add_subparsers(dest="action", help="Available actions")

    # Install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install completion in virtual environment",
    )
    install_parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish", "powershell"],
        help="Shell type (auto-detected if not specified)",
    )
    install_parser.add_argument(
        "--env",
        "-e",
        help="Target environment name (current environment if not specified)",
    )
    install_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force reinstallation if already installed",
    )

    # Uninstall command
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Remove completion from environment(s)",
    )
    uninstall_parser.add_argument(
        "--env",
        "-e",
        help="Target environment name (current environment if not specified)",
    )
    uninstall_parser.add_argument(
        "--all",
        action="store_true",
        help="Remove from all environments",
    )

    # List command
    subparsers.add_parser(
        "list",
        help="List environments with completion status",
    )

    # Status command
    subparsers.add_parser(
        "status",
        help="Show completion status for current environment",
    )

    # Info command
    subparsers.add_parser(
        "info",
        help="Show information about the completion system",
    )

    return parser


def handle_completion_command(args: argparse.Namespace) -> int:
    """Handle completion subcommands."""
    installer = CompletionInstaller()

    try:
        if args.action == "install":
            success = installer.install(
                shell=args.shell,
                target_env=args.env,
                force=args.force,
            )
            return 0 if success else 1

        elif args.action == "uninstall":
            success = installer.uninstall(
                target_env=args.env,
                all_envs=args.all,
            )
            return 0 if success else 1

        elif args.action == "list":
            installer.list_environments()
            return 0

        elif args.action == "status":
            installer.status()
            return 0

        elif args.action == "info":
            show_completion_info()
            return 0

        else:
            print("❌ No action specified. Use --help for usage information.")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def show_completion_info() -> None:
    """Show information about the completion system."""
    manager = CompletionManager()

    print("🔧 XRayLabTool Shell Completion System v2")
    print("=" * 50)
    print("Features:")
    print("  • Virtual environment-centric installation")
    print("  • Automatic activation/deactivation with environments")
    print("  • Support for multiple environment managers")
    print("  • No system-wide modifications required")
    print("  • Native completion for multiple shells")
    print()

    print("Supported Shells:")
    for shell in manager.get_supported_shells():
        print(f"  • {shell}")
    print()

    print("Supported Environment Types:")
    print("  • venv / virtualenv")
    print("  • conda / mamba")
    print("  • Poetry")
    print("  • Pipenv")
    print()

    print("Usage Examples:")
    print("  xraylabtool completion install           # Install in current environment")
    print("  xraylabtool completion install --shell zsh")
    print("  xraylabtool completion list              # List all environments")
    print("  xraylabtool completion uninstall --all   # Remove from all environments")


def completion_main(args: Any = None) -> int:
    """Main entry point for completion commands."""
    if args is None:
        args = sys.argv[1:]

    # If called with install-completion (legacy compatibility)
    if isinstance(args, argparse.Namespace) and hasattr(args, "action"):
        # This is the legacy interface, redirect to new system
        installer = CompletionInstaller()

        # Convert legacy args to new format
        shell = getattr(args, "shell", None)
        force = getattr(args, "force", False)

        success = installer.install(shell=shell, force=force)
        return 0 if success else 1

    # Parse new completion commands
    parser = create_completion_parser()
    parsed_args = parser.parse_args(args)

    return handle_completion_command(parsed_args)


# Legacy compatibility functions
def install_completion_main(args: argparse.Namespace) -> int:
    """Legacy install completion main function."""
    return completion_main(args)


def uninstall_completion_main(args: Any) -> int:
    """Legacy uninstall completion main function."""
    # Convert to new format
    installer = CompletionInstaller()
    success = installer.uninstall()
    return 0 if success else 1
