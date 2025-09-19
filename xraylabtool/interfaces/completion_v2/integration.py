"""Integration layer for the new completion system.

This module provides backward compatibility and integration with the existing CLI
while gradually migrating to the new virtual environment-centric completion system.
"""

import argparse
import os
from typing import Any

from .cli import completion_main
from .installer import CompletionInstaller


def legacy_install_completion_main(args: argparse.Namespace) -> int:
    """Legacy install completion main function with new system backend.

    This function maintains backward compatibility with the existing CLI
    while using the new completion system internally.
    """
    installer = CompletionInstaller()

    # Handle legacy arguments
    shell = getattr(args, "shell", None) or getattr(args, "install_completion", None)
    force = getattr(args, "force", False) or getattr(args, "uninstall", False)
    system_wide = getattr(args, "system", False)
    test_mode = getattr(args, "test", False)
    uninstall_mode = getattr(args, "uninstall", False)

    try:
        if test_mode:
            # Test completion functionality
            return test_completion_installation()

        elif uninstall_mode:
            # Uninstall completion
            success = installer.uninstall()
            return 0 if success else 1

        elif system_wide:
            # System-wide installation not supported in new system
            print(
                "⚠️  System-wide installation is not supported in the new completion"
                " system."
            )
            print(
                "💡 The new system installs completion per virtual environment for"
                " better isolation."
            )
            print(
                "   This ensures completion is available only when the relevant"
                " environment is active."
            )

            # Offer alternative
            current_env = installer.env_detector.get_current_environment()
            if current_env:
                print(f"\n🔧 Installing in current environment: {current_env.name}")
                success = installer.install(shell=shell, force=force)
                return 0 if success else 1
            else:
                print("\n❌ No active virtual environment detected.")
                print("   Please activate a virtual environment and try again.")
                return 1

        else:
            # Regular installation in virtual environment
            success = installer.install(shell=shell, force=force)
            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def legacy_uninstall_completion_main(args: Any) -> int:
    """Legacy uninstall completion main function with new system backend."""
    installer = CompletionInstaller()

    # Handle legacy arguments
    shell = getattr(args, "shell", None) if hasattr(args, "shell") else None
    system_wide = getattr(args, "system", False) if hasattr(args, "system") else False
    cleanup = getattr(args, "cleanup", False) if hasattr(args, "cleanup") else False

    try:
        if cleanup:
            # Clean up all environments
            success = installer.uninstall(all_envs=True)
            return 0 if success else 1

        elif system_wide:
            print(
                "⚠️  System-wide uninstallation is not applicable in the new completion"
                " system."
            )
            print("💡 The new system manages completion per virtual environment.")

            # Offer to clean up current environment or all environments
            choice = (
                input("\nRemove from current environment only? (y/N): ").strip().lower()
            )
            if choice in ("y", "yes"):
                success = installer.uninstall()
                return 0 if success else 1
            else:
                choice = input("Remove from all environments? (y/N): ").strip().lower()
                if choice in ("y", "yes"):
                    success = installer.uninstall(all_envs=True)
                    return 0 if success else 1
                else:
                    print("Operation cancelled.")
                    return 0

        else:
            # Regular uninstallation from current environment
            # Use the legacy method for backward compatibility with tests
            success = installer.uninstall_completion(
                shell_type=shell, cleanup_session=True
            )
            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def test_completion_installation() -> int:
    """Test if completion is working in the current environment."""
    installer = CompletionInstaller()

    print("🧪 Testing completion installation...")

    # Check if we're in a virtual environment
    current_env = installer.env_detector.get_current_environment()
    if not current_env:
        print("❌ No active virtual environment detected.")
        print("   Completion testing requires an active virtual environment.")
        return 1

    print(f"📍 Testing in environment: {current_env.name} ({current_env.env_type})")

    # Check if completion is installed
    if not current_env.has_completion:
        print("❌ Completion is not installed in this environment.")
        print("   Run 'xraylabtool install-completion' to install it.")
        return 1

    print("✅ Completion is installed in this environment.")

    # Test shell-specific completion
    shell = installer._detect_current_shell()
    print(f"🐚 Detected shell: {shell}")

    # Check if completion script exists and is readable
    completion_dir = current_env.path / "share" / "xraylabtool" / "completion"
    script_filename = installer.completion_manager.get_filename(shell)
    script_path = completion_dir / script_filename

    if script_path.exists():
        print(f"✅ Completion script found: {script_path}")

        # Test if script is executable (on Unix-like systems)
        if os.name != "nt" and not os.access(script_path, os.X_OK):
            print("⚠️  Completion script is not executable.")
            print("   This may cause issues with completion loading.")

        # Provide testing instructions
        print("\n📋 To test completion manually:")
        print("   1. Start a new terminal session")
        print("   2. Activate this environment")
        print("   3. Type 'xraylabtool ' and press TAB")
        print("   4. You should see available commands")

        print("\n💡 If completion doesn't work:")
        print("   • Try starting a new shell session")
        print("   • Check if your shell supports completion")
        print("   • Reinstall with 'xraylabtool install-completion --force'")

        return 0

    else:
        print(f"❌ Completion script not found: {script_path}")
        print("   The completion may not be properly installed.")
        return 1


def handle_new_completion_command(args: list) -> int:
    """Handle new 'completion' subcommand."""
    # Remove 'completion' from args
    if args and args[0] == "completion":
        args = args[1:]

    return completion_main(args)


# Backward compatibility exports
install_completion_main = legacy_install_completion_main
uninstall_completion_main = legacy_uninstall_completion_main
