#!/usr/bin/env python3
"""XRayLabTool Shell Completion Installer.

A lightweight installer for shell completion functionality.
Supports Bash completion with automatic detection and installation.
"""

import shutil
import subprocess
import sys
from pathlib import Path


class CompletionInstaller:
    """Handles installation of shell completion for XRayLabTool."""

    def __init__(self) -> None:
        """Initialize the completion installer."""
        self.script_dir = Path(__file__).parent
        self.completion_script = self.script_dir / "_xraylabtool_completion.bash"

    def get_bash_completion_dir(self) -> Path | None:
        """Find the appropriate bash completion directory."""
        # Common bash completion directories in order of preference
        candidates = [
            Path("/usr/share/bash-completion/completions"),
            Path("/usr/local/share/bash-completion/completions"),
            Path.home() / ".bash_completion.d",
            Path("/etc/bash_completion.d"),
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate

        return None

    def get_user_bash_completion_dir(self) -> Path:
        """Get or create user bash completion directory."""
        user_dir = Path.home() / ".bash_completion.d"
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def install_bash_completion(self, system_wide: bool = False) -> bool:
        """Install bash completion script."""
        if not self.completion_script.exists():
            print(f"Error: Completion script not found at {self.completion_script}")
            return False

        if system_wide:
            target_dir = self.get_bash_completion_dir()
            if target_dir is None:
                print(
                    "Error: No bash completion directory found for "
                    "system-wide installation"
                )
                print("Try installing for current user only with --user flag")
                return False
        else:
            target_dir = self.get_user_bash_completion_dir()

        target_file = target_dir / "xraylabtool"

        try:
            if system_wide:
                # Copy with sudo for system-wide installation
                subprocess.run(
                    ["sudo", "cp", str(self.completion_script), str(target_file)],
                    check=True,
                )
                print(f"✓ Installed bash completion to {target_file} (system-wide)")
            else:
                # Direct copy for user installation
                shutil.copy2(self.completion_script, target_file)
                print(f"✓ Installed bash completion to {target_file} (user)")

            # Add sourcing instruction for user installation
            if not system_wide:
                self._add_bash_completion_sourcing()

            return True

        except subprocess.CalledProcessError:
            print("Error: Failed to install completion script (permission denied?)")
            return False
        except Exception as e:
            print(f"Error: Failed to install completion script: {e}")
            return False

    def _add_bash_completion_sourcing(self) -> None:
        """Add sourcing of bash completion to user's shell config."""
        bashrc = Path.home() / ".bashrc"
        bash_profile = Path.home() / ".bash_profile"

        sourcing_line = "# XRayLabTool completion"
        sourcing_cmd = "source ~/.bash_completion.d/xraylabtool"

        # Choose the appropriate file
        target_file = bashrc if bashrc.exists() else bash_profile

        if target_file.exists():
            content = target_file.read_text()
            if sourcing_cmd not in content:
                with open(target_file, "a") as f:
                    f.write(f"\n{sourcing_line}\n{sourcing_cmd}\n")
                print(f"✓ Added completion sourcing to {target_file}")
                print("  Please restart your shell or run: source ~/.bashrc")
            else:
                print("✓ Completion sourcing already present in shell config")

    def uninstall_bash_completion(self, system_wide: bool = False) -> bool:
        """Uninstall bash completion script."""
        if system_wide:
            target_dir = self.get_bash_completion_dir()
            if target_dir is None:
                print("Error: No bash completion directory found")
                return False
        else:
            target_dir = self.get_user_bash_completion_dir()

        target_file = target_dir / "xraylabtool"

        if not target_file.exists():
            print("Bash completion is not installed")
            return True

        try:
            if system_wide:
                subprocess.run(["sudo", "rm", str(target_file)], check=True)
                print(f"✓ Removed bash completion from {target_file} (system-wide)")
            else:
                target_file.unlink()
                print(f"✓ Removed bash completion from {target_file} (user)")

            return True

        except subprocess.CalledProcessError:
            print("Error: Failed to remove completion script (permission denied?)")
            return False
        except Exception as e:
            print(f"Error: Failed to remove completion script: {e}")
            return False

    def test_completion(self) -> bool:
        """Test if completion is working."""
        try:
            # Check if xraylabtool command is available
            subprocess.run(["which", "xraylabtool"], capture_output=True, check=True)
            print("✓ xraylabtool command found in PATH")

            # Try to check if completion is loaded (this is approximate)
            result = subprocess.run(
                ["bash", "-c", "complete -p xraylabtool"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and "xraylabtool" in result.stdout:
                print("✓ Bash completion appears to be loaded")
                return True
            else:
                print("⚠ Bash completion may not be loaded yet")
                print("  Try restarting your shell or run: source ~/.bashrc")
                return False

        except subprocess.CalledProcessError:
            print("⚠ xraylabtool command not found in PATH")
            print("  Make sure the package is installed and available")
            return False


def main() -> None:
    """Install shell completion for XRayLabTool."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Install shell completion for XRayLabTool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install completion for current user
  python install_completion.py install

  # Install completion system-wide (requires sudo)
  python install_completion.py install --system

  # Uninstall completion
  python install_completion.py uninstall

  # Test if completion is working
  python install_completion.py test
        """,
    )

    parser.add_argument(
        "action", choices=["install", "uninstall", "test"], help="Action to perform"
    )

    parser.add_argument(
        "--system", action="store_true", help="Install system-wide (requires sudo)"
    )

    args = parser.parse_args()

    installer = CompletionInstaller()

    if args.action == "install":
        success = installer.install_bash_completion(system_wide=args.system)
        if success:
            print("\n🎉 Installation completed!")
            print("You can now use tab completion with xraylabtool commands.")
            if not args.system:
                print("Please restart your shell or run: source ~/.bashrc")
        sys.exit(0 if success else 1)

    elif args.action == "uninstall":
        success = installer.uninstall_bash_completion(system_wide=args.system)
        sys.exit(0 if success else 1)

    elif args.action == "test":
        installer.test_completion()
        sys.exit(0)


if __name__ == "__main__":
    main()
