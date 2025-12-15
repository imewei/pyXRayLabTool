"""Virtual environment-centric completion installer.

This module handles installation and management of shell completion
that activates/deactivates with virtual environment changes.
"""

import os
from pathlib import Path
import shutil

from .environment import EnvironmentDetector, EnvironmentInfo, EnvironmentType
from .shells import CompletionManager, get_global_options, get_xraylabtool_commands


class CompletionInstaller:
    """Handles virtual environment-centric completion installation."""

    def __init__(self):
        self.env_detector = EnvironmentDetector()
        self.completion_manager = CompletionManager()
        self.marker_file = ".xraylabtool_completion"

    def install(
        self,
        shell: str | None = None,
        target_env: str | None = None,
        force: bool = False,
    ) -> bool:
        """Install completion in virtual environment.

        Args:
            shell: Shell type (bash, zsh, fish, powershell). Auto-detected if None.
            target_env: Target environment name. Current environment if None.
            force: Force reinstallation if already installed.

        Returns:
            True if installation succeeded, False otherwise.
        """
        # Auto-detect shell if not specified
        if shell is None:
            shell = self._detect_current_shell()

        if shell not in self.completion_manager.get_supported_shells():
            print(f"❌ Unsupported shell: {shell}")
            print(
                "Supported shells:"
                f" {', '.join(self.completion_manager.get_supported_shells())}"
            )
            return False

        # Get target environment
        if target_env:
            env_info = self._find_environment_by_name(target_env)
            if not env_info:
                print(f"❌ Environment '{target_env}' not found")
                return False
        else:
            env_info = self.env_detector.get_current_environment()
            if not env_info:
                print("❌ No active virtual environment detected")
                print("Please activate a virtual environment or specify one with --env")
                return False

        # Check if already installed
        if not force and env_info.has_completion:
            print(f"✅ Completion already installed in {env_info.name}")
            print("Use --force to reinstall")
            return True

        print(
            f"🔧 Installing {shell} completion in {env_info.env_type} environment:"
            f" {env_info.name}"
        )

        try:
            success = self._install_to_environment(env_info, shell)
            if success:
                print("✅ Completion installed successfully!")
                print(
                    f"💡 Completion will activate when you enter the '{env_info.name}'"
                    " environment"
                )
                self._show_activation_instructions(env_info, shell)
            else:
                print("❌ Installation failed")

            return success

        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False

    def uninstall(self, target_env: str | None = None, all_envs: bool = False) -> bool:
        """Uninstall completion from environment(s).

        Args:
            target_env: Target environment name. Current environment if None.
            all_envs: Remove from all environments if True.

        Returns:
            True if uninstallation succeeded, False otherwise.
        """
        if all_envs:
            return self._uninstall_from_all_environments()

        # Get target environment
        if target_env:
            env_info = self._find_environment_by_name(target_env)
            if not env_info:
                print(f"❌ Environment '{target_env}' not found")
                return False
        else:
            env_info = self.env_detector.get_current_environment()
            if not env_info:
                print("❌ No active virtual environment detected")
                return False

        if not env_info.has_completion:
            print(f"ℹ️  Completion not installed in {env_info.name}")
            return True

        print(
            f"🗑️  Removing completion from {env_info.env_type} environment:"
            f" {env_info.name}"
        )

        try:
            success = self._uninstall_from_environment(env_info)
            if success:
                print("✅ Completion removed successfully!")
            else:
                print("❌ Uninstallation failed")

            return success

        except Exception as e:
            print(f"❌ Uninstallation failed: {e}")
            return False

    def list_environments(self) -> None:
        """List all detected environments with completion status."""
        environments = self.env_detector.discover_all_environments()

        if not environments:
            print("No environments detected")
            return

        print("\n📁 Detected Python Environments:")
        print("=" * 70)

        self.env_detector.get_current_environment()

        for env in environments:
            status_icon = "🟢" if env.is_active else "⚪"
            completion_icon = "✅" if env.has_completion else "❌"

            print(f"{status_icon} {env.env_type:12} {env.name:20} {completion_icon}")
            print(f"   Path: {env.path}")
            if env.python_version:
                print(f"   Python: {env.python_version}")
            print()

        print(
            "Legend: 🟢 = Active, ⚪ = Inactive, ✅ = Has completion, ❌ = No"
            " completion"
        )

    def status(self) -> None:
        """Show completion status for current environment."""
        current_env = self.env_detector.get_current_environment()

        if not current_env:
            print("❌ No active virtual environment detected")
            return

        print(f"📊 Completion Status for {current_env.name}")
        print("=" * 50)
        print(f"Environment Type: {current_env.env_type}")
        print(f"Environment Path: {current_env.path}")
        print(f"Python Version: {current_env.python_version or 'Unknown'}")
        print(
            "Completion Installed:"
            f" {'✅ Yes' if current_env.has_completion else '❌ No'}"
        )

        if current_env.has_completion:
            installed_shells = self._get_installed_shells(current_env)
            if installed_shells:
                print(f"Installed Shells: {', '.join(installed_shells)}")

    def _detect_current_shell(self) -> str:
        """Auto-detect the current shell."""
        shell_env = os.environ.get("SHELL", "")

        # Check SHELL environment variable first (Unix-like systems)
        if "fish" in shell_env:
            return "fish"
        elif "zsh" in shell_env:
            return "zsh"
        elif "bash" in shell_env:
            return "bash"

        # Check for PowerShell indicators
        if self._is_powershell_environment():
            return "powershell"

        # Check for Windows-style shell paths even on non-Windows systems
        # (useful for WSL, Git Bash, etc.)
        if self._has_windows_shell_indicators(shell_env):
            return "powershell"

        # Check for Windows-style environment variables
        comspec = os.environ.get("ComSpec", "").lower()
        if "powershell" in comspec or "pwsh" in comspec:
            return "powershell"

        # Check Windows-specific shell detection
        if os.name == "nt":
            return self._detect_windows_shell()

        return "bash"  # Default fallback

    def _has_windows_shell_indicators(self, shell_env: str) -> bool:
        """Check if shell path indicates Windows PowerShell."""
        windows_indicators = [
            "powershell.exe",
            "pwsh.exe",
            "WindowsPowerShell",
            "PowerShell",
        ]

        shell_lower = shell_env.lower()
        return any(indicator.lower() in shell_lower for indicator in windows_indicators)

    def _is_powershell_environment(self) -> bool:
        """Detect if running in PowerShell environment."""
        # Check common PowerShell environment variables
        powershell_indicators = [
            "PSModulePath",
            "PSVersionTable",
            "POWERSHELL_DISTRIBUTION_CHANNEL",
        ]

        for indicator in powershell_indicators:
            if indicator in os.environ:
                return True

        # Check if parent process is PowerShell
        try:
            import psutil

            current_process = psutil.Process()
            parent = current_process.parent()
            if parent and "powershell" in parent.name().lower():
                return True
        except (ImportError, Exception):
            pass

        return False

    def _detect_windows_shell(self) -> str:
        """Detect shell on Windows systems."""
        # Check ComSpec for cmd.exe vs PowerShell
        comspec = os.environ.get("ComSpec", "").lower()

        if "powershell" in comspec or "pwsh" in comspec:
            return "powershell"

        # Check for Windows Terminal or modern shells
        wt_session = os.environ.get("WT_SESSION")
        if wt_session:
            # Windows Terminal - could be any shell, default to PowerShell
            return "powershell"

        # Check SHELL even on Windows (WSL, Git Bash, etc.)
        shell = os.environ.get("SHELL", "").lower()
        if "bash" in shell:
            return "bash"
        elif "zsh" in shell:
            return "zsh"
        elif "fish" in shell:
            return "fish"

        # Default to PowerShell on Windows
        return "powershell"

    def _find_environment_by_name(self, name: str) -> EnvironmentInfo | None:
        """Find environment by name."""
        environments = self.env_detector.discover_all_environments()
        for env in environments:
            if env.name == name:
                return env
        return None

    def _install_to_environment(self, env_info: EnvironmentInfo, shell: str) -> bool:
        """Install completion to a specific environment."""
        # Create completion directory in environment
        completion_dir = env_info.path / "share" / "xraylabtool" / "completion"
        completion_dir.mkdir(parents=True, exist_ok=True)

        # Generate completion script
        commands = get_xraylabtool_commands()
        global_options = get_global_options()
        completion_script = self.completion_manager.generate_completion(
            shell, commands, global_options
        )

        # Write completion script
        script_filename = self.completion_manager.get_filename(shell)
        script_path = completion_dir / script_filename
        script_path.write_text(completion_script)

        # Make script executable on Unix-like systems
        if os.name != "nt":
            script_path.chmod(0o755)

        # Create marker file
        marker_path = env_info.path / self.marker_file
        marker_data = {
            "shell": shell,
            "script_path": str(script_path),
            "installed_by": "xraylabtool-completion-v2",
        }
        import json

        marker_path.write_text(json.dumps(marker_data, indent=2))

        # Install activation hooks
        return self._install_activation_hooks(env_info, shell, script_path)

    def _install_activation_hooks(
        self, env_info: EnvironmentInfo, shell: str, script_path: Path
    ) -> bool:
        """Install hooks in environment activation scripts."""
        if env_info.env_type in (EnvironmentType.CONDA, EnvironmentType.MAMBA):
            return self._install_conda_hooks(env_info, shell, script_path)
        elif env_info.env_type in (EnvironmentType.VENV, EnvironmentType.VIRTUALENV):
            return self._install_venv_hooks(env_info, shell, script_path)
        elif env_info.env_type == EnvironmentType.POETRY:
            return self._install_poetry_hooks(env_info, shell, script_path)
        elif env_info.env_type == EnvironmentType.PIPENV:
            return self._install_pipenv_hooks(env_info, shell, script_path)

        return False

    def _install_conda_hooks(
        self, env_info: EnvironmentInfo, shell: str, script_path: Path
    ) -> bool:
        """Install hooks for conda/mamba environments."""
        # For conda, we create activation/deactivation scripts
        activate_dir = env_info.path / "etc" / "conda" / "activate.d"
        deactivate_dir = env_info.path / "etc" / "conda" / "deactivate.d"

        activate_dir.mkdir(parents=True, exist_ok=True)
        deactivate_dir.mkdir(parents=True, exist_ok=True)

        if shell == "bash":
            # Bash activation script
            activate_script = activate_dir / "xraylabtool-completion.sh"
            activate_script.write_text(f"""#!/bin/bash
# XRayLabTool completion activation
if [ -f "{script_path}" ]; then
    source "{script_path}"
fi
""")

            # Bash deactivation script
            deactivate_script = deactivate_dir / "xraylabtool-completion.sh"
            deactivate_script.write_text("""#!/bin/bash
# XRayLabTool completion deactivation
complete -r xraylabtool 2>/dev/null || true
""")

        elif shell == "fish":
            # Fish activation script
            activate_script = activate_dir / "xraylabtool-completion.fish"
            activate_script.write_text(f"""# XRayLabTool completion activation
if test -f "{script_path}"
    source "{script_path}"
end
""")

        elif shell == "zsh":
            # Zsh activation script
            activate_script = activate_dir / "xraylabtool-completion.zsh"
            activate_script.write_text(f"""#!/bin/zsh
# XRayLabTool completion activation
if [ -f "{script_path}" ]; then
    source "{script_path}"
fi
""")

        # Make scripts executable
        if os.name != "nt":
            for script in activate_dir.glob("xraylabtool-completion.*"):
                script.chmod(0o755)

        return True

    def _install_venv_hooks(
        self, env_info: EnvironmentInfo, shell: str, script_path: Path
    ) -> bool:
        """Install hooks for venv/virtualenv environments."""
        # Modify activation scripts
        if shell == "bash":
            activate_script = env_info.path / "bin" / "activate"
            if activate_script.exists():
                self._modify_activation_script(activate_script, script_path, shell)

        elif shell == "fish":
            activate_script = env_info.path / "bin" / "activate.fish"
            if activate_script.exists():
                self._modify_activation_script(activate_script, script_path, shell)

        elif shell == "zsh":
            # For zsh, we typically use the bash activation script
            activate_script = env_info.path / "bin" / "activate"
            if activate_script.exists():
                self._modify_activation_script(activate_script, script_path, shell)

        return True

    def _install_poetry_hooks(
        self, env_info: EnvironmentInfo, shell: str, script_path: Path
    ) -> bool:
        """Install hooks for Poetry environments."""
        # Poetry environments are typically handled like regular venv
        return self._install_venv_hooks(env_info, shell, script_path)

    def _install_pipenv_hooks(
        self, env_info: EnvironmentInfo, shell: str, script_path: Path
    ) -> bool:
        """Install hooks for Pipenv environments."""
        # Pipenv environments are typically handled like regular venv
        return self._install_venv_hooks(env_info, shell, script_path)

    def _modify_activation_script(
        self, activate_script: Path, completion_script: Path, shell: str
    ) -> None:
        """Modify environment activation script to source completion."""
        if not activate_script.exists():
            return

        content = activate_script.read_text()

        # Check if completion is already added
        if "xraylabtool" in content.lower() and "completion" in content.lower():
            return

        # Add completion activation
        if shell == "fish":
            completion_code = f"""
# XRayLabTool completion (added by xraylabtool completion installer)
if test -f "{completion_script}"
    source "{completion_script}"
end
"""
        else:  # bash/zsh
            completion_code = f"""
# XRayLabTool completion (added by xraylabtool completion installer)
if [ -f "{completion_script}" ]; then
    source "{completion_script}"
fi
"""

        # Add before the end of the script
        content += completion_code
        activate_script.write_text(content)

    def _uninstall_from_environment(self, env_info: EnvironmentInfo) -> bool:
        """Uninstall completion from a specific environment."""
        # Remove completion directory
        completion_dir = env_info.path / "share" / "xraylabtool" / "completion"
        if completion_dir.exists():
            shutil.rmtree(completion_dir)

        # Clean up empty parent directories
        xraylabtool_dir = env_info.path / "share" / "xraylabtool"
        if xraylabtool_dir.exists():
            try:
                # Only remove if empty
                if not any(xraylabtool_dir.iterdir()):
                    xraylabtool_dir.rmdir()
            except OSError:
                # Directory not empty or permission issue, ignore
                pass

        # Don't remove share directory as it may contain other important files
        # (e.g., Jupyter configurations, man pages, etc.)

        # Remove marker file
        marker_path = env_info.path / self.marker_file
        if marker_path.exists():
            marker_path.unlink()

        # Remove activation hooks
        return self._remove_activation_hooks(env_info)

    def _remove_activation_hooks(self, env_info: EnvironmentInfo) -> bool:
        """Remove activation hooks from environment."""
        if env_info.env_type in (EnvironmentType.CONDA, EnvironmentType.MAMBA):
            return self._remove_conda_hooks(env_info)
        else:
            return self._remove_venv_hooks(env_info)

    def _remove_conda_hooks(self, env_info: EnvironmentInfo) -> bool:
        """Remove conda activation hooks."""
        activate_dir = env_info.path / "etc" / "conda" / "activate.d"
        deactivate_dir = env_info.path / "etc" / "conda" / "deactivate.d"

        # Remove activation scripts
        for script_path in activate_dir.glob("xraylabtool-completion.*"):
            script_path.unlink()

        for script_path in deactivate_dir.glob("xraylabtool-completion.*"):
            script_path.unlink()

        return True

    def _remove_venv_hooks(self, env_info: EnvironmentInfo) -> bool:
        """Remove venv activation hooks."""
        activation_scripts = [
            env_info.path / "bin" / "activate",
            env_info.path / "bin" / "activate.fish",
        ]

        for script_path in activation_scripts:
            if not script_path.exists():
                continue

            content = script_path.read_text()

            # Remove completion-related lines
            lines = content.split("\n")
            filtered_lines = []
            skip_until_end = False

            for line in lines:
                if "XRayLabTool completion" in line and "added by xraylabtool" in line:
                    skip_until_end = True
                    continue

                if skip_until_end:
                    if line.strip() == "" or line.startswith("#"):
                        continue
                    elif "fi" in line or "end" in line:
                        skip_until_end = False
                        continue

                if not skip_until_end:
                    filtered_lines.append(line)

            script_path.write_text("\n".join(filtered_lines))

        return True

    def _uninstall_from_all_environments(self) -> bool:
        """Uninstall completion from all environments."""
        environments = self.env_detector.discover_all_environments()
        success_count = 0
        total_count = 0

        for env in environments:
            if env.has_completion:
                total_count += 1
                print(f"🗑️  Removing completion from {env.name}...")
                if self._uninstall_from_environment(env):
                    success_count += 1
                    print(f"✅ Removed from {env.name}")
                else:
                    print(f"❌ Failed to remove from {env.name}")

        if total_count == 0:
            print("ℹ️  No environments with completion found")
            return True

        print(f"\n📊 Uninstalled from {success_count}/{total_count} environments")
        return success_count == total_count

    def _get_installed_shells(self, env_info: EnvironmentInfo) -> list[str]:
        """Get list of shells with completion installed."""
        completion_dir = env_info.path / "share" / "xraylabtool" / "completion"
        if not completion_dir.exists():
            return []

        shells = []
        for shell in self.completion_manager.get_supported_shells():
            filename = self.completion_manager.get_filename(shell)
            if (completion_dir / filename).exists():
                shells.append(shell)

        return shells

    def _show_activation_instructions(
        self, env_info: EnvironmentInfo, shell: str
    ) -> None:
        """Show instructions for activating completion."""
        if env_info.env_type in (EnvironmentType.CONDA, EnvironmentType.MAMBA):
            print(
                "💡 Completion will activate automatically when you activate the"
                " environment:"
            )
            if env_info.env_type == EnvironmentType.MAMBA:
                print(f"   mamba activate {env_info.name}")
                print(f"   # or: conda activate {env_info.name}")
            else:
                print(f"   conda activate {env_info.name}")
        else:
            print(
                "💡 Completion will activate automatically when you activate the"
                " environment."
            )
            if env_info.env_type == EnvironmentType.POETRY:
                print("   poetry shell")
            elif env_info.env_type == EnvironmentType.PIPENV:
                print("   pipenv shell")
            else:
                print(f"   source {env_info.path}/bin/activate")

        print(f"   You may need to start a new {shell} session to see the changes.")

    # Backward compatibility methods for tests
    def install_bash_completion(self, **kwargs) -> bool:
        """Install bash completion (backward compatibility)."""
        return self.install(shell="bash", **kwargs)

    def uninstall_bash_completion(self, **kwargs) -> bool:
        """Uninstall bash completion (backward compatibility)."""
        return self.uninstall(**kwargs)

    def uninstall_completion(
        self, shell_type=None, cleanup_session=True, **kwargs
    ) -> bool:
        """Uninstall completion (backward compatibility)."""
        # Handle the specific call pattern from tests
        if cleanup_session:
            return self.uninstall(all_envs=True, **kwargs)
        return self.uninstall(**kwargs)

    def test_completion(self) -> bool:
        """Test completion functionality (backward compatibility)."""
        current_env = self.env_detector.get_current_environment()
        return current_env is not None and current_env.has_completion

    def get_bash_completion_dir(self) -> Path:
        """Get bash completion directory (backward compatibility)."""
        return Path("/etc/bash_completion.d")

    def get_user_bash_completion_dir(self) -> Path:
        """Get user bash completion directory (backward compatibility)."""
        return Path.home() / ".bash_completion.d"
