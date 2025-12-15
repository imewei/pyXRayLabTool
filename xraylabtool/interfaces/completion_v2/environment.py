"""Environment detection and management for shell completion.

This module provides robust detection of Python virtual environments
and manages completion activation/deactivation based on environment state.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess


class EnvironmentType:
    """Constants for environment types."""

    SYSTEM = "system"
    VENV = "venv"
    VIRTUALENV = "virtualenv"
    CONDA = "conda"
    MAMBA = "mamba"
    PIPENV = "pipenv"
    POETRY = "poetry"


class EnvironmentInfo:
    """Information about a detected environment."""

    def __init__(
        self,
        env_type: str,
        path: Path,
        name: str,
        is_active: bool = False,
        python_version: str | None = None,
        has_completion: bool = False,
    ):
        self.env_type = env_type
        self.path = path
        self.name = name
        self.is_active = is_active
        self.python_version = python_version
        self.has_completion = has_completion

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "env_type": self.env_type,
            "path": str(self.path),
            "name": self.name,
            "is_active": self.is_active,
            "python_version": self.python_version,
            "has_completion": self.has_completion,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentInfo":
        """Create from dictionary."""
        return cls(
            env_type=data["env_type"],
            path=Path(data["path"]),
            name=data["name"],
            is_active=data.get("is_active", False),
            python_version=data.get("python_version"),
            has_completion=data.get("has_completion", False),
        )


class EnvironmentDetector:
    """Detects and manages Python virtual environments."""

    def __init__(self):
        self._cache_file = Path.home() / ".xraylabtool" / "env_cache.json"
        self._cache_file.parent.mkdir(exist_ok=True)
        self._cache_timeout = 3600  # 1 hour in seconds

    def get_current_environment(self) -> EnvironmentInfo | None:
        """Get information about the currently active environment."""
        env_type = self._detect_current_environment_type()
        env_path = self._get_current_environment_path(env_type)

        if not env_path:
            return None

        name = self._get_environment_name(env_path, env_type)
        python_version = self._get_python_version(env_path)
        has_completion = self._check_completion_installed(env_path)

        return EnvironmentInfo(
            env_type=env_type,
            path=env_path,
            name=name,
            is_active=True,
            python_version=python_version,
            has_completion=has_completion,
        )

    def discover_all_environments(
        self, use_cache: bool = True
    ) -> list[EnvironmentInfo]:
        """Discover all available Python environments."""
        if use_cache and self._is_cache_valid():
            cached_envs = self._load_cache()
            if cached_envs:
                return [EnvironmentInfo.from_dict(env) for env in cached_envs]

        environments = []
        current_env = self.get_current_environment()

        # Add system environment
        system_env = EnvironmentInfo(
            env_type=EnvironmentType.SYSTEM,
            path=Path("/usr/bin"),  # Placeholder
            name="system",
            is_active=(current_env is None),
            python_version=self._get_system_python_version(),
            has_completion=self._check_system_completion(),
        )
        environments.append(system_env)

        # Discover virtual environments
        environments.extend(self._discover_venv_environments())
        environments.extend(self._discover_conda_environments())
        environments.extend(self._discover_poetry_environments())

        # Mark current environment as active
        if current_env:
            for env in environments:
                if env.path == current_env.path:
                    env.is_active = True
                    break

        # Cache results
        self._save_cache([env.to_dict() for env in environments])

        return environments

    def _detect_current_environment_type(self) -> str:
        """Detect the type of the currently active environment."""
        # Check for conda/mamba (highest priority)
        if os.environ.get("CONDA_PREFIX"):
            # Check if mamba is being used - improved detection logic
            if self._is_mamba_environment():
                return EnvironmentType.MAMBA
            return EnvironmentType.CONDA

        # Check for Poetry virtual environment
        if os.environ.get("POETRY_ACTIVE"):
            return EnvironmentType.POETRY

        # Check for Pipenv
        if os.environ.get("PIPENV_ACTIVE"):
            return EnvironmentType.PIPENV

        # Check for venv/virtualenv
        if os.environ.get("VIRTUAL_ENV"):
            # Try to distinguish between venv and virtualenv
            venv_path = Path(os.environ["VIRTUAL_ENV"])
            if (venv_path / "pyvenv.cfg").exists():
                return EnvironmentType.VENV
            else:
                return EnvironmentType.VIRTUALENV

        return EnvironmentType.SYSTEM

    def _is_mamba_environment(self) -> bool:
        """Check if the current environment is using mamba."""
        # Method 1: Check if mamba executable is available
        if not shutil.which("mamba"):
            return False

        # Method 2: Check if CONDA_EXE points to mamba or miniforge/mambaforge
        conda_exe = os.environ.get("CONDA_EXE", "")
        if any(
            keyword in conda_exe.lower()
            for keyword in ["mamba", "miniforge", "mambaforge"]
        ):
            return True

        # Method 3: Check if conda prefix is under mamba/miniforge directory structure
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix and any(
            keyword in conda_prefix.lower()
            for keyword in ["mamba", "miniforge", "mambaforge"]
        ):
            return True

        # Method 4: Check if mamba is the default package manager (in some setups)
        try:
            # Try to get conda info and check the channels/setup
            result = subprocess.run(
                ["conda", "info", "--json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            import json

            info = json.loads(result.stdout)

            # Check if conda-forge is the default channel (common in mamba setups)
            default_channels = info.get("default_channels", [])
            if "conda-forge" in str(default_channels) and len(default_channels) == 1:
                # This is a strong indicator of a mamba/miniforge setup
                return True

            # Check if the conda executable path contains mamba-related paths
            conda_exe_path = info.get("conda_build_version") or info.get(
                "conda_version"
            )
            if conda_exe_path and "mamba" in str(conda_exe_path).lower():
                return True

        except (
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        ):
            pass

        return False

    def _detect_conda_env_type(self, env_path: Path) -> str:
        """Detect whether a conda environment is actually mamba-managed."""
        # Check if the environment path contains mamba-related keywords
        path_str = str(env_path).lower()
        if any(keyword in path_str for keyword in ["mamba", "miniforge", "mambaforge"]):
            return EnvironmentType.MAMBA

        # Check if mamba is available and if this environment was created by mamba
        if shutil.which("mamba"):
            # If the global conda installation is mamba-based, treat all envs as mamba
            conda_exe = os.environ.get("CONDA_EXE", "")
            if any(
                keyword in conda_exe.lower()
                for keyword in ["mamba", "miniforge", "mambaforge"]
            ):
                return EnvironmentType.MAMBA

        return EnvironmentType.CONDA

    def _get_current_environment_path(self, env_type: str) -> Path | None:
        """Get the path of the current environment."""
        if env_type in (EnvironmentType.CONDA, EnvironmentType.MAMBA):
            conda_prefix = os.environ.get("CONDA_PREFIX")
            return Path(conda_prefix) if conda_prefix else None

        elif env_type in (
            EnvironmentType.VENV,
            EnvironmentType.VIRTUALENV,
            EnvironmentType.PIPENV,
            EnvironmentType.POETRY,
        ):
            virtual_env = os.environ.get("VIRTUAL_ENV")
            return Path(virtual_env) if virtual_env else None

        return None

    def _get_environment_name(self, env_path: Path, env_type: str) -> str:
        """Get the display name for an environment."""
        if env_type == EnvironmentType.SYSTEM:
            return "system"

        if env_type in (EnvironmentType.CONDA, EnvironmentType.MAMBA):
            # For conda, try to get the environment name from conda info
            try:
                result = subprocess.run(
                    ["conda", "info", "--json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                info = json.loads(result.stdout)
                active_prefix = info.get("active_prefix")
                if active_prefix and Path(active_prefix) == env_path:
                    envs = info.get("envs", [])
                    for env in envs:
                        if Path(env) == env_path:
                            return Path(env).name
            except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
                pass

        return env_path.name

    def _get_python_version(self, env_path: Path) -> str | None:
        """Get the Python version for an environment."""
        try:
            python_exe = self._find_python_executable(env_path)
            if python_exe and python_exe.exists():
                result = subprocess.run(
                    [str(python_exe), "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout.strip().replace("Python ", "")
        except (subprocess.CalledProcessError, OSError):
            pass

        return None

    def _find_python_executable(self, env_path: Path) -> Path | None:
        """Find the Python executable in an environment."""
        # Common locations for Python executable
        candidates = [
            env_path / "bin" / "python",
            env_path / "bin" / "python3",
            env_path / "Scripts" / "python.exe",  # Windows
            env_path / "Scripts" / "python3.exe",  # Windows
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _check_completion_installed(self, env_path: Path) -> bool:
        """Check if completion is installed in an environment."""
        # Check for completion marker files
        completion_markers = [
            env_path / ".xraylabtool_completion",
            env_path / "share" / "bash-completion" / "completions" / "xraylabtool",
            env_path / "etc" / "bash_completion.d" / "xraylabtool",
        ]

        for marker in completion_markers:
            if marker.exists():
                return True

        # Check activation scripts for completion hooks
        activation_files = [
            env_path / "bin" / "activate",
            env_path / "bin" / "activate.fish",
            env_path / "bin" / "activate.csh",
            env_path / "Scripts" / "activate.bat",  # Windows
        ]

        for activation_file in activation_files:
            if activation_file.exists():
                try:
                    content = activation_file.read_text()
                    if (
                        "xraylabtool" in content.lower()
                        and "completion" in content.lower()
                    ):
                        return True
                except Exception:
                    continue

        return False

    def _discover_venv_environments(self) -> list[EnvironmentInfo]:
        """Discover venv/virtualenv environments."""
        environments = []

        # Common locations for virtual environments
        search_paths = [
            Path.home() / ".virtualenvs",
            Path.home() / "venvs",
            Path.home() / ".venv",
            Path.cwd() / "venv",
            Path.cwd() / ".venv",
        ]

        # Also check WORKON_HOME for virtualenvwrapper
        workon_home = os.environ.get("WORKON_HOME")
        if workon_home:
            search_paths.append(Path(workon_home))

        for search_path in search_paths:
            if not search_path.exists():
                continue

            # If it's a single venv directory
            if self._is_virtual_environment(search_path):
                env_info = self._create_env_info(search_path)
                if env_info:
                    environments.append(env_info)

            # If it's a directory containing multiple venvs
            else:
                for item in search_path.iterdir():
                    if item.is_dir() and self._is_virtual_environment(item):
                        env_info = self._create_env_info(item)
                        if env_info:
                            environments.append(env_info)

        return environments

    def _discover_conda_environments(self) -> list[EnvironmentInfo]:
        """Discover conda/mamba environments."""
        environments = []

        try:
            # Get conda environments list
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)

            for env_path in data.get("envs", []):
                env_path = Path(env_path)
                if env_path.exists():
                    # Improved mamba detection for environments
                    env_type = self._detect_conda_env_type(env_path)
                    env_info = EnvironmentInfo(
                        env_type=env_type,
                        path=env_path,
                        name=env_path.name,
                        python_version=self._get_python_version(env_path),
                        has_completion=self._check_completion_installed(env_path),
                    )
                    environments.append(env_info)

        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass

        return environments

    def _discover_poetry_environments(self) -> list[EnvironmentInfo]:
        """Discover Poetry environments."""
        environments = []

        try:
            # Get Poetry environments
            result = subprocess.run(
                ["poetry", "env", "list", "--full-path"],
                capture_output=True,
                text=True,
                check=True,
            )

            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Parse Poetry env list output
                    parts = line.split()
                    if len(parts) >= 2:
                        env_path = Path(parts[-1])  # Last part is the path
                        if env_path.exists():
                            env_info = EnvironmentInfo(
                                env_type=EnvironmentType.POETRY,
                                path=env_path,
                                name=env_path.name,
                                python_version=self._get_python_version(env_path),
                                has_completion=self._check_completion_installed(
                                    env_path
                                ),
                            )
                            environments.append(env_info)

        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return environments

    def _is_virtual_environment(self, path: Path) -> bool:
        """Check if a path is a virtual environment."""
        if not path.exists() or not path.is_dir():
            return False

        # Check for common virtual environment markers
        markers = [
            path / "bin" / "activate",
            path / "Scripts" / "activate.bat",  # Windows
            path / "pyvenv.cfg",
        ]

        return any(marker.exists() for marker in markers)

    def _create_env_info(self, env_path: Path) -> EnvironmentInfo | None:
        """Create EnvironmentInfo for a virtual environment path."""
        if not self._is_virtual_environment(env_path):
            return None

        # Determine environment type
        env_type = EnvironmentType.VENV
        if (env_path / "pyvenv.cfg").exists():
            env_type = EnvironmentType.VENV
        elif "virtualenv" in str(env_path).lower():
            env_type = EnvironmentType.VIRTUALENV

        return EnvironmentInfo(
            env_type=env_type,
            path=env_path,
            name=env_path.name,
            python_version=self._get_python_version(env_path),
            has_completion=self._check_completion_installed(env_path),
        )

    def _get_system_python_version(self) -> str | None:
        """Get the system Python version."""
        try:
            result = subprocess.run(
                ["python3", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().replace("Python ", "")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                result = subprocess.run(
                    ["python", "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout.strip().replace("Python ", "")
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

    def _check_system_completion(self) -> bool:
        """Check if system-wide completion is installed."""
        system_completion_locations = [
            Path("/usr/share/bash-completion/completions/xraylabtool"),
            Path("/usr/local/share/bash-completion/completions/xraylabtool"),
            Path("/etc/bash_completion.d/xraylabtool"),
            Path.home() / ".bash_completion.d/xraylabtool",
        ]

        for location in system_completion_locations:
            try:
                if location.exists():
                    return True
            except (PermissionError, OSError):
                # Skip locations we can't access
                continue

        return False

    def _is_cache_valid(self) -> bool:
        """Check if the environment cache is still valid."""
        if not self._cache_file.exists():
            return False

        try:
            cache_age = self._cache_file.stat().st_mtime
            current_time = os.path.getmtime(self._cache_file)
            return (current_time - cache_age) < self._cache_timeout
        except OSError:
            return False

    def _load_cache(self) -> list[dict] | None:
        """Load cached environment data."""
        try:
            with open(self._cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _save_cache(self, environments: list[dict]) -> None:
        """Save environment data to cache."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(environments, f, indent=2)
        except OSError:
            pass  # Fail silently if we can't write cache

    def clear_cache(self) -> None:
        """Clear the environment cache."""
        if self._cache_file.exists():
            self._cache_file.unlink()
