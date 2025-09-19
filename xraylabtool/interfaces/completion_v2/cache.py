"""Performance optimizations and caching for shell completion.

This module provides caching mechanisms and performance optimizations
to make completion faster and more responsive.
"""

import hashlib
import json
from pathlib import Path
import time
from typing import Any


class CompletionCache:
    """Fast caching system for completion data."""

    def __init__(self, cache_dir: Path | None = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".xraylabtool" / "cache"

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache timeouts (in seconds)
        self.default_timeout = 3600  # 1 hour
        self.command_cache_timeout = 86400  # 24 hours
        self.env_cache_timeout = 1800  # 30 minutes

    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)

        return hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()

    def get(self, key: str, timeout: int | None = None) -> Any | None:
        """Get cached data by key."""
        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            if timeout is None:
                timeout = self.default_timeout

            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > timeout:
                cache_file.unlink()  # Remove expired cache
                return None

            # Load cached data
            with open(cache_file) as f:
                cache_data = json.load(f)
                return cache_data.get("data")

        except (json.JSONDecodeError, OSError, KeyError):
            # Remove corrupted cache
            if cache_file.exists():
                cache_file.unlink()
            return None

    def set(self, key: str, data: Any, metadata: dict | None = None) -> None:
        """Cache data with optional metadata."""
        cache_file = self.cache_dir / f"{key}.json"

        cache_data = {
            "data": data,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        try:
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)
        except OSError:
            pass  # Fail silently if we can't write cache

    def invalidate(self, key: str) -> None:
        """Invalidate cached data."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def clear(self) -> None:
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


class CompletionDataManager:
    """Manages and caches completion data for fast access."""

    def __init__(self):
        self.cache = CompletionCache()
        self._commands_cache_key = "xraylabtool_commands"
        self._options_cache_key = "xraylabtool_options"

    def get_commands(self, force_refresh: bool = False) -> dict[str, dict]:
        """Get command definitions with caching."""
        if not force_refresh:
            cached_commands = self.cache.get(
                self._commands_cache_key, timeout=self.cache.command_cache_timeout
            )
            if cached_commands:
                return cached_commands

        # Import here to avoid circular imports
        from .shells import get_xraylabtool_commands

        commands = get_xraylabtool_commands()
        self.cache.set(self._commands_cache_key, commands)
        return commands

    def get_global_options(self, force_refresh: bool = False) -> list[str]:
        """Get global options with caching."""
        if not force_refresh:
            cached_options = self.cache.get(
                self._options_cache_key, timeout=self.cache.command_cache_timeout
            )
            if cached_options:
                return cached_options

        # Import here to avoid circular imports
        from .shells import get_global_options

        options = get_global_options()
        self.cache.set(self._options_cache_key, options)
        return options

    def get_completion_script(
        self, shell: str, force_refresh: bool = False
    ) -> str | None:
        """Get cached completion script for shell."""
        script_key = f"completion_script_{shell}"

        if not force_refresh:
            cached_script = self.cache.get(script_key)
            if cached_script:
                return cached_script

        return None

    def cache_completion_script(self, shell: str, script: str) -> None:
        """Cache completion script for shell."""
        script_key = f"completion_script_{shell}"
        self.cache.set(script_key, script)

    def invalidate_command_cache(self) -> None:
        """Invalidate command-related caches."""
        self.cache.invalidate(self._commands_cache_key)
        self.cache.invalidate(self._options_cache_key)

        # Invalidate all completion scripts
        for shell in ["bash", "zsh", "fish", "powershell"]:
            script_key = f"completion_script_{shell}"
            self.cache.invalidate(script_key)


class FastCompletionProvider:
    """Optimized completion provider for runtime performance."""

    def __init__(self):
        self.data_manager = CompletionDataManager()
        self._completion_cache = {}

    def get_command_completions(self, partial_command: str) -> list[str]:
        """Get command completions for partial input."""
        commands = self.data_manager.get_commands()

        if not partial_command:
            return list(commands.keys())

        # Fast prefix matching
        matches = []
        for cmd in commands:
            if cmd.startswith(partial_command):
                matches.append(cmd)

        return sorted(matches)

    def get_option_completions(self, command: str, partial_option: str) -> list[str]:
        """Get option completions for a command."""
        commands = self.data_manager.get_commands()
        global_options = self.data_manager.get_global_options()

        options = set(global_options)

        if command in commands:
            cmd_options = commands[command].get("options", [])
            options.update(cmd_options)

        if not partial_option:
            return sorted(options)

        # Fast prefix matching
        matches = []
        for opt in options:
            if opt.startswith(partial_option):
                matches.append(opt)

        return sorted(matches)

    def get_file_completions(self, partial_path: str) -> list[str]:
        """Get file path completions (optimized)."""
        try:
            if not partial_path:
                path = Path(".")
                pattern = "*"
            else:
                path_obj = Path(partial_path)
                if partial_path.endswith("/") or path_obj.is_dir():
                    path = path_obj
                    pattern = "*"
                else:
                    path = path_obj.parent
                    pattern = f"{path_obj.name}*"

            if not path.exists():
                return []

            # Fast globbing with limit
            matches = []
            for item in path.glob(pattern):
                if len(matches) >= 100:  # Limit results for performance
                    break

                if item.is_dir():
                    matches.append(f"{item}/")
                else:
                    matches.append(str(item))

            return sorted(matches)

        except (OSError, PermissionError):
            return []

    def warm_cache(self) -> None:
        """Pre-warm caches for better performance."""
        # Load commands and options
        self.data_manager.get_commands()
        self.data_manager.get_global_options()

        # Pre-generate common completions
        commands = self.data_manager.get_commands()
        for cmd in commands:
            self.get_option_completions(cmd, "")


class PerformanceMonitor:
    """Monitor completion performance for optimization."""

    def __init__(self):
        self.metrics = {
            "completion_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_completions": 0,
        }

    def record_completion_time(self, duration: float) -> None:
        """Record completion timing."""
        self.metrics["completion_times"].append(duration)
        self.metrics["total_completions"] += 1

        # Keep only recent measurements
        if len(self.metrics["completion_times"]) > 1000:
            self.metrics["completion_times"] = self.metrics["completion_times"][-500:]

    def record_cache_hit(self) -> None:
        """Record cache hit."""
        self.metrics["cache_hits"] += 1

    def record_cache_miss(self) -> None:
        """Record cache miss."""
        self.metrics["cache_misses"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        completion_times = self.metrics["completion_times"]

        stats = {
            "total_completions": self.metrics["total_completions"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
        }

        if completion_times:
            stats.update(
                {
                    "avg_completion_time_ms": round(
                        sum(completion_times) / len(completion_times) * 1000, 2
                    ),
                    "max_completion_time_ms": round(max(completion_times) * 1000, 2),
                    "min_completion_time_ms": round(min(completion_times) * 1000, 2),
                }
            )

        if self.metrics["cache_hits"] + self.metrics["cache_misses"] > 0:
            total_cache_requests = (
                self.metrics["cache_hits"] + self.metrics["cache_misses"]
            )
            stats["cache_hit_rate"] = round(
                self.metrics["cache_hits"] / total_cache_requests * 100, 1
            )

        return stats


# Global instances for easy access
_data_manager = None
_performance_monitor = None


def get_data_manager() -> CompletionDataManager:
    """Get global completion data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = CompletionDataManager()
    return _data_manager


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
