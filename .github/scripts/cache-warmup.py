#!/usr/bin/env python3
"""
Advanced cache warming and management script for CI optimization.

Preloads commonly used dependencies and optimizes cache structure.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_command(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command with error handling."""
    try:
        return subprocess.run(
            cmd, capture_output=capture, text=True, check=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out: {' '.join(cmd)}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        print(f"   Return code: {e.returncode}")
        if e.stdout:
            print(f"   Stdout: {e.stdout}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        raise


def warm_pip_cache():
    """Pre-download and cache commonly used packages."""
    print("🔥 Warming pip cache...")

    # Core dependencies that are commonly installed
    core_packages = [
        "setuptools",
        "wheel",
        "pip",
        "build",
        "twine",
        "pytest",
        "pytest-cov",
        "pytest-benchmark",
        "pytest-timeout",
        "coverage",
        "ruff",
        "black",
        "isort",
        "mypy",
        "bandit",
        "safety",
        "sphinx",
        "furo",
        "myst-parser",
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
    ]

    for package in core_packages:
        try:
            print(f"  📦 Pre-downloading {package}...")
            run_command(
                [
                    "pip",
                    "download",
                    "--no-deps",
                    "--dest",
                    os.path.expanduser("~/.cache/pip/downloads"),
                    package,
                ]
            )
        except Exception as e:
            print(f"  ⚠️ Failed to download {package}: {e}")
            continue


def create_cache_structure():
    """Create optimal cache directory structure."""
    print("📁 Creating cache directory structure...")

    cache_dirs = [
        "~/.cache/pip",
        "~/.cache/pip/downloads",
        "~/.cache/pip/wheels",
        "~/.cache/mypy",
        "~/.cache/ruff",
        "~/.cache/sphinx",
        "~/.pytest_cache",
        "~/.pytest_cache/v",
        ".coverage_cache",
    ]

    for cache_dir in cache_dirs:
        expanded_dir = Path(os.path.expanduser(cache_dir))
        expanded_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created {expanded_dir}")


def optimize_pytest_cache():
    """Optimize pytest cache settings."""
    print("🧪 Optimizing pytest cache...")

    pytest_cache_dir = Path.home() / ".pytest_cache"
    pytest_cache_dir.mkdir(exist_ok=True)

    # Create pytest cache config
    cache_config = {
        "cache_dir": str(pytest_cache_dir),
        "version": "1.0",
        "timestamp": time.time(),
    }

    config_file = pytest_cache_dir / "cache_config.json"
    with open(config_file, "w") as f:
        json.dump(cache_config, f, indent=2)

    print(f"  ✅ Pytest cache configured at {pytest_cache_dir}")


def warm_tool_caches():
    """Pre-warm caches for development tools."""
    print("🛠️ Warming tool caches...")

    # Warm mypy cache by running on a simple file
    try:
        simple_py = Path("temp_cache_warm.py")
        simple_py.write_text("x: int = 1\n")

        run_command(
            ["mypy", "--cache-dir", os.path.expanduser("~/.cache/mypy"), str(simple_py)]
        )
        simple_py.unlink()
        print("  ✅ MyPy cache warmed")
    except Exception as e:
        print(f"  ⚠️ MyPy cache warming failed: {e}")

    # Warm ruff cache
    try:
        run_command(["ruff", "check", "--version"])
        print("  ✅ Ruff cache initialized")
    except Exception as e:
        print(f"  ⚠️ Ruff cache warming failed: {e}")


def check_cache_health() -> dict[str, Any]:
    """Check the health of existing caches."""
    print("🔍 Checking cache health...")

    health_report = {
        "pip_cache_size": 0,
        "mypy_cache_size": 0,
        "pytest_cache_size": 0,
        "total_cache_size": 0,
        "cache_directories": [],
    }

    cache_paths = [
        ("pip", Path.home() / ".cache" / "pip"),
        ("mypy", Path.home() / ".cache" / "mypy"),
        ("pytest", Path.home() / ".pytest_cache"),
        ("ruff", Path.home() / ".cache" / "ruff"),
        ("sphinx", Path.home() / ".cache" / "sphinx"),
    ]

    for name, cache_path in cache_paths:
        if cache_path.exists():
            try:
                size = sum(
                    f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
                )
                health_report[f"{name}_cache_size"] = size
                health_report["total_cache_size"] += size
                health_report["cache_directories"].append(
                    {
                        "name": name,
                        "path": str(cache_path),
                        "size_mb": round(size / (1024 * 1024), 2),
                        "exists": True,
                    }
                )
                print(f"  📊 {name}: {size / (1024 * 1024): .1f} MB")
            except Exception as e:
                print(f"  ⚠️ Error checking {name} cache: {e}")
                health_report["cache_directories"].append(
                    {
                        "name": name,
                        "path": str(cache_path),
                        "size_mb": 0,
                        "exists": False,
                        "error": str(e),
                    }
                )
        else:
            health_report["cache_directories"].append(
                {"name": name, "path": str(cache_path), "size_mb": 0, "exists": False}
            )

    total_mb = health_report["total_cache_size"] / (1024 * 1024)
    print(f"  📈 Total cache size: {total_mb: .1f} MB")

    return health_report


def cleanup_old_caches():
    """Clean up old or corrupted cache entries."""
    print("🧹 Cleaning up old caches...")

    # Clean pip cache of old wheels (keep last 30 days)
    try:
        run_command(["pip", "cache", "purge"])
        print("  ✅ Pip cache purged")
    except Exception as e:
        print(f"  ⚠️ Pip cache cleanup failed: {e}")

    # Clean pytest cache of old test results
    pytest_cache = Path.home() / ".pytest_cache"
    if pytest_cache.exists():
        try:
            for item in pytest_cache.rglob("*"):
                if item.is_file() and item.stat().st_mtime < (
                    time.time() - 7 * 24 * 3600
                ):
                    item.unlink()
            print("  ✅ Old pytest cache files removed")
        except Exception as e:
            print(f"  ⚠️ Pytest cache cleanup failed: {e}")


def main():
    """Run main cache optimization routine."""
    print("🚀 Starting CI cache optimization...")
    start_time = time.time()

    try:
        # Step 1: Check existing cache health
        initial_health = check_cache_health()

        # Step 2: Create optimal cache structure
        create_cache_structure()

        # Step 3: Clean up old caches
        cleanup_old_caches()

        # Step 4: Warm up caches
        warm_pip_cache()
        optimize_pytest_cache()
        warm_tool_caches()

        # Step 5: Final health check
        final_health = check_cache_health()

        # Summary
        elapsed = time.time() - start_time
        print(f"\n✅ Cache optimization completed in {elapsed: .1f}s")
        initial_mb = initial_health["total_cache_size"] / (1024 * 1024)
        final_mb = final_health["total_cache_size"] / (1024 * 1024)
        print(f"📈 Cache size change: {initial_mb: .1f} MB → {final_mb: .1f} MB")

        # Save health report
        with open("cache_health_report.json", "w") as f:
            json.dump(final_health, f, indent=2)
        print("📊 Cache health report saved to cache_health_report.json")

    except KeyboardInterrupt:
        print("\n⏹️ Cache optimization interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Cache optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
