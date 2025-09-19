"""Modular shell completion system for XRayLabTool.

This package provides a virtual environment-centric shell completion system
that automatically activates/deactivates with virtual environment changes.
"""

from .environment import EnvironmentDetector
from .installer import CompletionInstaller

__all__ = ["CompletionInstaller", "EnvironmentDetector"]
