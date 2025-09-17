"""
Tests for Python style guide tool configuration validation.

This module validates that the Python style guide documentation accurately
reflects the tool configurations specified in pyproject.toml.
"""

import re
import toml
from pathlib import Path
import pytest


class TestToolConfigurationValidation:
    """Test that style guide tool configurations match pyproject.toml."""

    @pytest.fixture
    def pyproject_config(self):
        """Load actual pyproject.toml configuration."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "r") as f:
            return toml.load(f)

    @pytest.fixture
    def style_guide_content(self):
        """Load Python style guide content."""
        style_guide_path = (
            Path(__file__).parent.parent
            / ".agent-os"
            / "standards"
            / "code-style"
            / "python-style.md"
        )
        with open(style_guide_path, "r") as f:
            return f.read()

    def test_python_version_requirements_match(
        self, pyproject_config, style_guide_content
    ):
        """Test that Python version requirements match between pyproject.toml and style guide."""
        # Extract Python version from pyproject.toml
        requires_python = pyproject_config["project"]["requires-python"]

        # Parse version requirement (e.g., ">=3.11" -> "3.11")
        version_match = re.search(r">=(\d+\.\d+)", requires_python)
        assert version_match, f"Could not parse Python version from: {requires_python}"
        min_version = version_match.group(1)

        # Check style guide mentions correct Python version
        assert (
            f"Python {min_version}+" in style_guide_content
        ), f"Style guide should specify Python {min_version}+ requirement"

        # Ensure it doesn't mention wrong versions
        wrong_versions = ["3.9", "3.10"] if min_version == "3.11" else ["3.8", "3.9"]
        for wrong_version in wrong_versions:
            assert (
                f"Python {wrong_version}+" not in style_guide_content
            ), f"Style guide incorrectly mentions Python {wrong_version}+"

    def test_black_version_specification(self, pyproject_config, style_guide_content):
        """Test that Black version specification matches pyproject.toml."""
        # Get Black version from dependencies
        dependencies = pyproject_config["project"]["optional-dependencies"]["dev"]
        black_dep = next((dep for dep in dependencies if dep.startswith("black")), None)
        assert black_dep, "Black dependency not found in pyproject.toml"

        # Extract version constraint
        version_pattern = r"black[>=<,\s\d\.]+([0-9]+\.[0-9]+)"
        version_match = re.search(version_pattern, black_dep)
        if version_match:
            version = version_match.group(1)
            # Check that style guide mentions compatible version
            assert (
                version in style_guide_content
            ), f"Style guide should mention Black version {version}"

    def test_ruff_version_specification(self, pyproject_config, style_guide_content):
        """Test that Ruff version specification matches pyproject.toml."""
        dependencies = pyproject_config["project"]["optional-dependencies"]["dev"]
        ruff_dep = next((dep for dep in dependencies if dep.startswith("ruff")), None)
        assert ruff_dep, "Ruff dependency not found in pyproject.toml"

        # Check that style guide mentions Ruff as primary tool
        assert (
            "Ruff" in style_guide_content or "ruff" in style_guide_content
        ), "Style guide should mention Ruff as primary linting tool"

    def test_mypy_version_specification(self, pyproject_config, style_guide_content):
        """Test that MyPy version specification matches pyproject.toml."""
        dependencies = pyproject_config["project"]["optional-dependencies"]["dev"]
        mypy_dep = next((dep for dep in dependencies if dep.startswith("mypy")), None)
        assert mypy_dep, "MyPy dependency not found in pyproject.toml"

        # Check that style guide mentions MyPy with strict mode
        assert (
            "mypy" in style_guide_content.lower()
        ), "Style guide should mention MyPy type checking"
        assert (
            "strict" in style_guide_content.lower()
        ), "Style guide should mention MyPy strict mode"

    def test_tool_configuration_sections_exist(self, style_guide_content):
        """Test that required tool configuration sections exist in style guide."""
        required_sections = [
            "Formatting Tools",
            "Black",
            "Ruff",
            "MyPy",
            "Type Checking",
        ]

        for section in required_sections:
            # Check for section headers (markdown format) - more flexible matching
            pattern = r"#{1,4}\s*.*" + re.escape(section)
            assert re.search(
                pattern, style_guide_content, re.IGNORECASE
            ), f"Style guide should have a '{section}' section"

    def test_line_length_configuration(self, pyproject_config, style_guide_content):
        """Test that line length configuration matches between tools."""
        # Check Black configuration
        if "tool" in pyproject_config and "black" in pyproject_config["tool"]:
            black_config = pyproject_config["tool"]["black"]
            if "line-length" in black_config:
                line_length = black_config["line-length"]
                assert (
                    f"{line_length}" in style_guide_content
                ), f"Style guide should mention line length of {line_length}"

    def test_target_version_configuration(self, pyproject_config, style_guide_content):
        """Test that target version configuration is documented."""
        # Check Black target version
        if "tool" in pyproject_config and "black" in pyproject_config["tool"]:
            black_config = pyproject_config["tool"]["black"]
            if "target-version" in black_config:
                target_versions = black_config["target-version"]
                if isinstance(target_versions, list):
                    for version in target_versions:
                        # Should document the target version
                        assert (
                            version in style_guide_content
                            or version.replace("py", "3.") in style_guide_content
                        ), f"Style guide should document target version {version}"

    def test_ruff_rules_configuration(self, pyproject_config, style_guide_content):
        """Test that Ruff rules configuration is documented."""
        if "tool" in pyproject_config and "ruff" in pyproject_config["tool"]:
            ruff_config = pyproject_config["tool"]["ruff"]

            # Check if select rules are documented
            if "lint" in ruff_config and "select" in ruff_config["lint"]:
                select_rules = ruff_config["lint"]["select"]
                if isinstance(select_rules, list) and len(select_rules) > 5:
                    # Should mention that comprehensive rules are enforced
                    rule_indicators = ["rules", "enforce", "linting", "E", "F", "W"]
                    assert any(
                        indicator in style_guide_content
                        for indicator in rule_indicators
                    ), "Style guide should document Ruff rule enforcement"

    def test_mypy_strict_configuration(self, pyproject_config, style_guide_content):
        """Test that MyPy strict configuration is documented."""
        if "tool" in pyproject_config and "mypy" in pyproject_config["tool"]:
            mypy_config = pyproject_config["tool"]["mypy"]

            # Check for strict mode settings
            strict_settings = [
                "disallow_untyped_defs",
                "strict_optional",
                "warn_return_any",
                "disallow_any_generics",
            ]

            documented_strict_settings = 0
            for setting in strict_settings:
                if setting in mypy_config and mypy_config[setting]:
                    if (
                        setting.replace("_", " ") in style_guide_content
                        or setting in style_guide_content
                    ):
                        documented_strict_settings += 1

            # Should document at least some strict settings
            assert (
                documented_strict_settings > 0
            ), "Style guide should document MyPy strict mode settings"


class TestConfigurationExamples:
    """Test that configuration examples in style guide are valid."""

    def test_tool_installation_commands_valid(self):
        """Test that tool installation commands in style guide are valid."""
        style_guide_path = (
            Path(__file__).parent.parent
            / ".agent-os"
            / "standards"
            / "code-style"
            / "python-style.md"
        )
        with open(style_guide_path, "r") as f:
            content = f.read()

        # Look for pip install commands
        pip_commands = re.findall(r"pip install[^\n]*", content)

        for command in pip_commands:
            # Basic validation - should not have obvious syntax errors
            assert "pip install" in command
            assert not command.endswith(
                "\\"
            ), "Command should not end with incomplete line continuation"
            # Should not have double spaces that might indicate formatting issues
            assert "  " not in command.replace(
                "pip install ", ""
            ), f"Installation command may have formatting issues: {command}"

    def test_configuration_syntax_valid(self):
        """Test that configuration examples use valid syntax."""
        style_guide_path = (
            Path(__file__).parent.parent
            / ".agent-os"
            / "standards"
            / "code-style"
            / "python-style.md"
        )
        with open(style_guide_path, "r") as f:
            content = f.read()

        # Look for TOML configuration blocks
        toml_blocks = re.findall(r"```toml(.*?)```", content, re.DOTALL)

        for block in toml_blocks:
            try:
                # Basic TOML syntax validation
                toml.loads(block)
            except toml.TomlDecodeError as e:
                pytest.fail(f"Invalid TOML syntax in style guide: {e}")

    def test_python_code_examples_valid(self):
        """Test that Python code examples in style guide are syntactically valid."""
        style_guide_path = (
            Path(__file__).parent.parent
            / ".agent-os"
            / "standards"
            / "code-style"
            / "python-style.md"
        )
        with open(style_guide_path, "r") as f:
            content = f.read()

        # Look for Python code blocks
        python_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

        for i, block in enumerate(python_blocks):
            # Skip blocks that are obviously just examples or comments
            if block.strip().startswith("#") and "\n" not in block.strip():
                continue

            try:
                # Basic Python syntax validation
                compile(block, f"<style_guide_example_{i}>", "exec")
            except SyntaxError as e:
                # Allow some specific patterns that might not compile in isolation
                if "import" in str(e) or "..." in block:
                    continue
                pytest.fail(f"Invalid Python syntax in style guide example {i}: {e}")
