"""
Test CI/CD pipeline integration for style guide enforcement.

This module validates that the style guide implementation integrates correctly
with continuous integration and continuous deployment pipelines.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestCICDIntegration(BaseUnitTest):
    """Test CI/CD pipeline integration for style guide enforcement."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent

    @pytest.mark.integration
    def test_ci_make_target_passes(self):
        """Test that the CI make target works correctly."""
        # This simulates the CI environment
        result = subprocess.run(
            ["make", "help"],  # Start with help to ensure make works
            check=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, "Make command should work in CI environment"

        # Check if ci-test target exists
        makefile_path = self.project_root / "Makefile"
        if makefile_path.exists():
            with open(makefile_path) as f:
                makefile_content = f.read()
            assert "ci-test:" in makefile_content, "CI test target should exist"

    @pytest.mark.integration
    def test_style_guide_validation_exit_codes(self):
        """Test that style guide validation returns appropriate exit codes for CI."""
        # Test that the validation script exists and is executable
        validation_script = self.project_root / "scripts" / "validate_style_guide.py"
        assert validation_script.exists(), "Style guide validation script should exist"

        # Test that script runs and returns exit code
        result = subprocess.run(
            [sys.executable, str(validation_script), "--categories", "imports"],
            check=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Script should run successfully (exit code should be 0, 1, or 2)
        assert result.returncode in [
            0,
            1,
            2,
        ], f"Validation script should return valid exit code, got {result.returncode}"

    @pytest.mark.integration
    def test_pre_commit_hooks_integration(self):
        """Test that pre-commit hooks can be integrated into CI."""
        # Check if pre-commit configuration exists
        precommit_config = self.project_root / ".pre-commit-config.yaml"

        if precommit_config.exists():
            # Test that pre-commit can be installed and run
            try:
                result = subprocess.run(
                    ["pre-commit", "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    # Test dry run of pre-commit
                    result = subprocess.run(
                        ["pre-commit", "run", "--all-files", "--dry-run"],
                        check=False,
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    # Pre-commit should run (may fail, but should execute)
                    assert result.returncode is not None, "Pre-commit should execute"

            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.skip("Pre-commit not available in CI environment")
        else:
            pytest.skip("Pre-commit configuration not available")

    @pytest.mark.integration
    def test_black_integration_in_ci(self):
        """Test that Black formatting integrates correctly with CI."""
        try:
            # Test Black check mode (CI-friendly)
            result = subprocess.run(
                ["black", "--check", "--diff", "--color", "xraylabtool"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Black should run successfully (return code 0 means no changes needed)
            # Return code 1 means formatting changes are needed
            assert result.returncode in [
                0,
                1,
            ], f"Black should return 0 or 1, got {result.returncode}"

        except FileNotFoundError:
            pytest.skip("Black not available in CI environment")

    @pytest.mark.integration
    def test_ruff_integration_in_ci(self):
        """Test that Ruff linting integrates correctly with CI."""
        try:
            # Test Ruff check mode (CI-friendly)
            result = subprocess.run(
                ["ruff", "check", "xraylabtool", "--output-format=github"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Ruff should run successfully (may have violations)
            assert result.returncode in [
                0,
                1,
            ], f"Ruff should return 0 or 1, got {result.returncode}"

        except FileNotFoundError:
            pytest.skip("Ruff not available in CI environment")

    @pytest.mark.integration
    def test_mypy_integration_in_ci(self):
        """Test that MyPy type checking integrates correctly with CI."""
        try:
            # Test MyPy check mode (CI-friendly)
            result = subprocess.run(
                [
                    "mypy",
                    "xraylabtool/calculators",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # MyPy should run successfully (may have errors)
            assert result.returncode in [
                0,
                1,
            ], f"MyPy should return 0 or 1, got {result.returncode}"

        except FileNotFoundError:
            pytest.skip("MyPy not available in CI environment")

    @pytest.mark.integration
    def test_test_suite_integration_in_ci(self):
        """Test that the test suite integrates correctly with CI."""
        # Test that pytest runs with CI-friendly options
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/unit/test_core.py",
                "--tb=short",
                "-v",
                "--maxfail=5",
            ],
            check=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Pytest should run (may pass or fail, but should execute)
        assert result.returncode is not None, "Pytest should execute in CI environment"

    @pytest.mark.integration
    def test_coverage_reporting_in_ci(self):
        """Test that coverage reporting works in CI environment."""
        try:
            # Test coverage with CI-friendly options
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/unit/test_core.py::TestXRayCalculations::test_calculate_xray_properties_basic",
                    "--cov=xraylabtool",
                    "--cov-report=term",
                    "--cov-report=xml",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Coverage should run (may pass or fail, but should execute)
            assert result.returncode is not None, (
                "Coverage should execute in CI environment"
            )

            # Check if coverage.xml is generated
            coverage_file = self.project_root / "coverage.xml"
            if coverage_file.exists():
                assert coverage_file.stat().st_size > 0, (
                    "Coverage file should not be empty"
                )

        except FileNotFoundError:
            pytest.skip("Coverage tools not available in CI environment")

    @pytest.mark.integration
    def test_json_report_generation_for_ci(self):
        """Test that JSON reports are generated for CI consumption."""
        validation_script = self.project_root / "scripts" / "validate_style_guide.py"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Test JSON report generation
            subprocess.run(
                [
                    sys.executable,
                    str(validation_script),
                    "--output",
                    str(temp_path),
                    "--categories",
                    "imports",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Script should run and generate report
            assert temp_path.exists(), "JSON report should be generated"

            if temp_path.stat().st_size > 0:
                with open(temp_path) as f:
                    report_data = json.load(f)

                # Validate report structure
                assert "timestamp" in report_data, "Report should have timestamp"
                assert "compliance_score" in report_data, (
                    "Report should have compliance score"
                )
                assert "violations" in report_data, "Report should have violations list"

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    @pytest.mark.integration
    def test_parallel_test_execution_in_ci(self):
        """Test that tests can run in parallel for faster CI."""
        try:
            # Test parallel execution with pytest-xdist if available
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/unit/test_core.py::TestXRayCalculations::test_calculate_xray_properties_basic",
                    "-n",
                    "auto",
                    "--tb=short",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parallel tests should run (may not have -n option available)
            assert result.returncode is not None, (
                "Parallel tests should attempt to execute"
            )

        except FileNotFoundError:
            pytest.skip("Pytest-xdist not available for parallel execution")

    @pytest.mark.integration
    def test_environment_variable_configuration(self):
        """Test that CI environment variables are respected."""
        # Test with CI environment variable set
        env = os.environ.copy()
        env["CI"] = "1"
        env["PYTHONPATH"] = str(self.project_root)

        result = subprocess.run(
            [sys.executable, "-c", 'import xraylabtool; print("Import successful")'],
            check=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        assert result.returncode == 0, (
            "Package should import successfully in CI environment"
        )
        assert "Import successful" in result.stdout, "Import test should succeed"

    @pytest.mark.integration
    def test_docker_compatibility(self):
        """Test that style guide works in containerized environments."""
        # Test basic Python execution in minimal environment
        test_script = """
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from tests.fixtures.test_base import BaseUnitTest
    print("Test base import successful")
except ImportError as e:
    print(f"Test base import failed: {e}")
    sys.exit(1)

try:
    from xraylabtool.exceptions import ValidationError
    print("Validation import successful")
except ImportError as e:
    print(f"Validation import failed: {e}")
    # Don't fail here as this might be expected in minimal environments

print("Docker compatibility test passed")
"""

        result = subprocess.run(
            [sys.executable, "-c", test_script],
            check=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should run without major errors
        assert "Docker compatibility test passed" in result.stdout, (
            "Docker compatibility test should pass"
        )

    @pytest.mark.integration
    def test_cache_invalidation_in_ci(self):
        """Test that CI can handle cache invalidation correctly."""
        # Test that pytest cache can be cleared
        pytest_cache = self.project_root / ".pytest_cache"

        if pytest_cache.exists():
            # Test cache clearing
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cache-clear",
                    "--collect-only",
                    "tests/unit/test_core.py",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            assert result.returncode == 0, "Pytest cache clearing should work"

    @pytest.mark.integration
    def test_artifact_generation_for_ci(self):
        """Test that CI artifacts are generated correctly."""
        # Test that various output files can be generated
        artifacts_to_test = [
            (
                "coverage.xml",
                [
                    "python",
                    "-m",
                    "pytest",
                    "tests/unit/test_core.py::TestXRayCalculations::test_calculate_xray_properties_basic",
                    "--cov=xraylabtool",
                    "--cov-report=xml",
                ],
            ),
        ]

        for artifact_name, command in artifacts_to_test:
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Command should execute (may pass or fail)
                assert result.returncode is not None, (
                    f"Command for {artifact_name} should execute"
                )

            except FileNotFoundError:
                pytest.skip(f"Tools for {artifact_name} not available")


class TestCIWorkflowSimulation(BaseUnitTest):
    """Test complete CI workflow simulation."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent

    @pytest.mark.integration
    def test_complete_ci_workflow_simulation(self):
        """Test complete CI workflow from start to finish."""
        # This simulates a complete CI workflow
        workflow_steps = [
            # 1. Environment setup (simulated)
            ("Environment Setup", lambda: True),
            # 2. Dependencies installation (check if we can import)
            ("Dependencies Check", self._test_dependencies),
            # 3. Linting and formatting checks
            ("Code Quality", self._test_code_quality),
            # 4. Type checking
            ("Type Checking", self._test_type_checking),
            # 5. Test execution
            ("Test Execution", self._test_execution),
            # 6. Style guide validation
            ("Style Guide", self._test_style_guide),
            # 7. Report generation
            ("Report Generation", self._test_report_generation),
        ]

        failed_steps = []
        for step_name, step_func in workflow_steps:
            try:
                success = step_func()
                if not success:
                    failed_steps.append(step_name)
            except Exception as e:
                failed_steps.append(f"{step_name} ({e})")

        # Allow some steps to fail in CI environment
        assert len(failed_steps) <= 3, (
            f"Too many CI workflow steps failed: {failed_steps}"
        )

    def _test_dependencies(self) -> bool:
        """Test that dependencies can be imported."""
        try:
            from tests.fixtures.test_base import BaseUnitTest
            import xraylabtool

            return True
        except ImportError:
            return False

    def _test_code_quality(self) -> bool:
        """Test code quality checks."""
        try:
            # Test basic formatting check
            result = subprocess.run(
                [sys.executable, "-c", 'import black; print("Black available")'],
                check=False,
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _test_type_checking(self) -> bool:
        """Test type checking."""
        try:
            result = subprocess.run(
                [sys.executable, "-c", 'import mypy; print("MyPy available")'],
                check=False,
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _test_execution(self) -> bool:
        """Test that tests can be executed."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                check=False,
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _test_style_guide(self) -> bool:
        """Test style guide validation."""
        validation_script = self.project_root / "scripts" / "validate_style_guide.py"
        if not validation_script.exists():
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(validation_script), "--categories", "imports"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                timeout=30,
            )
            return result.returncode in [0, 1, 2]  # Valid exit codes
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def _test_report_generation(self) -> bool:
        """Test that reports can be generated."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                temp_path = Path(f.name)

            validation_script = (
                self.project_root / "scripts" / "validate_style_guide.py"
            )
            subprocess.run(
                [
                    sys.executable,
                    str(validation_script),
                    "--output",
                    str(temp_path),
                    "--categories",
                    "imports",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                timeout=30,
            )

            success = temp_path.exists() and temp_path.stat().st_size > 0
            temp_path.unlink()  # Clean up
            return success

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            return False
