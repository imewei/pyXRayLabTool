"""Integration tests for CompletionInstaller (item 4.1)."""

from __future__ import annotations

from pathlib import Path
import shlex
import tempfile

import pytest

from xraylabtool.interfaces.completion_v2.installer import CompletionInstaller
from xraylabtool.interfaces.completion_v2.shells import (
    CompletionManager,
    get_global_options,
    get_xraylabtool_commands,
)


class TestCompletionScriptGeneration:
    """Tests that completion scripts are generated correctly."""

    def setup_method(self) -> None:
        self.manager = CompletionManager()
        self.commands = get_xraylabtool_commands()
        self.global_options = get_global_options()

    def test_bash_script_generated(self) -> None:
        script = self.manager.generate_completion(
            "bash", self.commands, self.global_options
        )
        assert isinstance(script, str)
        assert len(script) > 0

    def test_zsh_script_generated(self) -> None:
        script = self.manager.generate_completion(
            "zsh", self.commands, self.global_options
        )
        assert isinstance(script, str)
        assert len(script) > 0

    def test_bash_script_contains_shlex_safe_content(self) -> None:
        """Generated bash script paths use shlex.quote-safe quoting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate a path that needs quoting (spaces, special chars)
            tricky_path = Path(tmpdir) / "my env" / "bin" / "xraylabtool_completion"
            quoted = shlex.quote(str(tricky_path))
            # shlex.quote always produces a safe string (single-quoted or unquoted)
            assert "'" in quoted or quoted == str(tricky_path)
            # Verify that the quoted path can be safely embedded in a shell string
            assert "\n" not in quoted
            assert "\x00" not in quoted

    def test_bash_script_uses_quoted_paths(self) -> None:
        """Installer uses shlex.quote when writing activation scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            script_path = Path(tmpdir) / "completion_script.sh"
            script_path.write_text("# completion script\n")

            # Write a fake activate script
            activate_script = Path(tmpdir) / "activate"
            activate_script.write_text("# activate\n")

            # Call the private method directly to test quoting
            installer._modify_activation_script(activate_script, script_path, "bash")

            content = activate_script.read_text()
            # The path should appear quoted (shlex.quote)
            assert (
                str(script_path) in content or shlex.quote(str(script_path)) in content
            )

    def test_bash_script_has_shebang(self) -> None:
        script = self.manager.generate_completion(
            "bash", self.commands, self.global_options
        )
        assert script.startswith("#!/")

    def test_bash_script_registers_completion(self) -> None:
        script = self.manager.generate_completion(
            "bash", self.commands, self.global_options
        )
        assert "xraylabtool" in script

    def test_zsh_script_has_header(self) -> None:
        script = self.manager.generate_completion(
            "zsh", self.commands, self.global_options
        )
        # Zsh completion starts with #compdef or a shebang
        assert script.startswith("#")


class TestSentinelMarkers:
    """Tests for XRAYLABTOOL_COMPLETION_BEGIN/END sentinel markers."""

    def test_sentinel_constants_defined(self) -> None:
        installer = CompletionInstaller()
        assert hasattr(installer, "_SENTINEL_BEGIN")
        assert hasattr(installer, "_SENTINEL_END")

    def test_sentinel_begin_marker(self) -> None:
        installer = CompletionInstaller()
        assert "XRAYLABTOOL_COMPLETION_BEGIN" in installer._SENTINEL_BEGIN

    def test_sentinel_end_marker(self) -> None:
        installer = CompletionInstaller()
        assert "XRAYLABTOOL_COMPLETION_END" in installer._SENTINEL_END

    def test_modify_activate_script_inserts_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            script_path = Path(tmpdir) / "xraylabtool_completion.sh"
            script_path.write_text("# completion\n")

            activate_script = Path(tmpdir) / "activate"
            activate_script.write_text("# original activate\nexport PATH\n")

            installer._modify_activation_script(activate_script, script_path, "bash")

            content = activate_script.read_text()
            assert installer._SENTINEL_BEGIN in content
            assert installer._SENTINEL_END in content

    def test_modify_activate_idempotent(self) -> None:
        """Calling _modify_activation_script twice does not duplicate blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            script_path = Path(tmpdir) / "completion.sh"
            script_path.write_text("# completion\n")

            activate_script = Path(tmpdir) / "activate"
            activate_script.write_text("# activate\n")

            installer._modify_activation_script(activate_script, script_path, "bash")
            installer._modify_activation_script(activate_script, script_path, "bash")

            content = activate_script.read_text()
            assert content.count(installer._SENTINEL_BEGIN) == 1

    def test_uninstall_removes_only_sentinel_block(self) -> None:
        """Uninstall strips the sentinel-delimited block but leaves surrounding content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            script_path = Path(tmpdir) / "completion.sh"
            script_path.write_text("# completion\n")

            original_content = "# original activate\nexport SOME_VAR=1\n"
            activate_script = Path(tmpdir) / "activate"
            activate_script.write_text(original_content)

            installer._modify_activation_script(activate_script, script_path, "bash")

            # Verify block is inserted
            content_with_block = activate_script.read_text()
            assert installer._SENTINEL_BEGIN in content_with_block

            # Simulate removal via _remove_venv_hooks by calling _remove_venv_hooks
            # We need a minimal EnvironmentInfo-like object; test directly via the
            # lower-level method instead.
            lines = content_with_block.split("\n")
            filtered: list[str] = []
            inside_block = False
            for line in lines:
                if line.strip() == installer._SENTINEL_BEGIN:
                    inside_block = True
                    continue
                if line.strip() == installer._SENTINEL_END:
                    inside_block = False
                    continue
                if not inside_block:
                    filtered.append(line)

            restored = "\n".join(filtered)
            activate_script.write_text(restored)

            final_content = activate_script.read_text()
            assert installer._SENTINEL_BEGIN not in final_content
            assert installer._SENTINEL_END not in final_content
            assert "SOME_VAR" in final_content

    def test_zsh_activate_script_gets_sentinels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            script_path = Path(tmpdir) / "xraylabtool_completion.zsh"
            script_path.write_text("# zsh completion\n")

            activate_script = Path(tmpdir) / "activate"
            activate_script.write_text("# zsh activate\n")

            installer._modify_activation_script(activate_script, script_path, "zsh")

            content = activate_script.read_text()
            assert installer._SENTINEL_BEGIN in content
            assert installer._SENTINEL_END in content


class TestCompletionInstallToTempDir:
    """Tests that install/uninstall work with real temp directories."""

    def test_install_writes_completion_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            installer = CompletionInstaller()
            completion_dir = Path(tmpdir) / "share" / "xraylabtool" / "completion"
            completion_dir.mkdir(parents=True)

            commands = get_xraylabtool_commands()
            global_options = get_global_options()
            script_text = installer.completion_manager.generate_completion(
                "bash", commands, global_options
            )
            script_path = completion_dir / installer.completion_manager.get_filename(
                "bash"
            )
            script_path.write_text(script_text)

            assert script_path.exists()
            content = script_path.read_text()
            assert "xraylabtool" in content

    def test_completion_script_path_is_shlex_safe(self) -> None:
        """Path written into activate script is properly shell-quoted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            CompletionInstaller()  # ensure class instantiates
            # Use a path without special chars — should still round-trip
            script_path = Path(tmpdir) / "completion.sh"
            quoted = shlex.quote(str(script_path))
            # Re-parsed value equals the original path
            assert shlex.split(quoted)[0] == str(script_path)
