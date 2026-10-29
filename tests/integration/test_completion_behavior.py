"""Exercise generated completion scripts with actual shell completion inputs."""

from __future__ import annotations

import shlex
import shutil
import subprocess

import pytest

from xraylabtool.interfaces.completion_v2.shells import (
    BashCompletionGenerator,
    get_global_options,
    get_xraylabtool_commands,
)


def _bash_complete(words: list[str]) -> set[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    script = BashCompletionGenerator().generate(
        get_xraylabtool_commands(), get_global_options()
    )
    result = subprocess.run(
        [bash, "--noprofile", "--norc"],
        input=(
            script
            + f"\nCOMP_WORDS=({shlex.join(words)})\n"
            + f"COMP_CWORD={len(words) - 1}\n"
            + '_xraylabtool_complete\nprintf "%s\\n" "${COMPREPLY[@]}"\n'
        ).encode(),  # bytes: text mode would send \r\n on Windows
        capture_output=True,
        check=True,
    )
    assert not result.stderr
    return set(result.stdout.decode().split())


@pytest.mark.parametrize(
    ("command", "required", "removed"),
    [
        ("calc", {"--fields", "--precision"}, {"--wavelength"}),
        ("batch", {"--workers", "--progress", "--fields"}, {"--parallel", "--input"}),
        ("bragg", {"--dspacing", "--order", "--output"}, {"--crystal", "--hkl"}),
        ("atomic", {"--output"}, {"--property"}),
        ("convert", {"--output", "--to"}, {"--from", "--unit"}),
    ],
)
def test_bash_completes_supported_command_options(
    command: str, required: set[str], removed: set[str]
) -> None:
    completions = _bash_complete(["xraylabtool", command, "--"])
    assert required <= completions
    assert completions.isdisjoint(removed)


@pytest.mark.parametrize(
    ("command", "formats"),
    [
        ("calc", {"table", "csv", "json"}),
        ("compare", {"table", "csv", "json"}),
        ("batch", {"csv", "json"}),
        ("kk", {"csv", "json"}),
    ],
)
def test_bash_format_choices_match_command(command: str, formats: set[str]) -> None:
    assert _bash_complete(["xraylabtool", command, "--format", ""]) == formats


def test_bash_completes_legacy_completion_commands() -> None:
    assert _bash_complete(["xraylabtool", "install-"]) == {"install-completion"}
    assert "--system" in _bash_complete(["xraylabtool", "install-completion", "--"])


def test_bash_global_options_exclude_removed_flags() -> None:
    completions = _bash_complete(["xraylabtool", "--"])
    assert "--debug" in completions
    assert completions.isdisjoint({"--config", "--log-level"})


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        (["xraylabtool", "--verbose", "ca"], {"calc"}),
        (["xraylabtool", "--verbose", "calc", "--fi"], {"--fields"}),
        (["xraylabtool", "--verbose", "completion", "in"], {"install", "info"}),
        (
            ["xraylabtool", "--verbose", "completion", "install", "--sh"],
            {"--shell"},
        ),
    ],
)
def test_bash_completion_handles_leading_global_options(
    words: list[str], expected: set[str]
) -> None:
    assert _bash_complete(words) == expected
