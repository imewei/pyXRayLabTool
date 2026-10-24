"""Shell completion scripts must register every CLI alias."""

import pytest

from xraylabtool.interfaces.completion_v2.shells import (
    BashCompletionGenerator,
    FishCompletionGenerator,
    PowerShellCompletionGenerator,
    ZshCompletionGenerator,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "generator",
    [
        BashCompletionGenerator,
        ZshCompletionGenerator,
        FishCompletionGenerator,
        PowerShellCompletionGenerator,
    ],
)
@pytest.mark.parametrize("alias", ["xtool", "xlt"])
def test_cli_aliases_registered(generator, alias):
    script = generator().generate({"calc": {}}, ["--help"])
    assert alias in script
    assert f"{alias}-gui" not in script
