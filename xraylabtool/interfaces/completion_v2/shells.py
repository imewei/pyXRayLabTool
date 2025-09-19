"""Shell-specific completion script generators.

This module provides modular completion script generation for different shells,
replacing the large hardcoded strings with template-based generation.
"""

from abc import ABC, abstractmethod
from typing import Any


class CompletionGenerator(ABC):
    """Base class for shell completion generators."""

    def __init__(self, command_name: str = "xraylabtool"):
        self.command_name = command_name

    @abstractmethod
    def generate(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate completion script for this shell."""
        pass

    @property
    @abstractmethod
    def shell_name(self) -> str:
        """Name of the shell this generator supports."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for completion scripts."""
        pass


class BashCompletionGenerator(CompletionGenerator):
    """Generates Bash completion scripts."""

    @property
    def shell_name(self) -> str:
        return "bash"

    @property
    def file_extension(self) -> str:
        return ""

    def generate(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Bash completion script."""
        template = self._get_template()

        # Generate command list
        command_list = " ".join(commands.keys())

        # Generate global options
        global_opts = " ".join(global_options)

        # Generate command-specific completion logic
        command_completions = self._generate_command_completions(commands)

        return template.format(
            command_name=self.command_name,
            commands=command_list,
            global_options=global_opts,
            command_completions=command_completions,
        )

    def _get_template(self) -> str:
        """Get the Bash completion template."""
        return """#!/bin/bash
# {command_name} shell completion for Bash
# Generated automatically by XRayLabTool completion system

_{command_name}_complete() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"

    # Safely get previous word
    if [[ ${{COMP_CWORD}} -gt 0 ]]; then
        prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    else
        prev=""
    fi

    # Main commands
    local commands="{commands}"

    # Global options
    local global_opts="{global_options}"

    # If we're at the first argument level (command selection)
    if [[ ${{COMP_CWORD}} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${{commands}} ${{global_opts}}" -- "${{cur}}") )
        return 0
    fi

    # Get the command
    local command=""
    if [[ ${{#COMP_WORDS[@]}} -gt 1 ]]; then
        command="${{COMP_WORDS[1]}}"
    fi

{command_completions}

    # Default fallback
    COMPREPLY=( $(compgen -W "${{global_opts}}" -- "${{cur}}") )
}}

# Register completion
complete -F _{command_name}_complete {command_name}
"""

    def _generate_command_completions(self, commands: dict[str, dict[str, Any]]) -> str:
        """Generate command-specific completion logic."""
        completions = []

        for cmd_name, cmd_info in commands.items():
            options = cmd_info.get("options", [])
            cmd_info.get("arguments", [])

            completion_logic = f"""    # {cmd_name} command
    if [[ "${{command}}" == "{cmd_name}" ]]; then
        local {cmd_name}_opts="{" ".join(options)}"

        # Handle file completions for specific options
        case "${{prev}}" in
            --output|-o|--input|-i|--file|-f)
                COMPREPLY=( $(compgen -f -- "${{cur}}") )
                return 0
                ;;
            --format)
                COMPREPLY=( $(compgen -W "json yaml csv" -- "${{cur}}") )
                return 0
                ;;
        esac

        COMPREPLY=( $(compgen -W "${{{cmd_name}_opts}} ${{global_opts}}" -- "${{cur}}") )
        return 0
    fi
"""
            completions.append(completion_logic)

        return "\n".join(completions)


class ZshCompletionGenerator(CompletionGenerator):
    """Generates native Zsh completion scripts."""

    @property
    def shell_name(self) -> str:
        return "zsh"

    @property
    def file_extension(self) -> str:
        return ""

    def generate(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate native Zsh completion script."""
        template = self._get_template()

        # Generate command definitions
        command_data = self._generate_command_definitions(commands, global_options)
        command_definitions, command_args = command_data.split("||")

        return template.format(
            command_name=self.command_name,
            command_definitions=command_definitions,
            command_args=command_args,
        )

    def _get_template(self) -> str:
        """Get the Zsh completion template."""
        return """#compdef {command_name}
# {command_name} shell completion for Zsh
# Generated automatically by XRayLabTool completion system

_{command_name}() {{
    local context state line

    _arguments -C \\
        '(-h --help)'{{{-h,--help}}}'[Show help message]' \\
        '(-v --verbose)'{{{-v,--verbose}}}'[Enable verbose output]' \\
        '(--version)'--version'[Show version information]' \\
        '1: :->command' \\
        '*:: :->args'

    case $state in
        command)
            local commands
            commands=(
{command_definitions}
            )
            _describe 'commands' commands
            ;;
        args)
            case $words[1] in
{command_args}
            esac
            ;;
    esac
}}

_{command_name} "$@"
"""

    def _generate_command_definitions(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Zsh command definitions."""
        definitions = []
        command_args = []

        for cmd_name, cmd_info in commands.items():
            description = cmd_info.get("description", f"Run {cmd_name} command")
            definitions.append(f'                "{cmd_name}:{description}"')

            # Generate arguments for this command
            options = cmd_info.get("options", [])
            cmd_args = f"""                {cmd_name})
                    _arguments \\"""

            for option in options:
                if option.startswith("--"):
                    opt_name = option.replace("--", "").replace("-", " ")
                    cmd_args += f"""
                        '{option}[{opt_name.title()}]' \\"""
                elif option.startswith("-"):
                    opt_name = option.replace("-", "")
                    cmd_args += f"""
                        '{option}[{opt_name.upper()}]' \\"""

            # Add global options
            for option in global_options:
                if option.startswith("--"):
                    opt_name = option.replace("--", "").replace("-", " ")
                    cmd_args += f"""
                        '{option}[{opt_name.title()}]' \\"""

            cmd_args = cmd_args.rstrip(" \\")
            cmd_args += """
                    ;;"""

            command_args.append(cmd_args)

        definitions_str = "\n".join(definitions)
        command_args_str = "\n".join(command_args)

        # Return properly formatted string for template
        return f"{definitions_str}||{command_args_str}"


class FishCompletionGenerator(CompletionGenerator):
    """Generates Fish shell completion scripts."""

    @property
    def shell_name(self) -> str:
        return "fish"

    @property
    def file_extension(self) -> str:
        return ".fish"

    def generate(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Fish completion script."""
        template = self._get_template()

        # Generate command completions
        command_completions = self._generate_command_completions(
            commands, global_options
        )

        return template.format(
            command_name=self.command_name,
            command_completions=command_completions,
        )

    def _get_template(self) -> str:
        """Get the Fish completion template."""
        return """#!/usr/bin/env fish
# {command_name} shell completion for Fish
# Generated automatically by XRayLabTool completion system

{command_completions}
"""

    def _generate_command_completions(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Fish command completions."""
        completions = []

        # Global options
        for option in global_options:
            if option.startswith("--"):
                opt_name = option.replace("--", "").replace("-", " ")
                completions.append(
                    f"complete -c {self.command_name} -l {option[2:]} -d"
                    f" '{opt_name.title()}'"
                )
            elif option.startswith("-") and len(option) == 2:
                completions.append(
                    f"complete -c {self.command_name} -s {option[1]} -d 'Short option'"
                )

        # Command completions
        for cmd_name, cmd_info in commands.items():
            description = cmd_info.get("description", f"Run {cmd_name} command")
            completions.append(
                f"complete -c {self.command_name} -f -n '__fish_use_subcommand' -a"
                f" '{cmd_name}' -d '{description}'"
            )

            # Command options
            options = cmd_info.get("options", [])
            for option in options:
                if option.startswith("--"):
                    opt_name = option.replace("--", "").replace("-", " ")
                    completions.append(
                        f"complete -c {self.command_name} -f -n"
                        f" '__fish_seen_subcommand_from {cmd_name}' -l {option[2:]} -d"
                        f" '{opt_name.title()}'"
                    )
                elif option.startswith("-") and len(option) == 2:
                    completions.append(
                        f"complete -c {self.command_name} -f -n"
                        f" '__fish_seen_subcommand_from {cmd_name}' -s {option[1]} -d"
                        " 'Short option'"
                    )

        return "\n".join(completions)


class PowerShellCompletionGenerator(CompletionGenerator):
    """Generates PowerShell completion scripts."""

    @property
    def shell_name(self) -> str:
        return "powershell"

    @property
    def file_extension(self) -> str:
        return ".ps1"

    def generate(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate PowerShell completion script."""
        template = self._get_template()

        # Generate command cases
        command_cases = self._generate_command_cases(commands)

        # Generate global options
        global_opts = ", ".join(f'"{opt}"' for opt in global_options)

        return template.format(
            command_name=self.command_name,
            command_cases=command_cases,
            global_options=global_opts,
        )

    def _get_template(self) -> str:
        """Get the PowerShell completion template."""
        return """# {command_name} shell completion for PowerShell
# Generated automatically by XRayLabTool completion system

Register-ArgumentCompleter -Native -CommandName {command_name} -ScriptBlock {{
    param($commandName, $wordToComplete, $cursorPosition)

    $command = $wordToComplete
    $words = $command -split '\\s+'

    # Remove empty elements
    $words = $words | Where-Object {{ $_ -ne '' }}

    # Global options
    $globalOptions = @({global_options})

    if ($words.Count -le 1) {{
        # Complete main commands and global options
        $commands = @('calc', 'batch', 'convert', 'formula', 'atomic', 'bragg', 'list', 'install-completion')
        $completions = $commands + $globalOptions
        $completions | Where-Object {{ $_ -like "$wordToComplete*" }}
        return
    }}

    $subcommand = $words[1]

    switch ($subcommand) {{
{command_cases}
        default {{
            $globalOptions | Where-Object {{ $_ -like "$wordToComplete*" }}
        }}
    }}
}}
"""

    def _generate_command_cases(self, commands: dict[str, dict[str, Any]]) -> str:
        """Generate PowerShell command case statements."""
        cases = []

        for cmd_name, cmd_info in commands.items():
            options = cmd_info.get("options", [])
            option_list = ", ".join(f'"{opt}"' for opt in options)

            case = f"""        '{cmd_name}' {{
            $options = @({option_list})
            $options | Where-Object {{ $_ -like "$wordToComplete*" }}
        }}"""
            cases.append(case)

        return "\n".join(cases)


class CompletionManager:
    """Manages completion script generation for all supported shells."""

    def __init__(self) -> None:
        self.generators = {
            "bash": BashCompletionGenerator(),
            "zsh": ZshCompletionGenerator(),
            "fish": FishCompletionGenerator(),
            "powershell": PowerShellCompletionGenerator(),
        }

    def get_supported_shells(self) -> list[str]:
        """Get list of supported shell types."""
        return list(self.generators.keys())

    def generate_completion(
        self, shell: str, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate completion script for specified shell."""
        if shell not in self.generators:
            raise ValueError(f"Unsupported shell: {shell}")

        generator = self.generators[shell]
        return generator.generate(commands, global_options)

    def get_filename(self, shell: str, command_name: str = "xraylabtool") -> str:
        """Get the appropriate filename for a completion script."""
        if shell not in self.generators:
            raise ValueError(f"Unsupported shell: {shell}")

        generator = self.generators[shell]
        return f"{command_name}{generator.file_extension}"


def get_xraylabtool_commands() -> dict[str, dict[str, Any]]:
    """Get the XRayLabTool command definitions for completion."""
    return {
        "calc": {
            "description": "Calculate X-ray optical properties",
            "options": [
                "--material",
                "-m",
                "--energy",
                "-e",
                "--density",
                "-d",
                "--output",
                "-o",
                "--format",
                "-f",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["material"],
        },
        "batch": {
            "description": "Process multiple calculations from file",
            "options": [
                "--input",
                "-i",
                "--output",
                "-o",
                "--format",
                "-f",
                "--parallel",
                "-p",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["input_file"],
        },
        "convert": {
            "description": "Convert between file formats",
            "options": [
                "--input",
                "-i",
                "--output",
                "-o",
                "--from-format",
                "--to-format",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["input_file", "output_file"],
        },
        "formula": {
            "description": "Parse and analyze chemical formulas",
            "options": [
                "--formula",
                "-f",
                "--output",
                "-o",
                "--format",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["formula"],
        },
        "atomic": {
            "description": "Get atomic properties and data",
            "options": [
                "--element",
                "-e",
                "--property",
                "-p",
                "--output",
                "-o",
                "--format",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["element"],
        },
        "bragg": {
            "description": "Calculate Bragg diffraction conditions",
            "options": [
                "--crystal",
                "--energy",
                "-e",
                "--wavelength",
                "-w",
                "--output",
                "-o",
                "--format",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["crystal_system"],
        },
        "list": {
            "description": "List available materials, elements, or data",
            "options": [
                "--type",
                "-t",
                "--filter",
                "--output",
                "-o",
                "--format",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": ["list_type"],
        },
        "install-completion": {
            "description": "Install shell completion",
            "options": [
                "--shell",
                "-s",
                "--user",
                "--system",
                "--uninstall",
                "--list-environments",
                "--verbose",
                "-v",
                "--help",
                "-h",
            ],
            "arguments": [],
        },
    }


def get_global_options() -> list[str]:
    """Get global options available for all commands."""
    return ["--help", "-h", "--version", "--verbose", "-v", "--config", "--log-level"]
