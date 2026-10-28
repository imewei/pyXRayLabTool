"""Shell-specific completion script generators.

This module provides modular completion script generation for different shells,
replacing the large hardcoded strings with template-based generation.
"""

from abc import ABC, abstractmethod
import argparse
from typing import Any, cast


def _zsh_text(text: str) -> str:
    """``text`` made inert inside a double-quoted zsh string."""
    for ch in ("\\", '"', "$", "`"):
        text = text.replace(ch, "\\" + ch)
    return text


def _fish_text(text: str) -> str:
    """``text`` made inert inside a single-quoted fish string."""
    return text.replace("\\", "\\\\").replace("'", "\\'")


class CompletionGenerator(ABC):
    """Base class for shell completion generators."""

    def __init__(
        self, command_name: str = "xraylabtool", aliases: list[str] | None = None
    ):
        self.command_name = command_name
        # CLI command aliases that share the same completion (the ``xtool`` and ``xlt``
        # short aliases). The ``-gui`` aliases are intentionally excluded: they launch the
        # GUI and take no CLI subcommands.
        self.aliases = ["xtool", "xlt"] if aliases is None else aliases

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

        script = template.format(
            command_name=self.command_name,
            commands=command_list,
            global_options=global_opts,
            command_completions=command_completions,
        )
        # Register aliases against the same completion function.
        for alias in self.aliases:
            script += f"complete -F _{self.command_name}_complete {alias}\n"
        return script

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

    # Global flags can precede the command, so locate it among completed words.
    local command="" command_index=1
    while [[ ${{command_index}} -lt ${{COMP_CWORD}} ]]; do
        if [[ " ${{commands}} " == *" ${{COMP_WORDS[command_index]}} "* ]]; then
            command="${{COMP_WORDS[command_index]}}"
            break
        fi
        ((command_index++))
    done

    # No completed command yet: offer commands and global options.
    if [[ -z "${{command}}" ]]; then
        COMPREPLY=( $(compgen -W "${{commands}} ${{global_opts}}" -- "${{cur}}") )
        return 0
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
            option_var = cmd_name.replace("-", "_")

            if "subcommands" in cmd_info:
                # Handle commands with subcommands (like completion)
                subcommands = cmd_info.get("subcommands", {})
                subcmd_list = " ".join(subcommands.keys())

                completion_logic = f"""    # {cmd_name} command with subcommands
    if [[ "${{command}}" == "{cmd_name}" ]]; then
        local {cmd_name}_opts="{" ".join(options)}"
        local {cmd_name}_subcommands="{subcmd_list}"

        # If we have 3+ words, check for subcommand completion
        if [[ ${{COMP_CWORD}} -gt $((command_index + 1)) ]]; then
            local subcommand="${{COMP_WORDS[command_index+1]}}"
            case "${{subcommand}}" in"""

                for subcmd_name, subcmd_info in subcommands.items():
                    subcmd_options = subcmd_info.get("options", [])
                    completion_logic += f"""
                {subcmd_name})
                    local {subcmd_name}_opts="{" ".join(subcmd_options)}"
                    COMPREPLY=( $(compgen -W "${{{subcmd_name}_opts}}" -- "${{cur}}") )
                    return 0
                    ;;"""

                completion_logic += f"""
            esac
        fi

        # Complete subcommands if at the right position
        if [[ ${{COMP_CWORD}} -eq $((command_index + 1)) ]]; then
            COMPREPLY=( $(compgen -W "${{{cmd_name}_subcommands}} ${{{cmd_name}_opts}}" -- "${{cur}}") )
        else
            COMPREPLY=( $(compgen -W "${{{cmd_name}_opts}}" -- "${{cur}}") )
        fi
        return 0
    fi
"""
            else:
                # Handle regular commands
                choice_cases = ""
                for option, choices in cmd_info.get("option_choices", {}).items():
                    choice_cases += f"""
            {option})
                COMPREPLY=( $(compgen -W "{" ".join(choices)}" -- "${{cur}}") )
                return 0
                ;;"""
                completion_logic = f"""    # {cmd_name} command
    if [[ "${{command}}" == "{cmd_name}" ]]; then
        local {option_var}_opts="{" ".join(options)}"

        # Handle file completions for specific options
        case "${{prev}}" in
            --output|-o|--input|-i|--file|-f)
                COMPREPLY=( $(compgen -f -- "${{cur}}") )
                return 0
                ;;
{choice_cases}
        esac

        COMPREPLY=( $(compgen -W "${{{option_var}_opts}}" -- "${{cur}}") )
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

        script = template.format(
            command_name=self.command_name,
            command_definitions=command_definitions,
            command_args=command_args,
        )
        # Register aliases against the same completion function.
        for alias in self.aliases:
            script += f"compdef _{self.command_name} {alias}\n"
        return script

    def _get_template(self) -> str:
        """Get the Zsh completion template."""
        return """#compdef {command_name}
# {command_name} shell completion for Zsh
# Generated automatically by XRayLabTool completion system

# Load zsh completion system if not already loaded
if ! command -v _arguments >/dev/null 2>&1; then
    autoload -U compinit && compinit
fi

_{command_name}() {{
    local context state line

    _arguments -C \\
        '(-h --help){{-h,--help}}[Show help message]' \\
        '(-v --verbose){{-v,--verbose}}[Enable verbose output]' \\
        '(--version)--version[Show version information]' \\
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

# Register the completion function
compdef _{command_name} {command_name}
"""

    def _zsh_command_with_subcommands(
        self, cmd_name: str, cmd_info: dict[str, Any]
    ) -> str:
        """Build the Zsh case-statement block for a command with subcommands."""
        # Handle nested subcommands (like completion)
        cmd_args = f"""                {cmd_name})
                    _arguments \\"""

        # Add main command options
        main_options = cmd_info.get("options", [])
        for option in main_options:
            if option.startswith("--"):
                opt_name = option.replace("--", "").replace("-", " ")
                cmd_args += f"""
                        '{option}[{opt_name.title()}]' \\"""
            elif option.startswith("-"):
                opt_name = option.replace("-", "")
                cmd_args += f"""
                        '{option}[{opt_name.upper()}]' \\"""

        cmd_args += """
                        '1: :->subcommand' \\
                        '*:: :->subargs'

                    case $state in
                        subcommand)
                            local subcommands
                            subcommands=("""

        # Add subcommand definitions
        subcommands = cmd_info.get("subcommands", {})
        for subcmd_name, subcmd_info in subcommands.items():
            subcmd_desc = subcmd_info.get("description", f"{subcmd_name} subcommand")
            cmd_args += f"""
                                "{subcmd_name}:{_zsh_text(subcmd_desc)}\""""

        cmd_args += """
                            )
                            _describe 'subcommands' subcommands
                            ;;
                        subargs)
                            case $words[1] in"""

        # Add subcommand argument handling
        for subcmd_name, subcmd_info in subcommands.items():
            subcmd_options = subcmd_info.get("options", [])
            cmd_args += f"""
                                {subcmd_name})
                                    _arguments \\"""

            for option in subcmd_options:
                if option.startswith("--"):
                    opt_name = option.replace("--", "").replace("-", " ")
                    cmd_args += f"""
                                        '{option}[{opt_name.title()}]' \\"""
                elif option.startswith("-"):
                    opt_name = option.replace("-", "")
                    cmd_args += f"""
                                        '{option}[{opt_name.upper()}]' \\"""

            cmd_args = cmd_args.rstrip(" \\")
            cmd_args += """
                                    ;;"""

        cmd_args += """
                            esac
                            ;;
                    esac
                    ;;"""
        return cmd_args

    def _zsh_command_without_subcommands(
        self, cmd_name: str, cmd_info: dict[str, Any], global_options: list[str]
    ) -> str:
        """Build the Zsh case-statement block for a command with no subcommands."""
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
        return cmd_args

    def _generate_command_definitions(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Zsh command definitions."""
        definitions = []
        command_args = []

        for cmd_name, cmd_info in commands.items():
            description = cmd_info.get("description", f"Run {cmd_name} command")
            definitions.append(f'                "{cmd_name}:{_zsh_text(description)}"')

            if "subcommands" in cmd_info:
                cmd_args = self._zsh_command_with_subcommands(cmd_name, cmd_info)
            else:
                cmd_args = self._zsh_command_without_subcommands(
                    cmd_name, cmd_info, global_options
                )

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

        script = template.format(
            command_name=self.command_name,
            command_completions=command_completions,
        )
        # Aliases inherit the command's completions via --wraps.
        for alias in self.aliases:
            script += f"complete -c {alias} --wraps {self.command_name}\n"
        return script

    def _get_template(self) -> str:
        """Get the Fish completion template."""
        return """# {command_name} shell completion for Fish
# Generated automatically by XRayLabTool completion system

{command_completions}
"""

    def _fish_global_option_completions(self, global_options: list[str]) -> list[str]:
        """Build Fish completion lines for the top-level global options."""
        completions = []
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
        return completions

    def _fish_subcommand_completions(
        self, cmd_name: str, cmd_info: dict[str, Any]
    ) -> list[str]:
        """Build Fish completion lines for a command that has subcommands."""
        completions = []
        subcommands = cmd_info.get("subcommands", {})

        # Add main command options
        main_options = cmd_info.get("options", [])
        for option in main_options:
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

        # Add subcommand completions
        for subcmd_name, subcmd_info in subcommands.items():
            subcmd_desc = subcmd_info.get("description", f"{subcmd_name} subcommand")
            completions.append(
                f"complete -c {self.command_name} -f -n"
                f" '__fish_seen_subcommand_from {cmd_name}' -a"
                f" '{subcmd_name}' -d '{_fish_text(subcmd_desc)}'"
            )

            # Add subcommand options
            subcmd_options = subcmd_info.get("options", [])
            for option in subcmd_options:
                if option.startswith("--"):
                    opt_name = option.replace("--", "").replace("-", " ")
                    completions.append(
                        f"complete -c {self.command_name} -f -n"
                        f" '__fish_seen_subcommand_from {cmd_name}' -n"
                        f" 'test (count (commandline -opc)) -ge 3; and contains -- {subcmd_name} (commandline -opc)' -l {option[2:]} -d"
                        f" '{opt_name.title()}'"
                    )
                elif option.startswith("-") and len(option) == 2:
                    completions.append(
                        f"complete -c {self.command_name} -f -n"
                        f" '__fish_seen_subcommand_from {cmd_name}' -n"
                        f" 'test (count (commandline -opc)) -ge 3; and contains -- {subcmd_name} (commandline -opc)' -s {option[1]} -d"
                        " 'Short option'"
                    )
        return completions

    def _fish_regular_command_completions(
        self, cmd_name: str, cmd_info: dict[str, Any]
    ) -> list[str]:
        """Build Fish completion lines for a command with no subcommands."""
        completions = []
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
        return completions

    def _generate_command_completions(
        self, commands: dict[str, dict[str, Any]], global_options: list[str]
    ) -> str:
        """Generate Fish command completions."""
        completions = self._fish_global_option_completions(global_options)

        for cmd_name, cmd_info in commands.items():
            description = cmd_info.get("description", f"Run {cmd_name} command")
            completions.append(
                f"complete -c {self.command_name} -f -n '__fish_use_subcommand' -a"
                f" '{cmd_name}' -d '{_fish_text(description)}'"
            )

            if "subcommands" in cmd_info:
                completions.extend(
                    self._fish_subcommand_completions(cmd_name, cmd_info)
                )
            else:
                completions.extend(
                    self._fish_regular_command_completions(cmd_name, cmd_info)
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

        # One completer covers the command and all its aliases.
        names = [self.command_name, *self.aliases]
        command_names = "@(" + ", ".join(f"'{n}'" for n in names) + ")"

        return template.format(
            command_name=self.command_name,
            command_names=command_names,
            command_cases=command_cases,
            global_options=global_opts,
        )

    def _get_template(self) -> str:
        """Get the PowerShell completion template."""
        return """# {command_name} shell completion for PowerShell
# Generated automatically by XRayLabTool completion system

Register-ArgumentCompleter -Native -CommandName {command_names} -ScriptBlock {{
    param($commandName, $wordToComplete, $cursorPosition)

    $command = $wordToComplete
    $words = $command -split '\\s+'

    # Remove empty elements
    $words = $words | Where-Object {{ $_ -ne '' }}

    # Global options
    $globalOptions = @({global_options})

    if ($words.Count -le 1) {{
        # Complete main commands and global options
        $commands = @('calc', 'batch', 'compare', 'convert', 'kk', 'formula', 'atomic', 'bragg', 'list', 'completion')
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

            if "subcommands" in cmd_info:
                # Handle commands with subcommands (like completion)
                subcommands = cmd_info.get("subcommands", {})
                subcmd_list = ", ".join(f'"{subcmd}"' for subcmd in subcommands.keys())
                option_list = ", ".join(f'"{opt}"' for opt in options)

                case = f"""        '{cmd_name}' {{
            if ($words.Count -eq 2) {{
                # Complete subcommands and main options
                $subcommands = @({subcmd_list})
                $options = @({option_list})
                $completions = $subcommands + $options
                $completions | Where-Object {{ $_ -like "$wordToComplete*" }}
            }} elseif ($words.Count -gt 2) {{
                # Handle subcommand options
                $subcommand = $words[2]
                switch ($subcommand) {{"""

                for subcmd_name, subcmd_info in subcommands.items():
                    subcmd_options = subcmd_info.get("options", [])
                    subcmd_option_list = ", ".join(f'"{opt}"' for opt in subcmd_options)
                    case += f"""
                    '{subcmd_name}' {{
                        $subOptions = @({subcmd_option_list})
                        $subOptions | Where-Object {{ $_ -like "$wordToComplete*" }}
                    }}"""

                case += """
                    default {
                        $globalOptions | Where-Object { $_ -like "$wordToComplete*" }
                    }
                }
            }
        }"""
            else:
                # Handle regular commands
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
    """Get completion definitions from the CLI's current argument parser."""
    from xraylabtool.interfaces.cli.parser import create_parser

    def command_definitions(parser: argparse.ArgumentParser) -> dict[str, Any]:
        definition: dict[str, Any] = {
            "description": parser.description or "",
            "options": [],
            "arguments": [],
            "option_choices": {},
        }
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                definition["subcommands"] = {
                    name: command_definitions(subparser)
                    for name, subparser in action.choices.items()
                }
            elif action.option_strings:
                definition["options"].extend(action.option_strings)
                if action.choices is not None:
                    for option in action.option_strings:
                        definition["option_choices"][option] = [
                            str(choice) for choice in action.choices
                        ]
            else:
                definition["arguments"].append(action.dest)
        return definition

    return cast(
        "dict[str, dict[str, Any]]",
        command_definitions(create_parser())["subcommands"],
    )


def get_global_options() -> list[str]:
    """Get options accepted before the CLI subcommand."""
    from xraylabtool.interfaces.cli.parser import create_parser

    return [
        option
        for action in create_parser()._actions
        for option in action.option_strings
    ]
