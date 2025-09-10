#!/usr/bin/env python3
"""XRayLabTool Shell Completion Installer.

A lightweight installer for shell completion functionality.
Supports Bash completion with automatic detection and installation.
"""

import argparse
import os
from pathlib import Path
import subprocess
from typing import Any

# Bash completion script content
BASH_COMPLETION_SCRIPT = """#!/bin/bash
# XRayLabTool shell completion for Bash
# This file provides shell completion for the xraylabtool CLI

_xraylabtool_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"

    # Safely get previous word
    if [[ ${COMP_CWORD} -gt 0 ]]; then
        prev="${COMP_WORDS[COMP_CWORD-1]}"
    else
        prev=""
    fi

    # Main commands
    local commands="calc batch convert formula atomic bragg list install-completion"

    # Global options
    local global_opts="--help --version --verbose -h -v"

    # Common options that appear across commands
    local output_opts="--output -o"
    local format_opts="--format"

    # If we're at the first argument level (command selection)
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${commands} ${global_opts}" -- "${cur}") )
        return 0
    fi

    # Safely get command
    local command=""
    if [[ ${#COMP_WORDS[@]} -gt 1 ]]; then
        command="${COMP_WORDS[1]}"
    fi

    case "${command}" in
        calc)
            _xraylabtool_calc_complete
            ;;
        batch)
            _xraylabtool_batch_complete
            ;;
        convert)
            _xraylabtool_convert_complete
            ;;
        formula)
            _xraylabtool_formula_complete
            ;;
        atomic)
            _xraylabtool_atomic_complete
            ;;
        bragg)
            _xraylabtool_bragg_complete
            ;;
        list)
            _xraylabtool_list_complete
            ;;
        install-completion)
            _xraylabtool_install_completion_complete
            ;;
        *)
            COMPREPLY=( $(compgen -W "${global_opts}" -- "${cur}") )
            ;;
    esac
}

_xraylabtool_calc_complete() {
    local calc_opts="--energy --density --output --format --fields --precision -e -d -o"
    local format_values="table csv json"

    case "${prev}" in
        --format)
            COMPREPLY=( $(compgen -W "${format_values}" -- "${cur}") )
            return 0
            ;;
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --energy|-e)
            # Suggest some common energy patterns
            local energy_examples="10.0 8.048 5.0,10.0,15.0 5-15:11 1-30:100:log"
            COMPREPLY=( $(compgen -W "${energy_examples}" -- "${cur}") )
            return 0
            ;;
        --density|-d)
            # Suggest some common material densities
            local density_examples="2.2 2.33 3.95 5.24 7.87"
            COMPREPLY=( $(compgen -W "${density_examples}" -- "${cur}") )
            return 0
            ;;
        --fields)
            local field_names="formula,energy_kev,dispersion_delta energy_kev,wavelength_angstrom,dispersion_delta formula,molecular_weight_g_mol,density_g_cm3"  # noqa: E501
            COMPREPLY=( $(compgen -W "${field_names}" -- "${cur}") )
            return 0
            ;;
        --precision)
            COMPREPLY=( $(compgen -W "3 4 5 6 7 8 10" -- "${cur}") )
            return 0
            ;;
        *)
            # Check if we haven't provided a formula yet
            local has_formula=0
            for ((i=2; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" =~ ^[A-Z][a-z]?[0-9]*$ ]] || \
                   [[ "${COMP_WORDS[i]}" =~ ^[A-Z][a-z]?[0-9]*[A-Z][a-z]?[0-9]*$ ]]; then  # noqa: E501
                    has_formula=1
                    break
                fi
            done

            if [[ $has_formula -eq 0 ]]; then
                # Suggest common chemical formulas
                local formulas="SiO2 Si Al2O3 Fe2O3 C TiO2 CaF2 BN"
                formulas="$formulas Al Cu Fe Ni Au Ag Pt"
                COMPREPLY=( $(compgen -W "${formulas} ${calc_opts}" -- "${cur}") )
            else
                COMPREPLY=( $(compgen -W "${calc_opts}" -- "${cur}") )
            fi
            ;;
    esac
}

_xraylabtool_batch_complete() {
    local batch_opts="--output --format --workers --fields -o"
    local format_values="csv json"

    case "${prev}" in
        --format)
            COMPREPLY=( $(compgen -W "${format_values}" -- "${cur}") )
            return 0
            ;;
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        --workers)
            COMPREPLY=( $(compgen -W "1 2 4 8 16" -- "${cur}") )
            return 0
            ;;
        --fields)
            local field_names="formula,energy_kev,dispersion_delta energy_kev,wavelength_angstrom,dispersion_delta formula,molecular_weight_g_mol,density_g_cm3"  # noqa: E501
            COMPREPLY=( $(compgen -W "${field_names}" -- "${cur}") )
            return 0
            ;;
        *)
            # Check if input file is provided (first positional arg)
            local has_input=0
            for ((i=2; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" == *.csv ]]; then
                    has_input=1
                    break
                fi
            done

            if [[ $has_input -eq 0 ]]; then
                # Complete CSV files for input
                csv_files=$(compgen -f -X '!*.csv' -- "${cur}")
                batch_completions=$(compgen -W "${batch_opts}" -- "${cur}")
                COMPREPLY=( $csv_files $batch_completions )
            else
                COMPREPLY=( $(compgen -W "${batch_opts}" -- "${cur}") )
            fi
            ;;
    esac
}

_xraylabtool_convert_complete() {
    local convert_opts="--to --output -o"
    local units="energy wavelength"

    case "${prev}" in
        --to)
            COMPREPLY=( $(compgen -W "${units}" -- "${cur}") )
            return 0
            ;;
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        energy)
            # Suggest energy values
            local energy_examples="10.0 8.048 5.0,10.0,15.0"
            COMPREPLY=( $(compgen -W "${energy_examples}" -- "${cur}") )
            return 0
            ;;
        wavelength)
            # Suggest wavelength values
            local wavelength_examples="1.24 1.54 0.8 1.0,1.2,1.4"
            COMPREPLY=( $(compgen -W "${wavelength_examples}" -- "${cur}") )
            return 0
            ;;
        *)
            # Check position for from_unit and values
            if [[ ${COMP_CWORD} -eq 2 ]]; then
                COMPREPLY=( $(compgen -W "${units}" -- "${cur}") )
            elif [[ ${COMP_CWORD} -eq 3 ]]; then
                # This is the values argument - provide examples based on unit type
                local unit_type=""
                if [[ ${#COMP_WORDS[@]} -gt 2 ]]; then
                    unit_type="${COMP_WORDS[2]}"
                fi
                if [[ "$unit_type" == "energy" ]]; then
                    local energy_examples="10.0 8.048 5.0,10.0,15.0"
                    COMPREPLY=( $(compgen -W "${energy_examples}" -- "${cur}") )
                elif [[ "$unit_type" == "wavelength" ]]; then
                    local wavelength_examples="1.24 1.54 0.8 1.0,1.2,1.4"
                    COMPREPLY=( $(compgen -W "${wavelength_examples}" -- "${cur}") )
                fi
            else
                COMPREPLY=( $(compgen -W "${convert_opts}" -- "${cur}") )
            fi
            ;;
    esac
}

_xraylabtool_formula_complete() {
    local formula_opts="--output --verbose -o -v"

    case "${prev}" in
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        *)
            # Check if formula is provided (first positional arg)
            local has_formula=0
            for ((i=2; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" =~ ^[A-Z] ]]; then
                    has_formula=1
                    break
                fi
            done

            if [[ $has_formula -eq 0 ]]; then
                # Suggest common chemical formulas
                local formulas="SiO2 Al2O3 Fe2O3 TiO2 CaF2 BN Ca10P6O26H2 C6H12O6"
                COMPREPLY=( $(compgen -W "${formulas} ${formula_opts}" -- "${cur}") )
            else
                COMPREPLY=( $(compgen -W "${formula_opts}" -- "${cur}") )
            fi
            ;;
    esac
}

_xraylabtool_atomic_complete() {
    local atomic_opts="--output -o"
    # Common chemical elements
    local elements="H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca"
    elements="$elements Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
    elements="$elements Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"

    case "${prev}" in
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        *)
            # Check if elements are provided (first positional arg)
            local has_elements=0
            for ((i=2; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" =~ ^[A-Z][a-z]?$ ]]; then
                    has_elements=1
                    break
                fi
            done

            if [[ $has_elements -eq 0 ]]; then
                # Suggest common elements and element combinations
                local element_examples="Si H,C,N,O,Si Si,Al,Fe C,N,O"
                all_completions="$elements $element_examples $atomic_opts"
                COMPREPLY=( $(compgen -W "${all_completions}" -- "${cur}") )
            else
                COMPREPLY=( $(compgen -W "${atomic_opts}" -- "${cur}") )
            fi
            ;;
    esac
}

_xraylabtool_bragg_complete() {
    local bragg_opts="--dspacing --wavelength --energy --order --output -d -w -e -o"

    case "${prev}" in
        --dspacing|-d)
            # Suggest common d-spacings
            local dspacing_examples="3.14 2.45 1.92 3.14,2.45,1.92"
            COMPREPLY=( $(compgen -W "${dspacing_examples}" -- "${cur}") )
            return 0
            ;;
        --wavelength|-w)
            # Suggest common X-ray wavelengths
            local wavelength_examples="1.54 1.24 0.8 1.39"
            COMPREPLY=( $(compgen -W "${wavelength_examples}" -- "${cur}") )
            return 0
            ;;
        --energy|-e)
            # Suggest common X-ray energies
            local energy_examples="8.048 10.0 17.478 8.0"
            COMPREPLY=( $(compgen -W "${energy_examples}" -- "${cur}") )
            return 0
            ;;
        --order)
            COMPREPLY=( $(compgen -W "1 2 3 4" -- "${cur}") )
            return 0
            ;;
        --output|-o)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${bragg_opts}" -- "${cur}") )
            ;;
    esac
}

_xraylabtool_list_complete() {
    local list_types="constants fields examples"

    # Check if type is already provided (first positional arg)
    local has_type=0
    for ((i=2; i<COMP_CWORD; i++)); do
        if [[ " $list_types " =~ " ${COMP_WORDS[i]} " ]]; then
            has_type=1
            break
        fi
    done

    if [[ $has_type -eq 0 ]]; then
        COMPREPLY=( $(compgen -W "${list_types}" -- "${cur}") )
    fi
}

_xraylabtool_install_completion_complete() {
    local completion_opts="--user --system --uninstall --test --help"
    local ve_opts="--venv --conda --all-environments --no-cleanup-session"
    local shell_types="bash zsh fish powershell"

    # Check if shell type is already provided
    local has_shell=0
    for ((i=2; i<COMP_CWORD; i++)); do
        if [[ " $shell_types " =~ " ${COMP_WORDS[i]} " ]]; then
            has_shell=1
            break
        fi
    done

    # Check if --uninstall is present to show virtual environment options
    local has_uninstall=0
    for ((i=2; i<COMP_CWORD; i++)); do
        if [[ "${COMP_WORDS[i]}" == "--uninstall" ]]; then
            has_uninstall=1
            break
        fi
    done

    if [[ $has_uninstall -eq 1 ]]; then
        # Show both regular completion options and virtual environment options when --uninstall is present
        COMPREPLY=( $(compgen -W "${completion_opts} ${ve_opts}" -- "${cur}") )
    elif [[ $has_shell -eq 0 ]]; then
        # No shell type provided yet, suggest both shell types and options
        COMPREPLY=( $(compgen -W "${shell_types} ${completion_opts}" -- "${cur}") )
    else
        # Shell type already provided, only suggest options
        COMPREPLY=( $(compgen -W "${completion_opts}" -- "${cur}") )
    fi
}

# Register the completion function
complete -F _xraylabtool_complete xraylabtool
"""

# Fish completion script
FISH_COMPLETION_SCRIPT = """#!/usr/bin/env fish
# XRayLabTool shell completion for Fish
# This file provides shell completion for the xraylabtool CLI in Fish shell

# Disable file completion for xraylabtool by default
complete -c xraylabtool -f

# Main commands
complete -c xraylabtool -n "__fish_use_subcommand" -a "calc" \
    -d "Calculate X-ray properties for a single material"
complete -c xraylabtool -n "__fish_use_subcommand" -a "batch" \
    -d "Process multiple materials from CSV file"
complete -c xraylabtool -n "__fish_use_subcommand" -a "convert" \
    -d "Convert between energy and wavelength units"
complete -c xraylabtool -n "__fish_use_subcommand" -a "formula" \
    -d "Parse and analyze chemical formulas"
complete -c xraylabtool -n "__fish_use_subcommand" -a "atomic" \
    -d "Look up atomic data for elements"
complete -c xraylabtool -n "__fish_use_subcommand" -a "bragg" \
    -d "Calculate Bragg angles for diffraction"
complete -c xraylabtool -n "__fish_use_subcommand" -a "list" \
    -d "List available data and information"
complete -c xraylabtool -n "__fish_use_subcommand" -a "install-completion" \
    -d "Install shell completion"

# Global options
complete -c xraylabtool -n "__fish_use_subcommand" -l help -s h \
    -d "Show help message"
complete -c xraylabtool -n "__fish_use_subcommand" -l version \
    -d "Show version number"
complete -c xraylabtool -n "__fish_use_subcommand" -l verbose -s v \
    -d "Enable verbose output"

# calc command
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -s e -l energy" \
    -l energy -s e -d "X-ray energy in keV"
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -s d -l density" \
    -l density -s d -d "Material density in g/cm³"
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -s o -l output" \
    -l output -s o -F -d "Output file path"
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -l format" \
    -l format -a "table csv json" -d "Output format"
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -l fields" \
    -l fields -d "Fields to include in output"
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and not __fish_seen_argument -l precision" \
    -l precision -a "3 4 5 6 7 8 10" -d "Decimal precision"

# Common chemical formulas for calc
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and __fish_is_nth_token 2" -a "SiO2 Si Al2O3 Fe2O3 C TiO2 CaF2 BN Al Cu Fe Ni Au Ag Pt" -d "Chemical formula"  # noqa: E501

# Energy examples for calc
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and __fish_seen_argument -s e -l energy; and __fish_is_cursor_after_token_with_argument -s e -l energy" -a "10.0 8.048 5.0,10.0,15.0 5-15:11 1-30:100:log" -d "Energy value(s)"  # noqa: E501

# Density examples for calc
complete -c xraylabtool -n "__fish_seen_subcommand_from calc; and __fish_seen_argument -s d -l density; and __fish_is_cursor_after_token_with_argument -s d -l density" -a "2.2 2.33 3.95 5.24 7.87" -d "Density value"  # noqa: E501

# batch command
complete -c xraylabtool -n "__fish_seen_subcommand_from batch" -F -d "Input CSV file"
complete -c xraylabtool -n "__fish_seen_subcommand_from batch; and not __fish_seen_argument -s o -l output" -l output -s o -F -d "Output file path"  # noqa: E501
complete -c xraylabtool -n "__fish_seen_subcommand_from batch; and not __fish_seen_argument -l format" -l format -a "csv json" -d "Output format"  # noqa: E501
complete -c xraylabtool -n "__fish_seen_subcommand_from batch; and not __fish_seen_argument -l workers" -l workers -a "1 2 4 8 16" -d "Number of workers"  # noqa: E501
complete -c xraylabtool -n "__fish_seen_subcommand_from batch; and not __fish_seen_argument -l fields" -l fields -d "Fields to include"

# convert command
complete -c xraylabtool -n "__fish_seen_subcommand_from convert; and __fish_is_nth_token 2" -a "energy wavelength" -d "From unit"
complete -c xraylabtool -n "__fish_seen_subcommand_from convert; and not __fish_seen_argument -l to" -l to -a "energy wavelength" -d "To unit"
complete -c xraylabtool -n "__fish_seen_subcommand_from convert; and not __fish_seen_argument -s o -l output" -l output -s o -F -d "Output file"

# Energy values for convert
complete -c xraylabtool -n "__fish_seen_subcommand_from convert; and __fish_is_nth_token 3; and __fish_seen_argument energy" -a "10.0 8.048 5.0,10.0,15.0" -d "Energy value(s)"

# Wavelength values for convert
complete -c xraylabtool -n "__fish_seen_subcommand_from convert; and __fish_is_nth_token 3; and __fish_seen_argument wavelength" -a "1.24 1.54 0.8 1.0,1.2,1.4" -d "Wavelength value(s)"

# formula command
complete -c xraylabtool -n "__fish_seen_subcommand_from formula; and __fish_is_nth_token 2" -a "SiO2 Al2O3 Fe2O3 TiO2 CaF2 BN Ca10P6O26H2 C6H12O6" -d "Chemical formula"
complete -c xraylabtool -n "__fish_seen_subcommand_from formula; and not __fish_seen_argument -s o -l output" -l output -s o -F -d "Output file"
complete -c xraylabtool -n "__fish_seen_subcommand_from formula; and not __fish_seen_argument -s v -l verbose" -l verbose -s v -d "Verbose output"

# atomic command
set -l elements "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"
complete -c xraylabtool -n "__fish_seen_subcommand_from atomic; and __fish_is_nth_token 2" -a "$elements" -d "Element symbol(s)"
complete -c xraylabtool -n "__fish_seen_subcommand_from atomic; and not __fish_seen_argument -s o -l output" -l output -s o -F -d "Output file"

# bragg command
complete -c xraylabtool -n "__fish_seen_subcommand_from bragg; and not __fish_seen_argument -s d -l dspacing" -l dspacing -s d -a "3.14 2.45 1.92 3.14,2.45,1.92" -d "D-spacing in Angstroms"
complete -c xraylabtool -n "__fish_seen_subcommand_from bragg; and not __fish_seen_argument -s w -l wavelength" -l wavelength -s w -a "1.54 1.24 0.8 1.39" -d "X-ray wavelength"
complete -c xraylabtool -n "__fish_seen_subcommand_from bragg; and not __fish_seen_argument -s e -l energy" -l energy -s e -a "8.048 10.0 17.478 8.0" -d "X-ray energy in keV"
complete -c xraylabtool -n "__fish_seen_subcommand_from bragg; and not __fish_seen_argument -l order" -l order -a "1 2 3 4" -d "Diffraction order"
complete -c xraylabtool -n "__fish_seen_subcommand_from bragg; and not __fish_seen_argument -s o -l output" -l output -s o -F -d "Output file"

# list command
complete -c xraylabtool -n "__fish_seen_subcommand_from list; and __fish_is_nth_token 2" -a "constants fields examples" -d "Information type"

# install-completion command
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and __fish_is_nth_token 2" -a "bash zsh fish powershell" -d "Shell type"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and not __fish_seen_argument -l user" -l user -d "Install for current user (default)"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and not __fish_seen_argument -l system" -l system -d "Install system-wide"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and not __fish_seen_argument -l test" -l test -d "Test completion installation"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and not __fish_seen_argument -l uninstall" -l uninstall -d "Uninstall completion"

# Virtual environment options for install-completion (when --uninstall is present)
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and __fish_seen_argument -l uninstall; and not __fish_seen_argument -l venv" -l venv -d "Remove completion from virtual environment activation scripts"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and __fish_seen_argument -l uninstall; and not __fish_seen_argument -l conda" -l conda -d "Remove completion from conda/mamba environment hooks"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and __fish_seen_argument -l uninstall; and not __fish_seen_argument -l all-environments" -l all-environments -d "Remove completion from all discovered environments"
complete -c xraylabtool -n "__fish_seen_subcommand_from install-completion; and __fish_seen_argument -l uninstall; and not __fish_seen_argument -l no-cleanup-session" -l no-cleanup-session -d "Skip cleaning up completion from current shell session"

# Helper functions for Fish completion
function __fish_is_cursor_after_token_with_argument
    set -l token $argv[1]
    set -l alt_token $argv[2]
    set -l cmd (commandline -poc)
    set -l cursor_pos (commandline -C)

    for i in (seq (count $cmd))
        if test "$cmd[$i]" = "$token" -o "$cmd[$i]" = "$alt_token"
            return 0
        end
    end
    return 1
end
"""

# PowerShell completion script
POWERSHELL_COMPLETION_SCRIPT = """# XRayLabTool shell completion for PowerShell
# This file provides shell completion for the xraylabtool CLI in PowerShell

# Main completion function
function _xraylabtool_complete {
    param($wordToComplete, $commandAst, $cursorPosition)

    $words = $commandAst.CommandElements | ForEach-Object { $_.ToString() }
    $commandWord = $words[0]

    # Skip if not xraylabtool
    if ($commandWord -ne 'xraylabtool') {
        return
    }

    $position = $words.Count - 1
    if ($wordToComplete) {
        $position -= 1
    }

    # Main commands
    $commands = @('calc', 'batch', 'convert', 'formula', 'atomic', 'bragg', 'list', 'install-completion')

    # Global options
    $globalOptions = @('--help', '-h', '--version', '--verbose', '-v')

    # If we're at the first argument level (command selection)
    if ($position -eq 0) {
        $completions = $commands + $globalOptions
        return $completions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        }
    }

    # Get the subcommand
    $subcommand = $words[1]

    switch ($subcommand) {
        'calc' {
            $calcOptions = @('--energy', '-e', '--density', '-d', '--output', '-o', '--format', '--fields', '--precision')
            $formatValues = @('table', 'csv', 'json')
            $formulas = @('SiO2', 'Si', 'Al2O3', 'Fe2O3', 'C', 'TiO2', 'CaF2', 'BN', 'Al', 'Cu', 'Fe', 'Ni', 'Au', 'Ag', 'Pt')

            # Check previous word for context-specific completions
            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            switch ($prevWord) {
                '--format' {
                    return $formatValues | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--output', '-o') } {
                    # File completion for output
                    return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                    }
                }
                { $_ -in @('--energy', '-e') } {
                    $energyExamples = @('10.0', '8.048', '5.0,10.0,15.0', '5-15:11', '1-30:100:log')
                    return $energyExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--density', '-d') } {
                    $densityExamples = @('2.2', '2.33', '3.95', '5.24', '7.87')
                    return $densityExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                default {
                    $completions = $calcOptions + $formulas
                    return $completions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
            }
        }

        'batch' {
            $batchOptions = @('--output', '-o', '--format', '--workers', '--fields')
            $formatValues = @('csv', 'json')

            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            switch ($prevWord) {
                '--format' {
                    return $formatValues | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--output', '-o') } {
                    return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                    }
                }
                '--workers' {
                    $workerExamples = @('1', '2', '4', '8', '16')
                    return $workerExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                default {
                    # Check if input file is provided
                    $hasInput = $words | Where-Object { $_ -like "*.csv" }
                    if (-not $hasInput) {
                        # Complete CSV files for input
                        $csvFiles = Get-ChildItem -Path . -File -Filter "*.csv" | Where-Object { $_.Name -like "$wordToComplete*" }
                        $completions = $csvFiles | ForEach-Object {
                            [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                        }
                        $completions += $batchOptions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                        }
                        return $completions
                    } else {
                        return $batchOptions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                        }
                    }
                }
            }
        }

        'convert' {
            $convertOptions = @('--to', '--output', '-o')
            $units = @('energy', 'wavelength')

            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            switch ($prevWord) {
                '--to' {
                    return $units | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--output', '-o') } {
                    return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                    }
                }
                default {
                    # Position-based completion for convert command
                    if ($position -eq 1) {
                        return $units | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                        }
                    } else {
                        return $convertOptions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                        }
                    }
                }
            }
        }

        'formula' {
            $formulaOptions = @('--output', '-o', '--verbose', '-v')
            $formulas = @('SiO2', 'Al2O3', 'Fe2O3', 'TiO2', 'CaF2', 'BN', 'Ca10P6O26H2', 'C6H12O6')

            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            if ($prevWord -in @('--output', '-o')) {
                return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                    [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                }
            } else {
                $completions = $formulaOptions + $formulas
                return $completions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                }
            }
        }

        'atomic' {
            $atomicOptions = @('--output', '-o')
            $elements = @('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Fe', 'Ni', 'Cu', 'Zn', 'Ag', 'Au')

            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            if ($prevWord -in @('--output', '-o')) {
                return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                    [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                }
            } else {
                $completions = $atomicOptions + $elements
                return $completions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                    [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                }
            }
        }

        'bragg' {
            $braggOptions = @('--dspacing', '-d', '--wavelength', '-w', '--energy', '-e', '--order', '--output', '-o')

            $prevWord = if ($position -gt 1) { $words[$position] } else { '' }

            switch ($prevWord) {
                { $_ -in @('--dspacing', '-d') } {
                    $dspacingExamples = @('3.14', '2.45', '1.92', '3.14,2.45,1.92')
                    return $dspacingExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--wavelength', '-w') } {
                    $wavelengthExamples = @('1.54', '1.24', '0.8', '1.39')
                    return $wavelengthExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--energy', '-e') } {
                    $energyExamples = @('8.048', '10.0', '17.478', '8.0')
                    return $energyExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                '--order' {
                    $orderExamples = @('1', '2', '3', '4')
                    return $orderExamples | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                { $_ -in @('--output', '-o') } {
                    return Get-ChildItem -Path . -File | Where-Object { $_.Name -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_.Name, $_.Name, 'ProviderItem', $_.Name)
                    }
                }
                default {
                    return $braggOptions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
            }
        }

        'list' {
            $listTypes = @('constants', 'fields', 'examples')
            return $listTypes | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
            }
        }

        'install-completion' {
            $shells = @('bash', 'zsh', 'fish', 'powershell')
            $completionOptions = @('--user', '--system', '--uninstall', '--test', '--help')

            # Add virtual environment options if --uninstall is present
            $veOptions = @('--venv', '--conda', '--all-environments', '--no-cleanup-session')
            $hasUninstall = $words -contains '--uninstall'

            if ($hasUninstall) {
                $completions = $shells + $completionOptions + $veOptions
            } else {
                $completions = $shells + $completionOptions
            }

            return $completions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
            }
        }

        default {
            return $globalOptions | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
                [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
            }
        }
    }
}

# Register the completion function
Register-ArgumentCompleter -CommandName xraylabtool -ScriptBlock ${function:_xraylabtool_complete}
"""


class CompletionInstaller:
    """Handles installation of shell completion for XRayLabTool."""

    def __init__(self) -> None:
        """Initialize the completion installer."""
        pass

    def get_bash_completion_dir(self) -> Path | None:
        """Find the appropriate bash completion directory."""
        # Common bash completion directories in order of preference
        candidates = [
            Path("/usr/share/bash-completion/completions"),
            Path("/usr/local/share/bash-completion/completions"),
            Path.home() / ".bash_completion.d",
            Path("/etc/bash_completion.d"),
        ]

        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_dir():
                    return candidate
            except (PermissionError, OSError):
                # Skip directories we can't access
                continue

        return None

    def get_user_bash_completion_dir(self) -> Path:
        """Get or create user bash completion directory."""
        user_dir = Path.home() / ".bash_completion.d"
        user_dir.mkdir(exist_ok=True)
        return user_dir

    def install_completion(
        self, shell_type: str | None = None, system_wide: bool = False
    ) -> bool:
        """Install shell completion script for specified shell type."""
        # Check current environment context first
        current_env_path = self._get_current_environment_path()
        env_type = self._detect_environment_type()

        if not system_wide and not current_env_path:
            print("⚠ No active virtual environment detected.")
            print(
                "  Please activate a virtual environment or use --system for system-wide installation."
            )
            return False

        if current_env_path and not system_wide:
            print(
                f"Installing completion in {env_type} environment: {current_env_path.name}"
            )

        # Auto-detect shell if not specified
        if shell_type is None:
            shell_env = os.environ.get("SHELL", "")
            if "bash" in shell_env:
                shell_type = "bash"
            elif "zsh" in shell_env:
                shell_type = "zsh"
            elif "fish" in shell_env:
                shell_type = "fish"
            else:
                shell_type = "bash"  # Default to bash

        print(f"Installing {shell_type} completion...")

        # Check for bash-completion system if installing for bash
        if shell_type == "bash":
            bash_completion_check = subprocess.run(
                ["bash", "-c", "type complete >/dev/null 2>&1"],
                check=False,
                capture_output=True,
            )
            if bash_completion_check.returncode != 0:
                print("⚠ Warning: bash-completion system may not be available")
                print("  For full functionality, install bash-completion@2:")
                print("  brew install bash-completion@2")
                print("  Then configure your ~/.bash_profile")
                print("  Proceeding with installation anyway...")

        if shell_type == "fish":
            return self._install_fish_completion(system_wide)
        elif shell_type == "powershell":
            return self._install_powershell_completion(system_wide)
        else:
            # For bash and zsh, we use the bash completion script
            return self._install_bash_completion(shell_type, system_wide)

    def _install_fish_completion(self, system_wide: bool = False) -> bool:
        """Install Fish shell completion."""
        if system_wide:
            # Fish system completions directory
            fish_dirs = [
                Path("/usr/share/fish/vendor_completions.d"),
                Path("/usr/local/share/fish/vendor_completions.d"),
                Path("/opt/homebrew/share/fish/vendor_completions.d"),
            ]
            target_dir = None
            for fish_dir in fish_dirs:
                if fish_dir.exists():
                    target_dir = fish_dir
                    break

            if target_dir is None:
                print(
                    "Error: No Fish completion directory found for system-wide installation"
                )
                return False
        else:
            # User Fish completions directory
            target_dir = Path.home() / ".config" / "fish" / "completions"
            target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / "xraylabtool.fish"

        try:
            if system_wide:
                # Write to temp file first, then copy with sudo
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".fish", delete=False
                ) as temp_file:
                    temp_file.write(FISH_COMPLETION_SCRIPT)
                    temp_path = temp_file.name

                subprocess.run(["sudo", "cp", temp_path, str(target_file)], check=True)

                # Clean up temp file
                os.unlink(temp_path)
                print(f"✓ Installed Fish completion to {target_file} (system-wide)")
            else:
                # Direct write for user installation
                target_file.write_text(FISH_COMPLETION_SCRIPT)
                print(f"✓ Installed Fish completion to {target_file} (user)")

            print("Fish completion installed! Restart your Fish shell to use it.")
            return True

        except subprocess.CalledProcessError:
            print(
                "Error: Failed to install Fish completion script (permission denied?)"
            )
            return False
        except Exception as e:
            print(f"Error: Failed to install Fish completion script: {e}")
            return False

    def _install_powershell_completion(self, system_wide: bool = False) -> bool:
        """Install PowerShell completion script."""
        if system_wide:
            # PowerShell system-wide modules directory
            powershell_dirs = [
                Path("C:/Program Files/WindowsPowerShell/Modules"),
                Path("C:/Program Files/PowerShell/7/Modules"),
                Path("/usr/local/share/powershell/Modules"),  # Linux/macOS
            ]
            target_dir = None
            for ps_dir in powershell_dirs:
                if ps_dir.exists():
                    target_dir = ps_dir / "XRayLabTool"
                    break

            if target_dir is None:
                print(
                    "Error: No PowerShell modules directory found for system-wide installation"
                )
                print("Try installing for current user only (without --system flag)")
                return False
        else:
            # User PowerShell profile directory
            home = Path.home()
            if os.name == "nt":  # Windows
                target_dir = (
                    home / "Documents" / "WindowsPowerShell" / "Modules" / "XRayLabTool"
                )
            else:  # Unix-like (Linux, macOS)
                # Check for PowerShell on Unix
                ps_config_dir = home / ".config" / "powershell"
                if not ps_config_dir.exists():
                    ps_config_dir = home / ".local" / "share" / "powershell" / "Modules"
                target_dir = ps_config_dir / "XRayLabTool"

        # Create target directory
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / "XRayLabTool.psm1"

            if system_wide:
                # Write to temp file first, then copy with sudo/elevated permissions
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".psm1", delete=False
                ) as temp_file:
                    temp_file.write(POWERSHELL_COMPLETION_SCRIPT)
                    temp_path = temp_file.name

                if os.name == "nt":
                    # Windows - would need elevated permissions
                    print(
                        "⚠ System-wide PowerShell installation requires administrator privileges"
                    )
                    print("Please run as administrator or use --user flag")
                    os.unlink(temp_path)
                    return False
                else:
                    # Unix-like systems
                    subprocess.run(
                        ["sudo", "cp", temp_path, str(target_file)], check=True
                    )
                    os.unlink(temp_path)
                    print(
                        f"✓ Installed PowerShell completion to {target_file} (system-wide)"
                    )
            else:
                # Direct write for user installation
                target_file.write_text(POWERSHELL_COMPLETION_SCRIPT)
                print(f"✓ Installed PowerShell completion to {target_file} (user)")

            # Add installation instructions
            print(
                "\nTo activate PowerShell completion, add this to your PowerShell profile:"
            )
            print(f"Import-Module {target_dir}")

            # Try to detect PowerShell profile location
            if os.name == "nt":
                profile_hint = "$PROFILE (usually ~\\Documents\\WindowsPowerShell\\Microsoft.PowerShell_profile.ps1)"
            else:
                profile_hint = "~/.config/powershell/Microsoft.PowerShell_profile.ps1"

            print(f"\nYour PowerShell profile is typically at: {profile_hint}")
            print("If the profile doesn't exist, create it first.")

            return True

        except subprocess.CalledProcessError:
            print(
                "Error: Failed to install PowerShell completion script (permission denied?)"
            )
            return False
        except Exception as e:
            print(f"Error: Failed to install PowerShell completion script: {e}")
            return False

    def _install_bash_completion(
        self, shell_type: str, system_wide: bool = False
    ) -> bool:
        """Install bash/zsh completion script."""
        if system_wide:
            target_dir = self.get_bash_completion_dir()
            if target_dir is None:
                print(
                    "Error: No bash completion directory found for system-wide installation"
                )
                print("Try installing for current user only (without --system flag)")
                return False
        else:
            target_dir = self.get_user_bash_completion_dir()

        target_file = target_dir / "xraylabtool"

        try:
            if system_wide:
                # Write to temp file first, then copy with sudo
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".bash", delete=False
                ) as temp_file:
                    temp_file.write(BASH_COMPLETION_SCRIPT)
                    temp_path = temp_file.name

                subprocess.run(["sudo", "cp", temp_path, str(target_file)], check=True)

                # Clean up temp file
                os.unlink(temp_path)
                print(
                    f"✓ Installed {shell_type} completion to {target_file} (system-wide)"
                )
            else:
                # Direct write for user installation
                target_file.write_text(BASH_COMPLETION_SCRIPT)
                print(f"✓ Installed {shell_type} completion to {target_file} (user)")

            # Add sourcing instruction for user installation
            if not system_wide:
                self._add_bash_completion_sourcing(shell_type)

            return True

        except subprocess.CalledProcessError:
            print("Error: Failed to install completion script (permission denied?)")
            return False
        except Exception as e:
            print(f"Error: Failed to install completion script: {e}")
            return False

    # Keep backward compatibility
    def install_bash_completion(self, system_wide: bool = False) -> bool:
        """Install bash completion script. (Deprecated: use install_completion)"""
        return self.install_completion("bash", system_wide)

    def _add_bash_completion_sourcing(self, shell_type: str | None = None) -> None:
        """Add sourcing of bash completion to user's shell config."""
        # Detect shell type if not provided
        if shell_type is None:
            shell_env = os.environ.get("SHELL", "")
            shell_type = "zsh" if "zsh" in shell_env else "bash"

        # Shell configuration files
        bashrc = Path.home() / ".bashrc"
        bash_profile = Path.home() / ".bash_profile"
        zshrc = Path.home() / ".zshrc"

        sourcing_line = "# XRayLabTool completion"

        # Determine shell type and config file
        if shell_type == "zsh":
            # Zsh configuration
            target_file = zshrc
            # For Zsh, we need to enable bash completion compatibility first
            zsh_setup = """
# XRayLabTool completion
# Enable bash completion compatibility in Zsh
autoload -U +X compinit && compinit
autoload -U +X bashcompinit && bashcompinit
source ~/.bash_completion.d/xraylabtool"""
            sourcing_cmd = zsh_setup.strip()
            reload_cmd = "source ~/.zshrc"
        else:
            # Bash or other shells
            sourcing_cmd = "source ~/.bash_completion.d/xraylabtool"
            reload_cmd = "source ~/.bashrc"
            # Choose the appropriate file
            target_file = bashrc if bashrc.exists() else bash_profile

        if target_file.exists():
            content = target_file.read_text()
            # Check if any part of our completion is already there
            if "source ~/.bash_completion.d/xraylabtool" not in content:
                with open(target_file, "a") as f:
                    # Only write the sourcing line once, and include it in the sourcing_cmd for zsh
                    if shell_type == "zsh":
                        f.write(f"\n{sourcing_cmd}\n")
                    else:
                        f.write(f"\n{sourcing_line}\n{sourcing_cmd}\n")
                print(f"✓ Added completion sourcing to {target_file}")
                print(f"  Please restart your shell or run: {reload_cmd}")
            else:
                print("✓ Completion sourcing already present in shell config")

    def uninstall_bash_completion(self, system_wide: bool = False) -> bool:
        """Uninstall bash completion script."""
        if system_wide:
            target_dir = self.get_bash_completion_dir()
            if target_dir is None:
                print("Error: No bash completion directory found")
                return False
        else:
            target_dir = self.get_user_bash_completion_dir()

        target_file = target_dir / "xraylabtool"

        if not target_file.exists():
            print("Bash completion is not installed")
            return True

        try:
            if system_wide:
                subprocess.run(["sudo", "rm", str(target_file)], check=True)
                print(f"✓ Removed bash completion from {target_file} (system-wide)")
            else:
                target_file.unlink()
                print(f"✓ Removed bash completion from {target_file} (user)")

            # Also remove from shell configs
            self._remove_completion_from_shell_configs()

            return True

        except subprocess.CalledProcessError:
            print("Error: Failed to remove completion script (permission denied?)")
            return False

    def _remove_completion_from_shell_configs(self) -> None:
        """Remove completion sourcing from shell configuration files."""
        configs = [
            Path.home() / ".bashrc",
            Path.home() / ".bash_profile",
            Path.home() / ".zshrc",
        ]

        for config_file in configs:
            if config_file.exists():
                try:
                    content = config_file.read_text()
                    lines = content.split("\n")
                    new_lines = []
                    skip_next = False

                    for i, line in enumerate(lines):
                        if skip_next:
                            skip_next = False
                            continue
                        if "# XRayLabTool completion" in line:
                            # Skip this line and potentially the next few
                            skip_next = True
                            # Look ahead for related lines
                            for j in range(i + 1, min(i + 5, len(lines))):
                                if "xraylabtool" in lines[j].lower():
                                    skip_next = True
                                else:
                                    break
                        elif "source ~/.bash_completion.d/xraylabtool" in line or (
                            "bashcompinit" in line
                            and i > 0
                            and "XRayLabTool" in lines[i - 1]
                        ):
                            continue
                        else:
                            new_lines.append(line)

                    new_content = "\n".join(new_lines)
                    if new_content != content:
                        config_file.write_text(new_content)
                        print(f"✓ Removed completion sourcing from {config_file}")
                except Exception:
                    pass  # Silently continue if we can't modify the file

    def uninstall_completion(
        self,
        shell_type: str | None = None,
        system_wide: bool = False,
    ) -> bool:
        """
        Uninstall completion for specified shell type and environment.

        Parameters
        ----------
        shell_type : str, optional
            Shell type (bash, zsh, fish, powershell). Auto-detected if None.
        system_wide : bool, default False
            Remove system-wide shell completion installations
        venv : bool, default False
            Remove completion from virtual environment activation scripts
        conda : bool, default False
            Remove completion from conda environment hooks
        all_environments : bool, default False
            Remove completion from all discovered environments
        cleanup_session : bool, default True
            Clean up completion from current active shell session

        Returns
        -------
        bool
            True if uninstall succeeded, False otherwise
        """
        if shell_type is None:
            shell_type = self._detect_shell()

        print(f"Uninstalling {shell_type} completion...")

        # If system-wide uninstall is requested
        if system_wide:
            print("🧹 Discovering all environments with xraylabtool completion...")
            environments = self._discover_all_environments()

            total_found = sum(len(env_list) for env_list in environments.values())
            if total_found == 0:
                print("✓ No xraylabtool completions found in any environment")
                return True

            print(f"Found completions in {total_found} location(s):")
            for env_type, env_list in environments.items():
                if env_list:
                    print(f"  {env_type}: {len(env_list)} environment(s)")

            # Remove from all discovered environments
            if environments["venv"]:
                for venv_env in environments["venv"]:
                    venv_path = (
                        venv_env["path"] if isinstance(venv_env, dict) else venv_env
                    )
                    if isinstance(venv_path, Path):
                        if not self._uninstall_venv_completion(venv_path):
                            success = False

            if environments["conda"]:
                for conda_env in environments["conda"]:
                    conda_path = (
                        conda_env["path"] if isinstance(conda_env, dict) else conda_env
                    )
                    if isinstance(conda_path, Path):
                        if not self._uninstall_conda_completion(conda_path):
                            success = False

            if environments["system"]:
                if not self._uninstall_system_completion(shell_type, system_wide):
                    success = False

        else:
            # Try to uninstall from current environment context
            current_env_path = self._get_current_environment_path()
            env_type = self._detect_environment_type()

            if current_env_path and env_type == "conda":
                success = self._uninstall_conda_completion(current_env_path)
            elif current_env_path and env_type in ["venv", "pipenv"]:
                success = self._uninstall_venv_completion(current_env_path)
            else:
                # Fall back to system uninstall
                success = self._uninstall_system_completion(shell_type, False)

        # Clean up active shell session
        if success:
            self._cleanup_active_session(shell_type)

        if success:
            print("✅ Shell completion uninstallation completed successfully")
        else:
            print("⚠ Some uninstallation steps failed - check output above")

        return success

    def _uninstall_system_completion(
        self, shell_type: str, system_wide: bool = False
    ) -> bool:
        """Uninstall traditional system completion from shell config files."""
        if shell_type == "fish":
            return self._uninstall_fish_completion(system_wide)
        elif shell_type == "powershell":
            return self._uninstall_powershell_completion(system_wide)
        else:
            return self.uninstall_bash_completion(system_wide)

    def _uninstall_fish_completion(self, system_wide: bool = False) -> bool:
        """Uninstall Fish shell completion."""
        if system_wide:
            fish_dirs = [
                Path("/usr/share/fish/vendor_completions.d"),
                Path("/usr/local/share/fish/vendor_completions.d"),
                Path("/opt/homebrew/share/fish/vendor_completions.d"),
            ]
            target_file = None
            for fish_dir in fish_dirs:
                potential_file = fish_dir / "xraylabtool.fish"
                if potential_file.exists():
                    target_file = potential_file
                    break
        else:
            target_file = (
                Path.home() / ".config" / "fish" / "completions" / "xraylabtool.fish"
            )

        if target_file is None or not target_file.exists():
            print("Fish completion is not installed")
            return True

        try:
            if system_wide:
                subprocess.run(["sudo", "rm", str(target_file)], check=True)
                print(f"✓ Removed Fish completion from {target_file} (system-wide)")
            else:
                target_file.unlink()
                print(f"✓ Removed Fish completion from {target_file} (user)")
            return True
        except subprocess.CalledProcessError:
            print("Error: Failed to remove Fish completion script (permission denied?)")
            return False
        except Exception as e:
            print(f"Error: Failed to remove Fish completion script: {e}")
            return False

    def _uninstall_powershell_completion(self, system_wide: bool = False) -> bool:
        """Uninstall PowerShell completion script."""
        if system_wide:
            # PowerShell system-wide modules directory
            powershell_dirs = [
                Path("C:/Program Files/WindowsPowerShell/Modules"),
                Path("C:/Program Files/PowerShell/7/Modules"),
                Path("/usr/local/share/powershell/Modules"),  # Linux/macOS
            ]
            target_dir = None
            for ps_dir in powershell_dirs:
                potential_dir = ps_dir / "XRayLabTool"
                if potential_dir.exists():
                    target_dir = potential_dir
                    break

            if target_dir is None:
                print("⚠ No PowerShell completion found in system directories")
                return True
        else:
            # User PowerShell profile directory
            home = Path.home()
            if os.name == "nt":  # Windows
                target_dir = (
                    home / "Documents" / "WindowsPowerShell" / "Modules" / "XRayLabTool"
                )
            else:  # Unix-like (Linux, macOS)
                ps_config_dir = home / ".config" / "powershell"
                if not ps_config_dir.exists():
                    ps_config_dir = home / ".local" / "share" / "powershell" / "Modules"
                target_dir = ps_config_dir / "XRayLabTool"

        if not target_dir.exists():
            print(f"⚠ PowerShell completion directory not found: {target_dir}")
            return True

        try:
            if system_wide and os.name == "nt":
                print(
                    "⚠ System-wide PowerShell uninstallation requires administrator privileges"
                )
                print("Please run as administrator or use --user flag")
                return False
            elif system_wide:
                # Unix-like systems
                subprocess.run(["sudo", "rm", "-rf", str(target_dir)], check=True)
                print(
                    f"✓ Removed PowerShell completion from {target_dir} (system-wide)"
                )
            else:
                # User installation - direct removal
                import shutil as shutil_module

                shutil_module.rmtree(target_dir)
                print(f"✓ Removed PowerShell completion from {target_dir} (user)")

            print("PowerShell completion removed. You may need to restart PowerShell.")
            return True

        except subprocess.CalledProcessError:
            print("Error: Failed to remove PowerShell completion (permission denied?)")
            return False
        except Exception as e:
            print(f"Error: Failed to remove PowerShell completion: {e}")
            return False

    def _detect_shell(self) -> str:
        """Detect the current shell type."""
        shell_env = os.environ.get("SHELL", "")
        if "fish" in shell_env:
            return "fish"
        elif "zsh" in shell_env:
            return "zsh"
        else:
            return "bash"

    def _check_shell_available(self, shell_type: str) -> bool:
        """Check if the specified shell is available on the system."""
        shell_commands = {
            "bash": ["bash", "--version"],
            "zsh": ["zsh", "--version"],
            "fish": ["fish", "--version"],
            "powershell": ["pwsh", "--version"],
        }

        # For PowerShell, also try 'powershell' command as fallback
        if shell_type == "powershell":
            for ps_cmd in ["pwsh", "powershell"]:
                try:
                    subprocess.run(
                        [ps_cmd, "--version"],
                        capture_output=True,
                        check=True,
                        timeout=5,
                    )
                    return True
                except (
                    subprocess.CalledProcessError,
                    FileNotFoundError,
                    subprocess.TimeoutExpired,
                ):
                    continue
            return False

        # For other shells, check the standard command
        if shell_type not in shell_commands:
            return False

        try:
            subprocess.run(
                shell_commands[shell_type], capture_output=True, check=True, timeout=5
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def _detect_environment_type(self) -> str:
        """Detect the type of Python environment."""
        if os.environ.get("CONDA_PREFIX"):
            # Both conda and mamba set CONDA_PREFIX
            return "conda"
        elif os.environ.get("VIRTUAL_ENV"):
            return "venv"
        elif os.environ.get("PIPENV_ACTIVE"):
            return "pipenv"
        else:
            return "system"

    def _get_current_environment_path(self) -> Path | None:
        """Get the path of the current virtual environment."""
        env_type = self._detect_environment_type()
        if env_type == "conda" and os.environ.get("CONDA_PREFIX"):
            return Path(os.environ["CONDA_PREFIX"])
        elif (
            env_type == "venv"
            and os.environ.get("VIRTUAL_ENV")
            or env_type == "pipenv"
            and os.environ.get("VIRTUAL_ENV")
        ):
            return Path(os.environ["VIRTUAL_ENV"])
        return None

    def _is_venv_completion_installed(self, venv_path: Path) -> bool:
        """Check if virtual environment has completion modifications."""
        if not venv_path.exists():
            return False

        # Check for completion directories
        completions_dir = venv_path / "completions"
        if completions_dir.exists():
            return True

        # Check for modified activation scripts
        activate_script = venv_path / "bin" / "activate"
        if activate_script.exists():
            try:
                content = activate_script.read_text()
                if "xraylabtool" in content.lower() and "completion" in content.lower():
                    return True
            except Exception:
                pass

        # Check for fish activation script
        activate_fish = venv_path / "bin" / "activate.fish"
        if activate_fish.exists():
            try:
                content = activate_fish.read_text()
                if "xraylabtool" in content.lower() and "completion" in content.lower():
                    return True
            except Exception:
                pass

        # Check for PowerShell activation script
        activate_ps = venv_path / "bin" / "Activate.ps1"
        if not activate_ps.exists():
            activate_ps = venv_path / "Scripts" / "Activate.ps1"  # Windows path
        if activate_ps.exists():
            try:
                content = activate_ps.read_text()
                if "xraylabtool" in content.lower() and "completion" in content.lower():
                    return True
            except Exception:
                pass

        return False

    def _is_conda_completion_installed(self, conda_prefix: Path) -> bool:
        """Check if conda environment has completion hooks."""
        if not conda_prefix.exists():
            return False

        # Check for conda activate.d/deactivate.d hooks
        activate_d = conda_prefix / "etc" / "conda" / "activate.d"
        deactivate_d = conda_prefix / "etc" / "conda" / "deactivate.d"

        for hooks_dir in [activate_d, deactivate_d]:
            if hooks_dir.exists():
                for _hook_file in hooks_dir.glob("*xraylabtool*"):
                    return True

        # Check for completion directories
        completions_dir = conda_prefix / "completions"
        return bool(completions_dir.exists())

    def _detect_virtual_environments(self) -> list[dict[str, str | Path | bool]]:
        """Detect all virtual environments with xraylabtool completions."""
        environments: list[dict[str, str | Path | bool]] = []

        # Check current environment
        current_path = self._get_current_environment_path()
        if current_path:
            env_type = self._detect_environment_type()
            if env_type == "conda" and self._is_conda_completion_installed(
                current_path
            ):
                environments.append(
                    {
                        "type": "conda",
                        "path": current_path,
                        "name": current_path.name,
                        "current": True,
                    }
                )
            elif env_type in ["venv", "pipenv"] and self._is_venv_completion_installed(
                current_path
            ):
                environments.append(
                    {
                        "type": env_type,
                        "path": current_path,
                        "name": current_path.name,
                        "current": True,
                    }
                )

        # TODO: Add scanning for other environments in common locations
        # This would scan places like ~/envs/, ~/.conda/envs/, etc.
        # For now, we focus on the current environment to avoid complexity

        return environments

    def _uninstall_venv_completion(self, venv_path: Path | None = None) -> bool:
        """Remove completion from virtual environment activation scripts."""
        if venv_path is None:
            venv_path = self._get_current_environment_path()
            if venv_path is None:
                print("Error: No virtual environment detected")
                return False

        if not self._is_venv_completion_installed(venv_path):
            print("Virtual environment completion is not installed")
            return True

        success = True

        # Remove completion directories
        completions_dir = venv_path / "completions"
        if completions_dir.exists():
            try:
                import shutil

                shutil.rmtree(completions_dir)
                print(f"✓ Removed completion directory: {completions_dir}")
            except Exception as e:
                print(f"Error: Failed to remove completion directory: {e}")
                success = False

        # Restore activation scripts from backups
        activation_scripts = [
            ("bin/activate", "bin/activate.backup"),
            ("bin/activate.fish", "bin/activate.fish.backup"),
            ("bin/Activate.ps1", "bin/Activate.ps1.backup"),
            ("Scripts/Activate.ps1", "Scripts/Activate.ps1.backup"),  # Windows
        ]

        for script_path, backup_path in activation_scripts:
            script_file = venv_path / script_path
            backup_file = venv_path / backup_path

            if backup_file.exists():
                try:
                    if script_file.exists():
                        script_file.unlink()
                    backup_file.rename(script_file)
                    print(f"✓ Restored {script_path} from backup")
                except Exception as e:
                    print(f"Error: Failed to restore {script_path}: {e}")
                    success = False
            elif script_file.exists():
                # No backup found, try to clean the script manually
                try:
                    content = script_file.read_text()
                    lines = content.split("\n")
                    new_lines = []
                    skip_next = False

                    for line in lines:
                        if skip_next:
                            skip_next = False
                            continue

                        line_lower = line.lower()
                        if "xraylabtool" in line_lower and "completion" in line_lower:
                            # Skip xraylabtool completion lines
                            if line.strip().endswith("\\"):
                                skip_next = True
                            continue
                        elif "_XRAYLABTOOL_COMPLETION" in line:
                            continue
                        else:
                            new_lines.append(line)

                    new_content = "\n".join(new_lines)
                    if new_content != content:
                        script_file.write_text(new_content)
                        print(f"✓ Cleaned {script_path} (no backup found)")
                except Exception as e:
                    print(f"Warning: Could not clean {script_path}: {e}")

        return success

    def _uninstall_conda_completion(self, conda_prefix: Path | None = None) -> bool:
        """Remove completion from conda/mamba environment hooks."""
        if conda_prefix is None:
            if self._detect_environment_type() != "conda":
                print("Error: No conda/mamba environment detected")
                return False
            conda_prefix = self._get_current_environment_path()

        if conda_prefix is None or not self._is_conda_completion_installed(
            conda_prefix
        ):
            print("Conda/mamba environment completion is not installed")
            return True

        success = True

        # Remove completion directories
        completions_dir = conda_prefix / "completions"
        if completions_dir.exists():
            try:
                import shutil

                shutil.rmtree(completions_dir)
                print(f"✓ Removed completion directory: {completions_dir}")
            except Exception as e:
                print(f"Error: Failed to remove completion directory: {e}")
                success = False

        # Remove conda/mamba activate.d/deactivate.d hooks
        hook_dirs = [
            conda_prefix / "etc" / "conda" / "activate.d",
            conda_prefix / "etc" / "conda" / "deactivate.d",
        ]

        for hook_dir in hook_dirs:
            if hook_dir.exists():
                for hook_file in hook_dir.glob("*xraylabtool*"):
                    try:
                        hook_file.unlink()
                        print(f"✓ Removed conda/mamba hook: {hook_file}")
                    except Exception as e:
                        print(
                            f"Error: Failed to remove conda/mamba hook {hook_file}: {e}"
                        )
                        success = False

        return success

    def _cleanup_active_session(self, shell_type: str | None = None) -> bool:
        """Remove completions from current active shell session."""
        if shell_type is None:
            shell_type = self._detect_shell()

        try:
            if shell_type in ["bash", "zsh"]:
                # Remove completion function from current session
                subprocess.run(
                    [
                        shell_type,
                        "-c",
                        "complete -r xraylabtool 2>/dev/null; unset -f _xraylabtool_complete 2>/dev/null",
                    ],
                    check=False,
                    capture_output=True,
                )
                print(f"✓ Removed completion from active {shell_type} session")

            elif shell_type == "fish":
                # Remove Fish completions from current session
                subprocess.run(
                    ["fish", "-c", "complete -c xraylabtool -e"],
                    check=False,
                    capture_output=True,
                )
                print("✓ Removed completion from active Fish session")

            # Clean up environment variables
            env_vars = ["_XRAYLABTOOL_COMPLETION_ACTIVE"]
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]

            return True
        except Exception as e:
            print(f"Warning: Could not clean active session: {e}")
            return False

    def _discover_all_environments(
        self,
    ) -> dict[str, list[dict[str, str | Path | bool]]]:
        """Discover all Python environments with xraylabtool completions installed."""
        environments: dict[str, list[dict[str, str | Path | bool]]] = {
            "venv": [],
            "conda": [],
            "system": [],
        }

        # Find conda environments
        conda_base = self._get_conda_base_path()
        if conda_base:
            envs_dir = conda_base / "envs"
            if envs_dir.exists():
                for env_dir in envs_dir.iterdir():
                    if env_dir.is_dir() and self._is_conda_completion_installed(
                        env_dir
                    ):
                        environments["conda"].append(
                            {
                                "type": "conda",
                                "path": env_dir,
                                "name": env_dir.name,
                                "current": False,
                            }
                        )

        # Find virtual environments (this is more challenging as they can be anywhere)
        # We'll check common locations and recently activated environments
        venv_candidates = []

        # Check current virtual env
        current_venv = os.environ.get("VIRTUAL_ENV")
        if current_venv:
            venv_candidates.append(Path(current_venv))

        # Check for common venv locations relative to current directory
        common_venv_names = ["venv", "env", ".venv", ".env", "virtualenv"]
        for name in common_venv_names:
            venv_path = Path.cwd() / name
            if venv_path.exists() and (venv_path / "bin" / "activate").exists():
                venv_candidates.append(venv_path)

        # Check if any venvs have completion installed
        for venv_path in venv_candidates:
            if self._is_venv_completion_installed(venv_path):
                environments["venv"].append(
                    {
                        "type": "venv",
                        "path": venv_path,
                        "name": venv_path.name,
                        "current": str(venv_path) == os.environ.get("VIRTUAL_ENV", ""),
                    }
                )

        # Check system shell config files
        if self._is_system_completion_installed():
            environments["system"].append(
                {
                    "type": "system",
                    "path": "system",
                    "name": "system",
                    "current": False,
                }
            )

        return environments

    def _get_conda_base_path(self) -> Path | None:
        """Get the conda/mamba base installation path."""
        # Try CONDA_EXE environment variable (set by both conda and mamba)
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            return Path(conda_exe).parent.parent

        # Try MAMBA_EXE environment variable (mamba-specific)
        mamba_exe = os.environ.get("MAMBA_EXE")
        if mamba_exe:
            return Path(mamba_exe).parent.parent

        # Try common conda/mamba installation paths
        common_paths = [
            Path.home() / "miniconda3",
            Path.home() / "mambaforge",
            Path.home() / "miniforge3",
            Path.home() / "anaconda3",
            Path("/opt/conda"),
            Path("/opt/mambaforge"),
            Path("/usr/local/conda"),
            Path("/usr/local/mambaforge"),
        ]

        for path in common_paths:
            # Check for either conda or mamba executable
            if path.exists() and (
                (path / "bin" / "conda").exists() or (path / "bin" / "mamba").exists()
            ):
                return path

        return None

    def _is_system_completion_installed(self) -> bool:
        """Check if system-wide completion is installed in shell config files."""
        shell_configs = {
            "bash": [Path.home() / ".bashrc", Path.home() / ".bash_profile"],
            "zsh": [Path.home() / ".zshrc"],
            "fish": [Path.home() / ".config" / "fish" / "config.fish"],
        }

        for _shell_type, config_files in shell_configs.items():
            for config_file in config_files:
                if config_file.exists():
                    try:
                        content = config_file.read_text()
                        if "xraylabtool" in content and "completion" in content:
                            return True
                    except Exception:
                        continue

        return False

    def test_completion(self, shell_type: str | None = None) -> bool:
        """Test if completion is working for specified shell."""
        if shell_type is None:
            shell_type = self._detect_shell()

        try:
            # Check if xraylabtool command is available
            subprocess.run(["which", "xraylabtool"], capture_output=True, check=True)
            print("✓ xraylabtool command found in PATH")

            if shell_type == "fish":
                # First check if fish shell is available
                if not self._check_shell_available("fish"):
                    print("⚠ Fish shell not found")
                    print("  Install Fish shell: brew install fish")
                    return False

                # Test Fish completion
                result = subprocess.run(
                    ["fish", "-c", "complete -C 'xraylabtool ' | head -5"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0 and result.stdout.strip():
                    print("✓ Fish completion appears to be loaded")
                    return True
                else:
                    print("⚠ Fish completion may not be loaded yet")
                    print("  Try restarting your Fish shell")
                    return False
            elif shell_type == "zsh":
                # Test Zsh completion
                result = subprocess.run(
                    [
                        "zsh",
                        "-c",
                        "source ~/.zshrc 2>/dev/null && type _xraylabtool_complete",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    print("✓ Zsh completion appears to be loaded")
                    return True
                else:
                    print("⚠ Zsh completion may not be loaded yet")
                    print("  Try restarting your shell or run: source ~/.zshrc")
                    return False
            elif shell_type == "powershell":
                # Test PowerShell completion
                # Check if PowerShell is available
                if not self._check_shell_available("powershell"):
                    print("⚠ PowerShell not found")
                    print(
                        "  Install PowerShell from https://github.com/PowerShell/PowerShell"
                    )
                    return False

                # Determine which PowerShell executable is available
                ps_exe = None
                for ps_cmd in ["pwsh", "powershell"]:
                    try:
                        subprocess.run(
                            [ps_cmd, "-NoProfile", "-Command", "exit"],
                            capture_output=True,
                            check=True,
                            timeout=5,
                        )
                        ps_exe = ps_cmd
                        break
                    except (
                        subprocess.CalledProcessError,
                        FileNotFoundError,
                        subprocess.TimeoutExpired,
                    ):
                        continue

                if ps_exe is None:
                    print("⚠ PowerShell executable not found")
                    print("  Install PowerShell (pwsh or powershell)")
                    return False

                # Test if the completion module can be imported
                test_command = """
                    try {
                        $module = Get-ChildItem -Path ($env:PSModulePath -split ';' | Where-Object {$_}) -Recurse -Filter 'XRayLabTool.psm1' -ErrorAction SilentlyContinue | Select-Object -First 1
                        if ($module) {
                            Import-Module $module.FullName -ErrorAction SilentlyContinue
                            if (Get-Command _xraylabtool_complete -ErrorAction SilentlyContinue) {
                                Write-Output "SUCCESS"
                            } else {
                                Write-Output "FUNCTION_NOT_FOUND"
                            }
                        } else {
                            Write-Output "MODULE_NOT_FOUND"
                        }
                    } catch {
                        Write-Output "ERROR: $($_.Exception.Message)"
                    }
                """

                result = subprocess.run(
                    [ps_exe, "-NoProfile", "-Command", test_command],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output == "SUCCESS":
                        print("✓ PowerShell completion appears to be loaded")
                        return True
                    elif output == "MODULE_NOT_FOUND":
                        print("⚠ PowerShell completion module not found")
                        print(
                            "  Try reinstalling with: xraylabtool --install-completion powershell"
                        )
                        return False
                    elif output == "FUNCTION_NOT_FOUND":
                        print(
                            "⚠ PowerShell completion module found but function not loaded"
                        )
                        print("  Check your PowerShell profile configuration")
                        return False
                    else:
                        print(f"⚠ PowerShell completion test failed: {output}")
                        return False
                else:
                    print("⚠ PowerShell completion test failed")
                    print("  Check PowerShell installation and module configuration")
                    return False
            else:
                # Test Bash completion
                # First check if bash-completion is available
                bash_completion_check = subprocess.run(
                    ["bash", "-c", "source ~/.bashrc 2>/dev/null && type complete"],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if bash_completion_check.returncode != 0:
                    print("⚠ Bash completion system not available")
                    print("  Install bash-completion via Homebrew:")
                    print("  brew install bash-completion@2")
                    print("  Then add this to ~/.bash_profile:")
                    print(
                        "  export BASH_COMPLETION_COMPAT_DIR='/opt/homebrew/etc/bash_completion.d'"
                    )
                    print(
                        "  [[ -r '/opt/homebrew/etc/profile.d/bash_completion.sh' ]] && . '/opt/homebrew/etc/profile.d/bash_completion.sh'"
                    )
                    return False

                result = subprocess.run(
                    [
                        "bash",
                        "-c",
                        "source ~/.bashrc 2>/dev/null && complete -p xraylabtool 2>/dev/null",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0 and "xraylabtool" in result.stdout:
                    print("✓ Bash completion appears to be loaded")
                    return True
                else:
                    # Check if our completion function exists
                    func_check = subprocess.run(
                        [
                            "bash",
                            "-c",
                            "source ~/.bashrc 2>/dev/null && type _xraylabtool_complete 2>/dev/null",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )

                    if func_check.returncode == 0:
                        print("⚠ Completion function loaded but completion not active")
                        print("  This may be due to missing bash-completion package.")
                        print(
                            "  Install bash-completion@2 via Homebrew and configure your shell."
                        )
                    else:
                        print("⚠ Bash completion may not be loaded yet")
                        print("  Try restarting your shell or run: source ~/.bashrc")
                    return False

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # More specific error handling
            if "xraylabtool" in str(e):
                print("⚠ xraylabtool command not found in PATH")
                print("  Make sure the package is installed and available")
            else:
                print(f"⚠ Error testing completion: {type(e).__name__}")
                print("  Check shell installation and completion setup")
            return False


def install_completion_main(args: argparse.Namespace | Any) -> int:
    """Main function for the install-completion command."""
    installer = CompletionInstaller()

    if args.uninstall:
        success = installer.uninstall_completion(
            shell_type=args.shell,
            system_wide=args.system,
        )
        return 0 if success else 1

    elif args.test:
        installer.test_completion(shell_type=args.shell)
        return 0

    else:
        # Default action: install
        success = installer.install_completion(
            shell_type=args.shell, system_wide=args.system
        )
        if success:
            print("\n🎉 Installation completed!")
            print("You can now use tab completion with xraylabtool commands.")
            shell_type = args.shell or installer._detect_shell()
            if not args.system:
                if shell_type == "fish":
                    print("Restart your Fish shell to use completion.")
                elif shell_type == "zsh":
                    print("Please restart your shell or run: source ~/.zshrc")
                else:
                    print("Please restart your shell or run: source ~/.bashrc")
        return 0 if success else 1
