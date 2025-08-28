#!/bin/bash
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
            local field_names="formula,energy_kev,dispersion_delta energy_kev,wavelength_angstrom,dispersion_delta formula,molecular_weight_g_mol,density_g_cm3"
            COMPREPLY=( $(compgen -W "${field_names}" -- "${cur}") )
            return 0
            ;;
        --precision)
            COMPREPLY=( $(compgen -W "3 4 5 6 7 8 10" -- "${cur}") )
            return 0
            ;;
        *)
            # Check if we haven't provided a formula yet (should be first positional arg)
            local has_formula=0
            for ((i=2; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" =~ ^[A-Z][a-z]?[0-9]*$ ]] || [[ "${COMP_WORDS[i]}" =~ ^[A-Z][a-z]?[0-9]*[A-Z][a-z]?[0-9]*$ ]]; then
                    has_formula=1
                    break
                fi
            done

            if [[ $has_formula -eq 0 ]]; then
                # Suggest common chemical formulas
                local formulas="SiO2 Si Al2O3 Fe2O3 C TiO2 CaF2 BN Al Cu Fe Ni Au Ag Pt"
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
            local field_names="formula,energy_kev,dispersion_delta energy_kev,wavelength_angstrom,dispersion_delta formula,molecular_weight_g_mol,density_g_cm3"
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
                COMPREPLY=( $(compgen -f -X '!*.csv' -- "${cur}") $(compgen -W "${batch_opts}" -- "${cur}") )
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
    local elements="H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe"

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
                COMPREPLY=( $(compgen -W "${elements} ${element_examples} ${atomic_opts}" -- "${cur}") )
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
    COMPREPLY=( $(compgen -W "${completion_opts}" -- "${cur}") )
}

# Register the completion function
complete -F _xraylabtool_complete xraylabtool
