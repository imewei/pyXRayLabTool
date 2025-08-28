# XRayLabTool Shell Completion

This directory contains lightweight shell completion support for the XRayLabTool CLI.

## Features

- **Multi-Shell Support**: Full command and option completion for Bash, Zsh, Fish, and PowerShell
- **Auto-Detection**: Automatically detects your shell and configures accordingly
- **Shell-Specific Installation**: Choose specific shell or let the installer auto-detect
- **Context-Aware Suggestions**: Provides relevant suggestions based on command context
- **File Completion**: Intelligent file completion for input/output files
- **Chemical Formula Suggestions**: Common chemical formulas and element symbols
- **Parameter Value Hints**: Suggests common values for energies, densities, etc.

## Quick Installation

### Using the Built-in CLI Command (Recommended)

XRayLabTool supports two installation syntaxes:

#### Flag Syntax (Simple and Quick)
```bash
# Install for current shell (auto-detected)
xraylabtool --install-completion

# Install for specific shell
xraylabtool --install-completion bash
xraylabtool --install-completion zsh
xraylabtool --install-completion fish
xraylabtool --install-completion powershell

# Test installation
xraylabtool --install-completion --test
```

#### Subcommand Syntax (Full Feature Support)
```bash
# Install for current shell (auto-detected)
xraylabtool install-completion

# Install for specific shell
xraylabtool install-completion bash
xraylabtool install-completion zsh
xraylabtool install-completion fish
xraylabtool install-completion powershell

# Install system-wide (requires sudo)
xraylabtool install-completion --system

# Advanced options
xraylabtool install-completion bash --system --test
xraylabtool install-completion --uninstall

# Test if completion is working
xraylabtool install-completion --test

# Uninstall completion
xraylabtool install-completion --uninstall
```

### Using the Standalone Installer

```bash
# For current user (recommended)
python install_completion.py install

# System-wide installation (requires sudo)
python install_completion.py install --system

# Test installation
python install_completion.py test

# Uninstall
python install_completion.py uninstall
```

## Manual Installation

### For Bash

If you prefer manual installation:

1. Copy the completion script to your bash completion directory:
   ```bash
   mkdir -p ~/.bash_completion.d
   cp _xraylabtool_completion.bash ~/.bash_completion.d/xraylabtool
   ```

2. Add to your `.bashrc`:
   ```bash
   echo "source ~/.bash_completion.d/xraylabtool" >> ~/.bashrc
   ```

3. Reload your shell:
   ```bash
   source ~/.bashrc
   ```

### For Zsh

The installer automatically configures Zsh, but for manual installation:

1. Copy the completion script:
   ```bash
   mkdir -p ~/.bash_completion.d
   cp _xraylabtool_completion.bash ~/.bash_completion.d/xraylabtool
   ```

2. Add to your `.zshrc`:
   ```bash
   cat >> ~/.zshrc << 'EOF'
   # XRayLabTool completion
   # Enable bash completion compatibility in Zsh
   autoload -U +X compinit && compinit
   autoload -U +X bashcompinit && bashcompinit
   source ~/.bash_completion.d/xraylabtool
   EOF
   ```

3. Reload your shell:
   ```bash
   source ~/.zshrc
   ```

### For Fish

The installer automatically configures Fish, but for manual installation:

1. Copy the completion script to Fish completions directory:
   ```fish
   mkdir -p ~/.config/fish/completions
   cp xraylabtool_completion.fish ~/.config/fish/completions/xraylabtool.fish
   ```

2. Reload Fish completions (or restart Fish):
   ```fish
   # Fish will automatically load completions from ~/.config/fish/completions/
   # Just restart Fish or open a new Fish session
   ```

## Usage Examples

After installation, you can use tab completion with xraylabtool:

### Basic Command Completion
```bash
xraylabtool <TAB>         # Shows: calc batch convert formula atomic bragg list
xraylabtool c<TAB>        # Completes to: calc or convert
```

### Calc Command Completion
```bash
xraylabtool calc <TAB>           # Shows common formulas: SiO2 Si Al2O3 Fe2O3...
xraylabtool calc SiO2 <TAB>      # Shows options: --energy --density --output...
xraylabtool calc SiO2 -e <TAB>   # Shows energy examples: 10.0 8.048 5.0,10.0,15.0...
xraylabtool calc SiO2 -e 10.0 -d <TAB>  # Shows density examples: 2.2 2.33 3.95...
```

### File Completion
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2 -o <TAB>    # Shows available files
xraylabtool batch <TAB>                           # Shows *.csv files
```

### Format and Field Completion
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2 --format <TAB>    # Shows: table csv json
xraylabtool calc SiO2 -e 10.0 -d 2.2 --fields <TAB>    # Shows field combinations
```

### Convert Command Completion
```bash
xraylabtool convert <TAB>               # Shows: energy wavelength
xraylabtool convert energy 10.0 --to <TAB>  # Shows: wavelength
```

### Atomic Command Completion
```bash
xraylabtool atomic <TAB>        # Shows common elements: H He Li Be B C N O...
xraylabtool atomic Si<TAB>      # Shows elements starting with Si
```

## Supported Commands

The completion system supports all XRayLabTool commands:

- **calc**: Single material X-ray property calculations
- **batch**: Batch processing from CSV files
- **convert**: Energy/wavelength unit conversions
- **formula**: Chemical formula parsing and analysis
- **atomic**: Atomic data lookup
- **bragg**: Bragg diffraction angle calculations
- **list**: Reference information (constants, fields, examples)

## Completion Features by Command

### calc
- Chemical formulas (SiO2, Al2O3, Fe2O3, etc.)
- Common energy values and patterns
- Material density suggestions
- Output format options (table, csv, json)
- Field selection helpers
- File completion for output

### batch
- CSV file completion for input
- Output format options
- Worker count suggestions
- Field selection helpers

### convert
- Unit type completion (energy, wavelength)
- Value suggestions based on unit type
- File completion for output

### formula
- Common chemical formulas
- Complex molecular formulas
- Output file completion

### atomic
- All chemical elements
- Common element combinations
- Output format options

### bragg
- d-spacing value suggestions
- Energy and wavelength value suggestions
- Diffraction order options
- Output file completion

### list
- Information types (constants, fields, examples)

## Troubleshooting

### Completion Not Working
1. Make sure bash completion is installed on your system:
   ```bash
   # Ubuntu/Debian
   sudo apt install bash-completion

   # macOS with Homebrew
   brew install bash-completion
   ```

2. Check if xraylabtool is in your PATH:
   ```bash
   which xraylabtool
   ```

3. Test the completion installation:
   ```bash
   python install_completion.py test
   ```

4. Reload your shell configuration:
   ```bash
   source ~/.bashrc
   ```

### Permission Issues
- For system-wide installation, make sure you have sudo privileges
- For user installation, ensure `~/.bash_completion.d` is writable

### Completion Not Loading
- Check that the completion script is sourced in your shell configuration
- Verify the completion function is registered:
  ```bash
  complete -p xraylabtool
  ```

## Technical Details

### Files
- `_xraylabtool_completion.bash`: The main bash completion script
- `install_completion.py`: Automated installer with detection and setup
- `SHELL_COMPLETION.md`: This documentation

### Architecture
- Modular completion functions for each command
- Context-aware parameter completion
- Smart file type detection
- Chemical formula pattern matching
- Common value suggestion database

### Customization
You can customize the completion by editing `_xraylabtool_completion.bash`:
- Add more chemical formulas to the suggestion lists
- Modify energy/density value suggestions
- Add custom completion logic for specific workflows

## Contributing

To improve the shell completion:
1. Test with your workflow and identify missing completions
2. Add suggestions for common values in your domain
3. Extend completion logic for new command options
4. Test thoroughly across different shell environments

The completion system is designed to be lightweight and fast, with no external dependencies beyond bash and the xraylabtool CLI itself.
