#!/usr/bin/env python3
"""
Test script to verify XRayLabTool installation and basic functionality.

This script can be used to test:
1. Local installation from built wheel
2. Installation from Test PyPI
3. Installation from PyPI
4. Basic functionality verification

Usage:
    python test_installation.py [--source local|testpypi|pypi]
"""

import sys
import subprocess
import tempfile
import argparse
from pathlib import Path

def run_in_clean_env(commands, description):
    """Run commands in a clean virtual environment."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_venv"
        
        # Create virtual environment
        print("Creating test virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        
        # Get python executable in venv
        if sys.platform == "win32":
            python_exe = venv_path / "Scripts" / "python.exe"
            pip_exe = venv_path / "Scripts" / "pip.exe"
        else:
            python_exe = venv_path / "bin" / "python"
            pip_exe = venv_path / "bin" / "pip"
        
        try:
            # Run each command in the virtual environment
            for cmd in commands:
                print(f"\nRunning: {' '.join(cmd)}")
                if cmd[0] == "python":
                    cmd[0] = str(python_exe)
                elif cmd[0] == "pip":
                    cmd[0] = str(pip_exe)
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if result.stdout:
                    print(result.stdout)
            
            print(f"✅ {description} - SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - FAILED")
            print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False

def test_basic_functionality():
    """Test basic package functionality."""
    test_code = """
import xraylabtool as xlt
import numpy as np

print("🧪 Testing basic functionality...")

# Test SubRefrac
print("Testing SubRefrac...")
result = xlt.SubRefrac("SiO2", 10.0, 2.2)
print(f"Formula: {result.Formula}")
print(f"MW: {result.MW:.2f} g/mol")
print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")

# Test Refrac
print("\\nTesting Refrac...")
results = xlt.Refrac(["SiO2", "Al2O3"], [8.0, 10.0], [2.2, 3.95])
print(f"Number of materials: {len(results)}")
for material, data in results.items():
    print(f"{material}: MW = {data.MW:.2f} g/mol")

# Test utility functions
print("\\nTesting utility functions...")
energy = 10.0  # keV
wavelength = xlt.energy_to_wavelength(energy)
energy_back = xlt.wavelength_to_energy(wavelength)
print(f"Energy: {energy} keV -> Wavelength: {wavelength:.3f} Å -> Energy: {energy_back:.3f} keV")

# Test formula parsing
print("\\nTesting formula parsing...")
elements, counts = xlt.parse_formula("Al2O3")
print(f"Al2O3 -> Elements: {elements}, Counts: {counts}")

# Test atomic data
print("\\nTesting atomic data...")
atomic_num = xlt.get_atomic_number("Si")
atomic_weight = xlt.get_atomic_weight("Si")
print(f"Silicon: Z = {atomic_num}, MW = {atomic_weight:.3f}")

print("\\n✅ All basic functionality tests passed!")
"""
    return test_code

def test_local_installation():
    """Test installation from local wheel file."""
    # Find the wheel file
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory found. Run build_package.py first.")
        return False
    
    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("❌ No wheel file found in dist/. Run build_package.py first.")
        return False
    
    wheel_file = wheel_files[0]  # Use the first wheel file found
    
    commands = [
        ["pip", "install", str(wheel_file.absolute())],
        ["python", "-c", test_basic_functionality()]
    ]
    
    return run_in_clean_env(commands, f"Testing local installation from {wheel_file.name}")

def test_testpypi_installation():
    """Test installation from Test PyPI."""
    commands = [
        ["pip", "install", "--index-url", "https://test.pypi.org/simple/", 
         "--extra-index-url", "https://pypi.org/simple/", "xraylabtool"],
        ["python", "-c", test_basic_functionality()]
    ]
    
    return run_in_clean_env(commands, "Testing installation from Test PyPI")

def test_pypi_installation():
    """Test installation from PyPI."""
    commands = [
        ["pip", "install", "xraylabtool"],
        ["python", "-c", test_basic_functionality()]
    ]
    
    return run_in_clean_env(commands, "Testing installation from PyPI")

def main():
    parser = argparse.ArgumentParser(description="Test XRayLabTool installation")
    parser.add_argument("--source", choices=["local", "testpypi", "pypi"], 
                       default="local", help="Installation source to test")
    
    args = parser.parse_args()
    
    print("🧪 XRayLabTool Installation Tester")
    print("=" * 60)
    
    success = False
    
    if args.source == "local":
        success = test_local_installation()
    elif args.source == "testpypi":
        success = test_testpypi_installation()
    elif args.source == "pypi":
        success = test_pypi_installation()
    
    if success:
        print("\n🎉 Installation test completed successfully!")
        print("✅ Package is working correctly")
    else:
        print("\n❌ Installation test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()