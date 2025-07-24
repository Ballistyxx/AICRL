#!/usr/bin/env python3
"""
Virtual Environment Activation for AICRL GUI.

This script ensures the GUI runs with the proper virtual environment,
giving access to stable_baselines3 and other dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def activate_venv_and_run_gui():
    """Activate the AICRL virtual environment and run the GUI."""
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Path to the AICRL virtual environment
    venv_path = current_dir / "AICRL"
    
    # Check if venv exists
    if not venv_path.exists():
        print("❌ Error: AICRL virtual environment not found!")
        print(f"   Expected at: {venv_path}")
        return False
    
    # Get the Python executable from the venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print("❌ Error: Python executable not found in virtual environment!")
        print(f"   Expected at: {python_exe}")
        return False
    
    # Verify stable_baselines3 is available in venv
    result = subprocess.run([
        str(python_exe), "-c", 
        "import stable_baselines3; print('✓ stable_baselines3 available')"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Error: stable_baselines3 not found in virtual environment!")
        print(f"   Error: {result.stderr}")
        return False
    
    print("✓ Virtual environment validation successful")
    print("✓ stable_baselines3 is available")
    print("🚀 Launching AICRL GUI with proper environment...")
    
    # Run the GUI with the venv Python
    gui_script = current_dir / "gui_main.py"
    if not gui_script.exists():
        print("❌ Error: GUI script not found!")
        print(f"   Expected at: {gui_script}")
        return False
    
    # Launch the GUI
    try:
        subprocess.run([str(python_exe), str(gui_script)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running GUI: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 GUI terminated by user")
        return True

if __name__ == "__main__":
    print("AICRL GUI Launcher with Virtual Environment")
    print("=" * 50)
    
    success = activate_venv_and_run_gui()
    
    if not success:
        print("\n💡 To fix this issue:")
        print("   1. Make sure the AICRL virtual environment is properly set up")
        print("   2. Install stable_baselines3 in the venv: pip install stable-baselines3")
        print("   3. Run this script again")
        sys.exit(1)
    
    print("\n✅ GUI session completed successfully")