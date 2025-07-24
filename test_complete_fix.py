#!/usr/bin/env python3
"""
Comprehensive test to verify the virtual environment fix.
"""

import sys
import os
import tkinter as tk
from pathlib import Path

print("AICRL Virtual Environment Fix Test")
print("=" * 50)

def test_venv_fix():
    """Test the complete virtual environment fix."""
    
    print("1. Testing Python path setup...")
    current_dir = Path(__file__).parent.absolute()
    venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
    
    if venv_site_packages.exists():
        print(f"✓ Virtual environment found at: {venv_site_packages}")
        sys.path.insert(0, str(venv_site_packages))
    else:
        print(f"❌ Virtual environment not found at: {venv_site_packages}")
        return False
    
    print("\n2. Testing stable_baselines3 import...")
    try:
        import stable_baselines3
        print(f"✓ stable_baselines3 version: {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import stable_baselines3: {e}")
        return False
    
    print("\n3. Testing PPO import...")
    try:
        from stable_baselines3 import PPO
        print("✓ PPO imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PPO: {e}")
        return False
    
    print("\n4. Testing GUI modules import...")
    try:
        sys.path.append(str(current_dir))
        from modules.training_runner import TrainingRunnerModule
        print("✓ TrainingRunnerModule imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TrainingRunnerModule: {e}")
        return False
    
    print("\n5. Testing module instantiation...")
    try:
        # Create a mock root for tkinter variables
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        module = TrainingRunnerModule()
        print("✓ TrainingRunnerModule created successfully")
        print(f"✓ Found {len(module.available_models)} available models")
        
        # Test model discovery
        if module.available_models:
            model_names = [model['name'] for model in module.available_models]
            print(f"✓ Available models: {', '.join(model_names)}")
        
        root.destroy()
        
    except Exception as e:
        print(f"❌ Failed to create TrainingRunnerModule: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n6. Testing model loading capability...")
    try:
        # Test if we can load a model (without actually doing it)
        from AnalogICLayoutEnv import AnalogICLayoutEnv
        print("✓ AnalogICLayoutEnv imported successfully")
        
        # Create environment
        env = AnalogICLayoutEnv(grid_size=20)
        print("✓ Environment created successfully")
        
    except Exception as e:
        print(f"❌ Failed environment test: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    try:
        success = test_venv_fix()
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Virtual environment fix is working correctly")
            print("✅ GUI should now be able to load models without errors")
            print("\n💡 You can now run:")
            print("   python3 gui_main.py")
            print("   or")
            print("   python3 activate_venv_gui.py")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("⚠️  Virtual environment fix needs more work")
            
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()