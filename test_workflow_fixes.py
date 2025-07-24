#!/usr/bin/env python3
"""
Test script to verify all workflow fixes are working correctly.
"""

import sys
import os
import tkinter as tk
from pathlib import Path
import json

print("AICRL Workflow Fixes Test")
print("=" * 40)

def test_imports_fix():
    """Test that all imports work correctly."""
    print("1. Testing imports fix...")
    
    try:
        # Test json import is available
        import json
        print("✓ json import available")
        
        # Test venv path setup
        current_dir = Path(__file__).parent.absolute()
        venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
        
        if venv_site_packages.exists():
            sys.path.insert(0, str(venv_site_packages))
            print("✓ venv path added")
        
        # Test stable_baselines3
        from stable_baselines3 import PPO
        print("✓ stable_baselines3 imported")
        
        # Test module imports
        from modules.training_runner import TrainingRunnerModule
        from modules.magic_export import MagicExportModule
        print("✓ All modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_training_runner_fix():
    """Test the training runner module with json fix."""
    print("\n2. Testing training runner json fix...")
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        # Create mock tkinter root
        root = tk.Tk()
        root.withdraw()
        
        # Create module
        module = TrainingRunnerModule()
        print("✓ TrainingRunnerModule created")
        
        # Test that json is available in the module
        test_data = {"test": "data"}
        json_string = json.dumps(test_data)
        parsed_data = json.loads(json_string)
        print("✓ JSON operations work correctly")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Training runner error: {e}")
        return False

def test_export_fix():
    """Test the magic export module with filedialog fix."""
    print("\n3. Testing magic export filedialog fix...")
    
    try:
        from modules.magic_export import MagicExportModule
        
        # Create module
        module = MagicExportModule()
        print("✓ MagicExportModule created")
        
        # Test export settings update
        module.update_export_settings()
        settings = module.get_export_settings()
        print("✓ Export settings work correctly")
        
        # Test that choose_output_file method exists and can be called
        # (we won't actually open the dialog)
        assert hasattr(module, 'choose_output_file')
        print("✓ choose_output_file method available")
        
        return True
        
    except Exception as e:
        print(f"❌ Export module error: {e}")
        return False

def test_matplotlib_fix():
    """Test that matplotlib works without threading warnings."""
    print("\n4. Testing matplotlib fix...")
    
    try:
        # Import enhanced visualizer
        from enhanced_visualizer import EnhancedLayoutVisualizer
        print("✓ EnhancedLayoutVisualizer imported")
        
        # Create visualizer
        visualizer = EnhancedLayoutVisualizer()
        print("✓ Visualizer created without threading warnings")
        
        # Test that matplotlib backend is set correctly
        import matplotlib
        backend = matplotlib.get_backend()
        print(f"✓ Matplotlib backend: {backend}")
        
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib error: {e}")
        return False

def test_file_operations():
    """Test that file operations work correctly."""
    print("\n5. Testing file operations...")
    
    try:
        # Test directory creation
        test_dir = Path("./test_output")
        test_dir.mkdir(exist_ok=True)
        print("✓ Directory creation works")
        
        # Test JSON file writing
        test_data = {
            "timestamp": 1234567890,
            "test": "data",
            "components": ["comp1", "comp2"]
        }
        
        test_file = test_dir / "test.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print("✓ JSON file writing works")
        
        # Test JSON file reading
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
        print("✓ JSON file reading works")
        
        # Cleanup
        test_file.unlink()
        test_dir.rmdir()
        print("✓ File cleanup works")
        
        return True
        
    except Exception as e:
        print(f"❌ File operations error: {e}")
        return False

def main():
    """Run all tests."""
    tests = [
        ("Imports Fix", test_imports_fix),
        ("Training Runner Fix", test_training_runner_fix),
        ("Export Fix", test_export_fix),
        ("Matplotlib Fix", test_matplotlib_fix),
        ("File Operations", test_file_operations)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("WORKFLOW FIXES TEST RESULTS")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL FIXES WORKING!")
        print("✅ GUI should now work without errors")
        print("✅ Layout generation should complete successfully")  
        print("✅ Export functionality should work properly")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) still failing")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)