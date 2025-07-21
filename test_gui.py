#!/usr/bin/env python3
"""
Test script for the AICRL GUI application.
Tests import functionality and basic GUI creation without actually launching the GUI.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing module imports...")
    
    try:
        # Test main GUI import
        print("  Importing main GUI...")
        from gui_main import AICRLMainGUI
        print("  ✓ Main GUI imported successfully")
        
        # Test module imports
        print("  Importing modules...")
        from modules.schematic_import import SchematicImportModule
        from modules.training_runner import TrainingRunnerModule
        from modules.magic_export import MagicExportModule
        print("  ✓ All modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        return False

def test_module_creation():
    """Test that modules can be created without errors."""
    print("\nTesting module creation...")
    
    try:
        from modules.schematic_import import SchematicImportModule
        from modules.training_runner import TrainingRunnerModule
        from modules.magic_export import MagicExportModule
        
        # Create module instances
        print("  Creating schematic import module...")
        schematic_module = SchematicImportModule()
        print("  ✓ Schematic module created")
        
        print("  Creating training runner module...")
        training_module = TrainingRunnerModule()
        print("  ✓ Training module created")
        
        print("  Creating magic export module...")
        export_module = MagicExportModule()
        print("  ✓ Export module created")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Module creation error: {e}")
        return False

def test_gui_creation():
    """Test GUI creation with tkinter (but don't show it)."""
    print("\nTesting GUI creation...")
    
    try:
        import tkinter as tk
        from gui_main import AICRLMainGUI
        
        # Create a root window (but don't show it)
        print("  Creating tkinter root...")
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        print("  Creating main GUI application...")
        app = AICRLMainGUI(root)
        print("  ✓ GUI created successfully")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"  ✗ GUI creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("AICRL GUI Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_module_creation,
        test_gui_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"Test failed: {test.__name__}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! GUI is ready to use.")
        print("\nTo launch the GUI, run:")
        print("  python3 gui_main.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)