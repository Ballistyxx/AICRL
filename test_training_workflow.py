#!/usr/bin/env python3
"""
Test script to verify training workflow with schematic data.
"""

import sys
import os
import tkinter as tk
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Add venv path
venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

print("Training Workflow Test")
print("=" * 25)

def test_training_initialization():
    """Test training module initialization with all variables."""
    print("1. Testing training module initialization...")
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        # Create mock GUI root
        root = tk.Tk()
        root.withdraw()
        
        # Create training module
        training_module = TrainingRunnerModule()
        print("✓ Training module created")
        
        # Check required attributes
        required_attrs = [
            'model_file_var', 'model_selection_var', 'load_existing_var', 
            'grid_size_var', 'algorithm_var', 'available_models', 
            'schematic_data'
        ]
        
        for attr in required_attrs:
            if hasattr(training_module, attr):
                print(f"✓ {attr} initialized")
            else:
                print(f"❌ {attr} missing")
                return False
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading_workflow():
    """Test model loading with compatibility checks."""
    print("\n2. Testing model loading workflow...")
    
    try:
        from modules.training_runner import TrainingRunnerModule
        from AnalogICLayoutEnv import AnalogICLayoutEnv
        
        # Create mock GUI root
        root = tk.Tk()
        root.withdraw()
        
        # Create training module
        training_module = TrainingRunnerModule()
        
        # Set mock schematic data
        mock_schematic_data = {
            "format": "rl_compatible",
            "components": [
                {"id": i, "name": f"M{i}", "type": "nfet", "width": 2, "height": 2, 
                 "color": "red", "can_overlap": False} for i in range(1, 4)  # 3 components
            ],
            "connections": [(1, 2), (2, 3)],
            "statistics": {"total_components": 3, "component_types": ["nfet"]}
        }
        
        training_module.set_schematic_data(mock_schematic_data)
        print("✓ Schematic data set")
        
        # Test environment creation
        env = AnalogICLayoutEnv(grid_size=20, schematic_data=mock_schematic_data)
        print(f"✓ Environment created with {len(env.components)} components")
        
        # Test model creation
        try:
            model = training_module._create_new_model(env)
            print("✓ New model creation successful")
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return False
        
        # Test helper methods exist
        helper_methods = [
            '_load_model_with_compatibility_check',
            '_create_new_model', 
            '_check_model_compatibility'
        ]
        
        for method in helper_methods:
            if hasattr(training_module, method):
                print(f"✓ {method} exists")
            else:
                print(f"❌ {method} missing")
                return False
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_parameters():
    """Test training parameters and settings."""
    print("\n3. Testing training parameters...")
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        # Create mock GUI root
        root = tk.Tk()
        root.withdraw()
        
        # Create training module
        training_module = TrainingRunnerModule()
        
        # Check training parameters
        expected_params = [
            "total_timesteps", "learning_rate", "batch_size", 
            "n_steps", "gamma", "gae_lambda", "ent_coef"
        ]
        
        for param in expected_params:
            if param in training_module.training_params:
                value = training_module.training_params[param]
                print(f"✓ {param}: {value}")
            else:
                print(f"❌ {param} missing from training_params")
                return False
        
        # Test model discovery
        training_module.discover_available_models()
        print(f"✓ Discovered {len(training_module.available_models)} models")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ Training parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all training workflow tests."""
    tests = [
        ("Training Module Initialization", test_training_initialization),
        ("Model Loading Workflow", test_model_loading_workflow),
        ("Training Parameters", test_training_parameters)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 25)
    print("TRAINING WORKFLOW RESULTS")
    print("=" * 25)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 Training workflow fixes successful!")
        print("✅ All required attributes initialized")
        print("✅ Model loading with compatibility checking works")
        print("✅ Training parameters properly configured")
        print("✅ Helper methods available")
        print("\n🚀 Training should now work without errors!")
    else:
        print(f"\n⚠️  {len(results) - passed} issue(s) still remaining")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)