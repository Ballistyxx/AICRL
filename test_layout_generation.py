#!/usr/bin/env python3
"""
Test script for layout generation functionality using existing trained models.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_discovery():
    """Test model discovery functionality."""
    print("Testing Model Discovery")
    print("-" * 30)
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        # Create training module
        module = TrainingRunnerModule()
        print(f"✓ Training module created")
        print(f"✓ Available models found: {len(module.available_models)}")
        
        if module.available_models:
            print("\nDiscovered Models:")
            for i, model in enumerate(module.available_models):
                size_mb = model["size"] / (1024 * 1024)
                modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(model["modified"]))
                print(f"  {i+1}. {model['name']} ({model['type']})")
                print(f"     Size: {size_mb:.1f} MB, Modified: {modified}")
                print(f"     Status: {model['status']}")
                
        return len(module.available_models) > 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading an existing model."""
    print("\nTesting Model Loading")
    print("-" * 25)
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        module = TrainingRunnerModule()
        
        if not module.available_models:
            print("✗ No models available for testing")
            return False
            
        # Test loading the first available model
        model_info = module.available_models[0]
        print(f"Loading model: {model_info['name']}")
        
        # Simulate GUI selection
        module.model_selection_var = type('MockVar', (), {'get': lambda: model_info['name']})()
        
        # Load the model
        model = module.load_selected_model()
        
        if model:
            print(f"✓ Model loaded successfully: {type(model)}")
            print(f"✓ Model path: {module.loaded_model_path}")
            return True
        else:
            print("✗ Model loading failed")
            return False
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layout_generation():
    """Test layout generation with loaded model."""
    print("\nTesting Layout Generation")
    print("-" * 30)
    
    try:
        from modules.training_runner import TrainingRunnerModule
        import tkinter as tk
        
        # Create a mock root for tkinter variables
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        module = TrainingRunnerModule()
        
        # Set up mock GUI variables
        module.grid_size_var = tk.StringVar(value="20")
        module.model_selection_var = tk.StringVar()
        module.load_existing_var = tk.BooleanVar(value=True)
        
        if not module.available_models:
            print("✗ No models available for testing")
            root.destroy()
            return False
            
        # Select first available model
        model_info = module.available_models[0]
        module.model_selection_var.set(model_info['name'])
        
        print(f"Using model: {model_info['name']}")
        print(f"Grid size: {module.grid_size_var.get()}")
        
        # Load the model
        print("Loading model...")
        model = module.load_selected_model()
        
        if not model:
            print("✗ Failed to load model")
            root.destroy()
            return False
            
        print("✓ Model loaded successfully")
        
        # Create mock log method
        original_add_log = module.add_log_message
        def mock_log(msg):
            print(f"[LOG] {msg}")
        module.add_log_message = mock_log
        
        # Test the generation thread function directly
        print("\nStarting layout generation...")
        module._run_generation_thread()
        
        print("✓ Layout generation test completed")
        
        # Restore original log method
        module.add_log_message = original_add_log
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ Error in layout generation: {e}")
        import traceback
        traceback.print_exc()
        try:
            root.destroy()
        except:
            pass
        return False

def test_schematic_integration():
    """Test integration with schematic data."""
    print("\nTesting Schematic Integration")
    print("-" * 35)
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        module = TrainingRunnerModule()
        
        # Create mock schematic data
        mock_schematic = {
            "format": "spice",
            "file": "test_example.sp",
            "components": [
                {"instance_name": "XC1", "type": "capacitor"},
                {"instance_name": "XM1", "type": "mosfet"},
                {"instance_name": "XR1", "type": "resistor"}
            ],
            "nets": ["net1", "net2", "vdd", "vss"],
            "statistics": {
                "total_components": 3,
                "net_count": 4
            }
        }
        
        # Set schematic data
        module.set_schematic_data(mock_schematic)
        
        if module.schematic_data:
            print(f"✓ Schematic data set successfully")
            print(f"✓ Components: {len(module.schematic_data['components'])}")
            print(f"✓ Format: {module.schematic_data['format']}")
            return True
        else:
            print("✗ Failed to set schematic data")
            return False
            
    except Exception as e:
        print(f"✗ Error in schematic integration: {e}")
        return False

def main():
    """Run all tests."""
    print("Layout Generation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Model Discovery", test_model_discovery),
        ("Model Loading", test_model_loading), 
        ("Layout Generation", test_layout_generation),
        ("Schematic Integration", test_schematic_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("Test Results Summary")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Layout generation is ready to use.")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the logs above.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)