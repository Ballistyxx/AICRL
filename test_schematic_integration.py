#!/usr/bin/env python3
"""
Test script to verify complete schematic-to-layout integration.
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

print("AICRL Schematic Integration Test")
print("=" * 40)

def create_test_spice_file():
    """Create a test SPICE file for testing."""
    test_spice_content = """
* Test SPICE file for integration testing
.subckt test_circuit vdd vss in out
XM1 out in vdd vdd sky130_fd_pr__pfet_01v8 w=2u l=0.15u
XM2 out in vss vss sky130_fd_pr__nfet_01v8 w=1u l=0.15u  
XC1 out vss sky130_fd_pr__cap_mim_m3_1 c=1p
XR1 in vdd sky130_fd_pr__res_high_po_0p35 r=10k
.ends test_circuit
.end
"""
    test_file = current_dir / "test_schematic_integration.sp"
    with open(test_file, 'w') as f:
        f.write(test_spice_content)
    
    return str(test_file)

def test_schematic_parsing():
    """Test SPICE file parsing."""
    print("1. Testing SPICE file parsing...")
    
    try:
        from modules.schematic_import import SchematicImportModule
        
        # Create test SPICE file
        test_file = create_test_spice_file()
        print(f"✓ Created test SPICE file: {test_file}")
        
        # Create schematic import module
        schematic_module = SchematicImportModule()
        schematic_module.current_file = test_file
        
        # Parse the file
        schematic_data = schematic_module.parse_schematic_file()
        
        if schematic_data:
            print("✓ SPICE file parsed successfully")
            components = schematic_data.get("components", [])
            nets = schematic_data.get("nets", [])
            print(f"✓ Found {len(components)} components and {len(nets)} nets")
            
            # Test RL-compatible conversion
            rl_data = schematic_module.get_rl_compatible_data()
            if rl_data:
                print("✓ RL-compatible data generated successfully")
                stats = rl_data.get("statistics", {})
                print(f"✓ RL format: {stats.get('total_components', 0)} components, " +
                     f"{stats.get('total_connections', 0)} connections")
                return rl_data
            else:
                print("❌ RL-compatible conversion failed")
                return None
        else:
            print("❌ SPICE parsing failed")
            return None
            
    except Exception as e:
        print(f"❌ Schematic parsing error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment_integration(rl_data):
    """Test environment integration with schematic data."""
    print("\n2. Testing environment integration...")
    
    try:
        from AnalogICLayoutEnv import AnalogICLayoutEnv
        
        # Create environment with schematic data
        env = AnalogICLayoutEnv(grid_size=20, schematic_data=rl_data)
        print("✓ Environment created with schematic data")
        
        # Check components
        print(f"✓ Environment has {len(env.components)} components")
        print(f"✓ Environment has {len(env.connections)} connections")
        
        # Test component properties
        for i, comp in enumerate(env.components):
            if i < 3:  # Show first 3 components
                print(f"   Component {comp['id']}: {comp['name']} ({comp['type']}) - " +
                     f"{comp['width']}x{comp['height']}")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        print(f"✓ Observation shape: {obs['observation'].shape}")
        
        return env
        
    except Exception as e:
        print(f"❌ Environment integration error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_loading_with_schematic(env):
    """Test model loading with schematic-configured environment."""
    print("\n3. Testing model loading with schematic environment...")
    
    try:
        from stable_baselines3 import PPO
        
        # Try to load existing model
        model_path = current_dir / "models" / "best_model.zip"
        if model_path.exists():
            print(f"✓ Found model: {model_path}")
            
            # Load model with schematic environment
            model = PPO.load(str(model_path), env=env)
            print("✓ Model loaded successfully with schematic environment")
            
            # Test prediction
            obs, info = env.reset()
            action, _states = model.predict(obs)
            print("✓ Model prediction successful")
            
            return model
        else:
            print(f"⚠️  Model not found at {model_path}")
            print("   Skipping model loading test")
            return None
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        import traceback 
        traceback.print_exc()
        return None

def test_layout_generation(env, model):
    """Test layout generation with schematic constraints."""
    print("\n4. Testing layout generation with constraints...")
    
    if not model:
        print("⚠️  No model available, skipping generation test")
        return False
        
    try:
        # Reset environment
        obs, info = env.reset()
        print("✓ Environment reset for generation")
        
        # Generate a few steps
        steps = 0
        max_steps = 10
        
        while steps < max_steps:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = env.step(action.item())
            steps += 1
            
            if len(env.placements) > 0:
                break
                
        print(f"✓ Generated {steps} steps")
        print(f"✓ Placed {len(env.placements)} components")
        
        # Show placements
        for pos, comp_type in env.placements.items():
            print(f"   Component at {pos}: {comp_type}")
            
        return True
        
    except Exception as e:
        print(f"❌ Layout generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_integration():
    """Test GUI integration with schematic data."""
    print("\n5. Testing GUI integration...")
    
    try:
        from modules.training_runner import TrainingRunnerModule
        
        # Create mock tkinter environment
        root = tk.Tk()
        root.withdraw()
        
        # Create training module
        training_module = TrainingRunnerModule()
        print("✓ Training module created")
        
        # Create and set test schematic data
        test_rl_data = {
            "format": "rl_compatible",
            "components": [
                {"id": 1, "name": "M1", "type": "pfet", "width": 2, "height": 2, "color": "blue", "can_overlap": False},
                {"id": 2, "name": "M2", "type": "nfet", "width": 2, "height": 2, "color": "red", "can_overlap": False}
            ],
            "connections": [(1, 2)],
            "statistics": {
                "total_components": 2,
                "total_nets": 3,
                "total_connections": 1,
                "component_types": ["pfet", "nfet"]
            }
        }
        
        # Set schematic data
        training_module.set_schematic_data(test_rl_data)
        print("✓ Schematic data set in training module")
        
        root.destroy()
        return True
        
    except Exception as e:
        print(f"❌ GUI integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files."""
    test_file = current_dir / "test_schematic_integration.sp"
    if test_file.exists():
        test_file.unlink()
        print("✓ Cleaned up test files")

def main():
    """Run all integration tests."""
    tests = [
        ("SPICE Parsing & RL Conversion", test_schematic_parsing),
        ("Environment Integration", None),  # Will be called with rl_data
        ("Model Loading", None),  # Will be called with env
        ("Layout Generation", None),  # Will be called with env, model
        ("GUI Integration", test_gui_integration)
    ]
    
    print("Running AICRL Schematic Integration Tests...")
    print("=" * 50)
    
    # Run tests in sequence
    rl_data = None
    env = None
    model = None
    results = []
    
    try:
        # Test 1: Schematic parsing
        rl_data = test_schematic_parsing()
        results.append(("SPICE Parsing & RL Conversion", rl_data is not None))
        
        if rl_data:
            # Test 2: Environment integration
            env = test_environment_integration(rl_data)
            results.append(("Environment Integration", env is not None))
            
            if env:
                # Test 3: Model loading
                model = test_model_loading_with_schematic(env)
                results.append(("Model Loading", model is not None))
                
                # Test 4: Layout generation
                generation_result = test_layout_generation(env, model)
                results.append(("Layout Generation", generation_result))
        
        # Test 5: GUI integration
        gui_result = test_gui_integration()
        results.append(("GUI Integration", gui_result))
        
    finally:
        cleanup_test_files()
    
    # Results summary
    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 FULL SCHEMATIC INTEGRATION WORKING!")
        print("✅ SPICE files can be parsed and converted")
        print("✅ RL environment accepts schematic constraints")
        print("✅ Models can be loaded with custom environments")
        print("✅ Layout generation uses schematic data")
        print("✅ GUI integration is complete")
        print("\n🚀 Ready for production use!")
    else:
        print(f"\n⚠️  {len(results) - passed} integration issue(s) found")
        print("   Check the errors above for details")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)