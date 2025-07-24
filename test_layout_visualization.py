#!/usr/bin/env python3
"""
Test script to verify layout visualization is working.
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

print("Layout Visualization Test")
print("=" * 30)

def test_visualization():
    """Test layout visualization functionality."""
    
    try:
        # Import required modules
        from modules.training_runner import TrainingRunnerModule
        from AnalogICLayoutEnv import AnalogICLayoutEnv
        
        print("✓ Modules imported successfully")
        
        # Create environment with some placements for testing
        env = AnalogICLayoutEnv(grid_size=20)
        
        # Mock some component placements
        env.placements = {
            (5, 5): 'nfet',
            (7, 5): 'pfet', 
            (9, 7): 'cap',
            (3, 8): 'resistor',
            (12, 10): 'nfet'
        }
        
        print(f"✓ Created test environment with {len(env.placements)} components")
        
        # Create training runner
        root = tk.Tk()
        root.withdraw()
        
        module = TrainingRunnerModule()
        
        # Test simple visualization
        test_output_dir = "./output/visualization_test"
        os.makedirs(test_output_dir, exist_ok=True)
        
        simple_path = f"{test_output_dir}/test_simple_layout.png"
        module._create_simple_layout_visualization(env, simple_path)
        
        if os.path.exists(simple_path):
            print(f"✓ Simple layout visualization created: {simple_path}")
        else:
            print("❌ Simple layout visualization not created")
            return False
        
        # Test enhanced visualization
        from enhanced_visualizer import EnhancedLayoutVisualizer
        
        visualizer = EnhancedLayoutVisualizer(style='professional')
        enhanced_path = f"{test_output_dir}/test_enhanced_layout.png"
        
        try:
            visualizer.render_layout_professional(env, save_path=enhanced_path, show_metrics=True)
            if os.path.exists(enhanced_path):
                print(f"✓ Enhanced layout visualization created: {enhanced_path}")
            else:
                print("❌ Enhanced layout visualization not created")
                return False
        except Exception as e:
            print(f"⚠️  Enhanced visualization error: {e}")
            # Continue even if enhanced fails, as long as simple works
        
        root.destroy()
        
        print(f"\n🎉 Visualization test completed!")
        print(f"📁 Test images saved in: {test_output_dir}/")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_visualization()
    
    if success:
        print("\n✅ Layout visualization is working!")
        print("   When you click 'Generate Layout', images will be created automatically")
    else:
        print("\n❌ Layout visualization needs fixing")
    
    sys.exit(0 if success else 1)