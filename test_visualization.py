#!/usr/bin/env python3
"""Test script to verify the enhanced visualization works."""

from AnalogICLayoutEnv import AnalogICLayoutEnv
from enhanced_visualizer import EnhancedLayoutVisualizer
import numpy as np

def test_visualization():
    """Test the enhanced visualization with a simple layout."""
    print("Creating environment...")
    env = AnalogICLayoutEnv(grid_size=10)
    
    print("Manually placing some components for testing...")
    # Manually place some components to test visualization
    obs, info = env.reset()
    
    # Place first component (nfet_d1) at position (2, 2)
    action = 0 * (env.grid_size * env.grid_size) + 2 * env.grid_size + 2
    obs, reward, done, truncated, info = env.step(action)
    print(f"Placed component 1, reward: {reward}")
    
    # Place second component (nfet_d2) at position (2, 6) for symmetry
    action = 1 * (env.grid_size * env.grid_size) + 2 * env.grid_size + 6
    obs, reward, done, truncated, info = env.step(action)
    print(f"Placed component 2, reward: {reward}")
    
    # Place third component (pfet_m1) at position (5, 2)
    action = 2 * (env.grid_size * env.grid_size) + 5 * env.grid_size + 2
    obs, reward, done, truncated, info = env.step(action)
    print(f"Placed component 3, reward: {reward}")
    
    print("Creating enhanced visualizer...")
    visualizer = EnhancedLayoutVisualizer(style='professional')
    
    print("Generating professional layout visualization...")
    try:
        visualizer.render_layout_professional(env, save_path="./output/test_layout.png", show_metrics=True)
        print("✓ Professional layout visualization created successfully!")
    except Exception as e:
        print(f"✗ Error creating professional layout: {e}")
        import traceback
        traceback.print_exc()
    
    print("Generating mock training progress...")
    # Create some mock training data for testing
    visualizer.training_history = []
    for i in range(100):
        reward = -100 + 0.5 * i + np.random.normal(0, 5)  # Improving trend with noise
        visualizer.training_history.append({
            'step': i * 1000,
            'reward': reward,
            'loss': max(0.1, 10 - 0.05 * i + np.random.normal(0, 0.5))
        })
    
    try:
        visualizer.plot_training_progress(save_path="./output/test_training_progress.png")
        print("✓ Training progress visualization created successfully!")
    except Exception as e:
        print(f"✗ Error creating training progress: {e}")
        import traceback
        traceback.print_exc()
    
    print("Generating complete training report...")
    try:
        visualizer.create_training_report(env, save_dir="./reports")
        print("✓ Complete training report created successfully!")
    except Exception as e:
        print(f"✗ Error creating training report: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed! Check the ./output/ and ./reports/ directories for generated files.")

if __name__ == "__main__":
    test_visualization()