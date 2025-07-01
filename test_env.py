#!/usr/bin/env python3
"""
Simple test script to verify that the environment works correctly
before running the full training.
"""

from AnalogICLayoutEnv import AnalogICLayoutEnv
import numpy as np

def test_environment():
    """Test basic environment functionality."""
    print("Testing AnalogICLayoutEnv...")
    
    try:
        # Create environment
        env = AnalogICLayoutEnv(grid_size=10)
        print(f"✓ Environment created successfully")
        print(f"  - Grid size: {env.grid_size}")
        print(f"  - Number of components: {env.num_components}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset successfully")
        print(f"  - Observation keys: {obs.keys()}")
        print(f"  - Grid shape: {obs['observation'].shape}")
        print(f"  - Action mask shape: {obs['action_mask'].shape}")
        
        # Test a few steps
        print("\nTesting environment steps...")
        for i in range(3):
            # Get valid actions
            valid_actions = np.where(obs['action_mask'] == 1)[0]
            if len(valid_actions) == 0:
                print("No valid actions available")
                break
                
            # Take random valid action
            action = np.random.choice(valid_actions)
            obs, reward, done, info = env.step(action)
            
            comp_idx = action // (env.grid_size * env.grid_size)
            comp_name = env.components[comp_idx]['name']
            
            print(f"  Step {i+1}: Placed {comp_name}, reward: {reward:.2f}, done: {done}")
            
            if done:
                print("  Episode completed!")
                break
        
        print("✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("\n🎉 Environment is working correctly!")
        print("You can now run: python3 advanced_training.py")
    else:
        print("\n❌ Environment has issues that need to be fixed.")
