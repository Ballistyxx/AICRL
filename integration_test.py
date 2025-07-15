#!/usr/bin/env python3
"""Integration test to verify the enhanced visualizer works with the main training script."""

import sys
import traceback
from AnalogICLayoutEnv import AnalogICLayoutEnv
from enhanced_visualizer import EnhancedLayoutVisualizer
from stable_baselines3 import PPO
from layout_analyzer import LayoutAnalyzer

def test_integration():
    """Test that the enhanced visualizer integrates properly with the training system."""
    print("=== Enhanced Visualizer Integration Test ===\n")
    
    try:
        # 1. Create environment
        print("1. Creating environment...")
        env = AnalogICLayoutEnv(grid_size=10)
        print("   ✓ Environment created successfully")
        
        # 2. Create and train a simple model
        print("\n2. Creating and training PPO model...")
        model = PPO(
            "MultiInputPolicy", 
            env, 
            verbose=0,
            tensorboard_log="./ppo_tensorboard/",
            learning_rate=0.001,
            n_steps=512,
            batch_size=32
        )
        model.learn(total_timesteps=1000)
        print("   ✓ Model trained successfully")
        
        # 3. Run a quick evaluation
        print("\n3. Running model evaluation...")
        obs, info = env.reset()
        done = False
        step = 0
        MAX_STEPS = 20
        
        while not done and step < MAX_STEPS:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action.item())
            
        print(f"   ✓ Evaluation completed after {step} steps")
        print(f"   ✓ {len(env.placements)} components placed")
        
        # 4. Test enhanced visualizer
        print("\n4. Testing enhanced visualizer...")
        visualizer = EnhancedLayoutVisualizer(style='professional')
        print("   ✓ Visualizer created successfully")
        
        # 5. Test layout visualization
        print("\n5. Testing layout visualization...")
        if env.placements:
            fig = visualizer.render_layout_professional(
                env, 
                save_path="./output/integration_test_layout.png",
                show_metrics=True
            )
            print("   ✓ Professional layout visualization created")
        else:
            print("   ⚠ No components placed, skipping layout visualization")
        
        # 6. Test training progress visualization
        print("\n6. Testing training progress visualization...")
        # Add some mock training data
        visualizer.training_history = []
        for i in range(50):
            visualizer.training_history.append({
                'step': i * 20,
                'reward': -50 + i * 0.5 + (i % 10) * 0.1,
                'loss': max(0.1, 5 - i * 0.1)
            })
        
        fig = visualizer.plot_training_progress(
            tensorboard_log_dir="./ppo_tensorboard/",
            save_path="./output/integration_test_progress.png"
        )
        print("   ✓ Training progress visualization created")
        
        # 7. Test comprehensive report
        print("\n7. Testing comprehensive report generation...")
        visualizer.create_training_report(env, save_dir="./reports/integration_test")
        print("   ✓ Comprehensive report generated")
        
        # 8. Test layout analyzer integration
        print("\n8. Testing layout analyzer integration...")
        analyzer = LayoutAnalyzer(env)
        metrics = analyzer.analyze_layout()
        print(f"   ✓ Layout analysis completed: {len(metrics)} metrics calculated")
        
        print("\n=== Integration Test PASSED ===")
        print("All components work together successfully!")
        print("\nGenerated files:")
        print("- ./output/integration_test_layout.png")
        print("- ./output/integration_test_progress.png")
        print("- ./reports/integration_test/")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
