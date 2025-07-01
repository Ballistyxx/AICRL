from AnalogICLayoutEnv import AnalogICLayoutEnv
from layout_analyzer import LayoutAnalyzer
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import os

def get_unwrapped_env(env):
    """Get the underlying environment from wrapped environments."""
    unwrapped_env = env
    while hasattr(unwrapped_env, 'env'):
        unwrapped_env = unwrapped_env.env
    return unwrapped_env

def train_simple_model():
    """Train a simple IC layout model without complex callbacks."""
    
    # Create environment with monitoring
    env = AnalogICLayoutEnv(grid_size=20)
    env = Monitor(env, "./logs/")
    
    # Create directories
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./logs/", exist_ok=True)
    
    # Create model with good hyperparameters
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        learning_rate=0.0005,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        max_grad_norm=0.5,
    )
    
    print("Starting simple training...")
    model.learn(total_timesteps=100000, progress_bar=True)
    
    return model, env

def evaluate_model(model, env, num_episodes=3):
    """Evaluate model over multiple episodes."""
    print(f"\n{'='*50}")
    print("EVALUATING MODEL PERFORMANCE")
    print(f"{'='*50}")
    
    unwrapped_env = get_unwrapped_env(env)
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < 50:  # Safety limit
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            
            # Decode the action for logging
            grid_size = unwrapped_env.grid_size
            comp_idx = action.item() // (grid_size * grid_size)
            comp_name = unwrapped_env.components[comp_idx]['name'] if comp_idx < len(unwrapped_env.components) else "Invalid"
            print(f"  Step {step}: Placing {comp_name}")
            
            obs, reward, done, truncated, info = env.step(action.item())
            
            if reward < 0:
                print(f"    -> Penalty: {reward:.2f}")
            else:
                print(f"    -> Reward: {reward:.2f}")
        
        # Analyze this episode
        # Use unwrapped env for analysis
        unwrapped_env = env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
            
        analyzer = LayoutAnalyzer(unwrapped_env)
        metrics = analyzer.analyze_layout()
        
        if episode == 0:  # Show visual for first episode
            unwrapped_env.render(mode='human')
    
    return metrics

if __name__ == "__main__":
    # Train the model
    model, env = train_simple_model()
    
    # Save the model
    model.save("./models/simple_ic_layout_model")
    print("Model saved to ./models/simple_ic_layout_model")
    
    # Evaluate
    metrics = evaluate_model(model, env)
    
    print("\nTraining and evaluation complete!")
    print("To view training progress, run: tensorboard --logdir ./ppo_tensorboard/")
