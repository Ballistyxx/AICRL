from AnalogICLayoutEnv import AnalogICLayoutEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from layout_analyzer import LayoutAnalyzer
from enhanced_visualizer import EnhancedLayoutVisualizer
import os
import glob

def get_next_ppo_number(tensorboard_log_dir):
    """Get the next PPO number that will be used by the model."""
    # Find all existing PPO directories
    ppo_dirs = glob.glob(os.path.join(tensorboard_log_dir, "PPO_*"))
    if not ppo_dirs:
        return 1
    
    # Extract numbers and find the maximum
    numbers = []
    for ppo_dir in ppo_dirs:
        try:
            num = int(os.path.basename(ppo_dir).split('_')[1])
            numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    return max(numbers) + 1 if numbers else 1

# Create output directories if they don't exist
os.makedirs("./output", exist_ok=True)
os.makedirs("./reports", exist_ok=True)

# Determine the PPO number that will be used
tensorboard_log_dir = "./ppo_tensorboard/"
ppo_number = get_next_ppo_number(tensorboard_log_dir)
ppo_reports_dir = f"./reports/PPO_{ppo_number}"
os.makedirs(ppo_reports_dir, exist_ok=True)

print(f"Training PPO model #{ppo_number}")
print(f"Reports will be saved to: {ppo_reports_dir}")

env = AnalogICLayoutEnv(grid_size=20)
# check_env(env)

# Instantiate the agent
model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1,
    tensorboard_log=tensorboard_log_dir,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01
)

# Train the agent with more timesteps and better hyperparameters
model.learn(total_timesteps=2000000)

# Add timeout to prevent infinite loops
print("\n--- Evaluating Trained Model ---")
obs, info = env.reset()
done = False
step = 0
MAX_STEPS = 100  # Safety limit

# The environment will now only terminate when all components are placed.
# Step through the placement process with the trained agent.
while not done and step < MAX_STEPS:
    step += 1
    action, _states = model.predict(obs, deterministic=True)
    
    # Decode the action to get component info for logging
    comp_idx = action.item() // (env.grid_size * env.grid_size)
    comp_name = "Invalid Component"
    if comp_idx < len(env.components):
        comp_name = env.components[comp_idx]['name']

    print(f"Step {step}: Agent tries to place {comp_name}...")
    
    obs, reward, done, truncated, info = env.step(action.item())

    # Access the observation from the dictionary
    if isinstance(obs, dict) and 'observation' in obs:
        # For logging or other purposes, you might want to see the grid
        # grid_observation = obs['observation']
        pass

    print(f"  -> Reward: {reward:.2f}, Done: {done}")
    if info:
        print(f"  -> Info: {info}")

print(f"\nEvaluation finished after {step} steps.")

# Comprehensive analysis
analyzer = LayoutAnalyzer(env)
analyzer.analyze_layout()

# Enhanced visualization with professional styling
print("Creating enhanced visualization...")
visualizer = EnhancedLayoutVisualizer(style='professional')

# Generate professional layout visualization
print("Generating professional layout visualization...")
visualizer.render_layout_professional(env, save_path=f"{ppo_reports_dir}/trained_model_layout.png", show_metrics=True)

# Generate training progress visualization
print("Generating training progress visualization...")
visualizer.plot_training_progress(tensorboard_log_dir=f"{tensorboard_log_dir}/PPO_{ppo_number}", save_path=f"{ppo_reports_dir}/training_progress.png")

# Generate comprehensive training report
print("Generating comprehensive training report...")
visualizer.create_training_report(env, save_dir=ppo_reports_dir)

print(f"\nVisualization complete! Check the {ppo_reports_dir}/ directory for generated files.")

