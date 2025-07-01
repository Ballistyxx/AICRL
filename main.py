from AnalogICLayoutEnv import AnalogICLayoutEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from layout_analyzer import LayoutAnalyzer

env = AnalogICLayoutEnv(grid_size=20)
# check_env(env)

# Instantiate the agent
model = PPO(
    "MultiInputPolicy", 
    env, 
    verbose=1,
    tensorboard_log="./ppo_tensorboard/",
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01
)

# Train the agent with more timesteps and better hyperparameters
model.learn(total_timesteps=100000)

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

# Render the final layout
print("Rendering final layout...")
env.render(mode='human')

