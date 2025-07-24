#!/usr/bin/env python3
"""
Training script for relational placement model.

This script trains a PPO agent using the RelationalLayoutEnv and RelationalPolicy
for improved generalization across different circuit sizes and topologies.

Author: AICRL Team
"""

import sys
import os
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Add venv path
venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from RelationalLayoutEnv import RelationalLayoutEnv
from RelationalPolicy import RelationalActorCriticPolicy


def create_relational_env(grid_size=15, difficulty_level=1, schematic_data=None):
    """Create a relational layout environment."""
    def _init():
        return RelationalLayoutEnv(
            grid_size=grid_size,
            difficulty_level=difficulty_level,
            schematic_data=schematic_data
        )
    return _init


def create_curriculum_envs(grid_sizes=[10, 15, 20], max_difficulty=3):
    """Create curriculum environments with increasing difficulty."""
    envs = []
    
    for difficulty in range(1, max_difficulty + 1):
        for grid_size in grid_sizes:
            env_fn = create_relational_env(
                grid_size=grid_size,
                difficulty_level=difficulty
            )
            envs.append(env_fn)
    
    return envs


def train_relational_model(
    total_timesteps=200000,
    grid_size=15,
    difficulty_level=1,
    schematic_data=None,
    use_curriculum=False,
    save_path="./models/relational_model",
    log_dir="./relational_logs/"
):
    """
    Train a relational placement model.
    
    Args:
        total_timesteps: Total training timesteps
        grid_size: Grid size for environment
        difficulty_level: Difficulty level (1=easy, 2=medium, 3=hard)
        schematic_data: Optional schematic data for training
        use_curriculum: Whether to use curriculum learning
        save_path: Path to save the trained model
        log_dir: Directory for tensorboard logs
    """
    
    print("🚀 Starting Relational Model Training")
    print("=" * 50)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Grid size: {grid_size}")
    print(f"Difficulty level: {difficulty_level}")
    print(f"Use curriculum: {use_curriculum}")
    print(f"Save path: {save_path}")
    
    # Create environment
    if use_curriculum:
        print("📚 Setting up curriculum learning...")
        # Start with easy environments, gradually increase difficulty
        env_fns = create_curriculum_envs()
        env = make_vec_env(env_fns[0], n_envs=4)  # Start with easiest
    else:
        print("🏗️  Setting up single environment...")
        env_fn = create_relational_env(
            grid_size=grid_size,
            difficulty_level=difficulty_level,
            schematic_data=schematic_data
        )
        env = make_vec_env(env_fn, n_envs=4)
    
    # Normalize environment (only normalize Box spaces)
    env = VecNormalize(
        env, 
        norm_obs=True, 
        norm_reward=True,
        norm_obs_keys=['grid', 'placed_components', 'component_positions']
    )
    
    # Create model with custom policy
    print("🧠 Creating PPO model with relational policy...")
    model = PPO(
        policy=RelationalActorCriticPolicy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs={
            "features_extractor_kwargs": {"features_dim": 512}
        },
        verbose=1
    )
    
    # Setup callbacks
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Checkpoint callback - save model every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.dirname(save_path),
        name_prefix="relational_checkpoint"
    )
    
    # Evaluation callback
    eval_env = make_vec_env(
        create_relational_env(grid_size=grid_size, difficulty_level=difficulty_level),
        n_envs=1
    )
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=True,
        norm_obs_keys=['grid', 'placed_components', 'component_positions'],
        training=False
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(save_path),
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Training with curriculum (if enabled)
    if use_curriculum:
        print("📈 Training with curriculum learning...")
        
        # Phase 1: Easy environments (30% of timesteps)
        phase1_steps = int(total_timesteps * 0.3)
        print(f"Phase 1: Training on easy environments ({phase1_steps} steps)")
        model.learn(
            total_timesteps=phase1_steps,
            callback=callbacks,
            reset_num_timesteps=False
        )
        
        # Phase 2: Medium environments (40% of timesteps)
        print("Switching to medium difficulty environments...")
        env.close()
        env = make_vec_env(create_curriculum_envs()[1], n_envs=4)  # Medium difficulty
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=True,
            norm_obs_keys=['grid', 'placed_components', 'component_positions']
        )
        model.set_env(env)
        
        phase2_steps = int(total_timesteps * 0.4)
        print(f"Phase 2: Training on medium environments ({phase2_steps} steps)")
        model.learn(
            total_timesteps=phase2_steps,
            callback=callbacks,
            reset_num_timesteps=False
        )
        
        # Phase 3: Hard environments (30% of timesteps)
        print("Switching to hard difficulty environments...")
        env.close()
        env = make_vec_env(create_curriculum_envs()[-1], n_envs=4)  # Hard difficulty
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=True,
            norm_obs_keys=['grid', 'placed_components', 'component_positions']
        )
        model.set_env(env)
        
        phase3_steps = total_timesteps - phase1_steps - phase2_steps
        print(f"Phase 3: Training on hard environments ({phase3_steps} steps)")
        model.learn(
            total_timesteps=phase3_steps,
            callback=callbacks,
            reset_num_timesteps=False
        )
        
    else:
        # Standard training
        print("🎯 Starting standard training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
    
    # Save final model
    print(f"💾 Saving final model to {save_path}")
    model.save(save_path)
    
    # Save environment statistics
    env_stats_path = save_path.replace(".zip", "_vecnormalize.pkl")
    env.save(env_stats_path)
    
    print("✅ Training completed successfully!")
    return model


def evaluate_model(model_path, grid_size=15, difficulty_level=2, n_episodes=10):
    """Evaluate a trained relational model."""
    print("🔍 Evaluating trained model...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = RelationalLayoutEnv(grid_size=grid_size, difficulty_level=difficulty_level)
    
    total_rewards = []
    completion_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Use model to predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        # Calculate completion rate
        completion_rate = len(env.placed_components) / len(env.components)
        
        total_rewards.append(episode_reward)
        completion_rates.append(completion_rate)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
              f"Completion = {completion_rate:.1%}")
    
    # Summary statistics
    avg_reward = np.mean(total_rewards)
    avg_completion = np.mean(completion_rates)
    
    print(f"\n📊 Evaluation Results:")
    print(f"Average reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average completion: {avg_completion:.1%} ± {np.std(completion_rates):.1%}")
    print(f"Success rate (100% completion): {np.mean(np.array(completion_rates) == 1.0):.1%}")
    
    return avg_reward, avg_completion


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train relational placement model")
    
    parser.add_argument("--timesteps", type=int, default=200000,
                       help="Total training timesteps")
    parser.add_argument("--grid-size", type=int, default=15,
                       help="Grid size for environment")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3],
                       help="Difficulty level (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--curriculum", action="store_true",
                       help="Use curriculum learning")
    parser.add_argument("--save-path", type=str, default="./models/relational_model.zip",
                       help="Path to save trained model")
    parser.add_argument("--log-dir", type=str, default="./relational_logs/",
                       help="Directory for tensorboard logs")
    parser.add_argument("--evaluate", type=str, default=None,
                       help="Path to model to evaluate (skip training)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluation mode
        evaluate_model(args.evaluate, args.grid_size, args.difficulty)
    else:
        # Training mode
        model = train_relational_model(
            total_timesteps=args.timesteps,
            grid_size=args.grid_size,
            difficulty_level=args.difficulty,
            use_curriculum=args.curriculum,
            save_path=args.save_path,
            log_dir=args.log_dir
        )
        
        # Quick evaluation after training
        print("\n🎯 Running post-training evaluation...")
        evaluate_model(args.save_path, args.grid_size, args.difficulty, n_episodes=5)


if __name__ == "__main__":
    main()