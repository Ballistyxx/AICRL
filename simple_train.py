#!/usr/bin/env python3
"""
Simple training script for relational placement model without evaluation.

This script trains a PPO agent using the RelationalLayoutEnv and RelationalPolicy.
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


def train_simple(
    total_timesteps=50000,
    grid_size=15,
    difficulty_level=1,
    save_path="./models/simple_relational.zip",
    log_dir="./simple_logs/"
):
    """
    Simple training without callbacks or evaluation.
    """
    
    print("🚀 Starting Simple Relational Training")
    print("=" * 50)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Grid size: {grid_size}")
    print(f"Difficulty level: {difficulty_level}")
    print(f"Save path: {save_path}")
    
    # Create environment
    print("🏗️  Setting up environment...")
    env_fn = create_relational_env(
        grid_size=grid_size,
        difficulty_level=difficulty_level,
        schematic_data=None
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
    
    # Setup directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Simple training without callbacks
    print("🎯 Starting training...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save final model
    print(f"💾 Saving final model to {save_path}")
    model.save(save_path)
    
    # Save environment statistics
    env_stats_path = save_path.replace(".zip", "_vecnormalize.pkl")
    env.save(env_stats_path)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved: {save_path}")
    print(f"📊 Environment stats saved: {env_stats_path}")
    
    return model


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Simple relational placement training")
    
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--grid-size", type=int, default=15,
                       help="Grid size for environment")
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3],
                       help="Difficulty level (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--save-path", type=str, default="./models/simple_relational.zip",
                       help="Path to save trained model")
    parser.add_argument("--log-dir", type=str, default="./simple_logs/",
                       help="Directory for tensorboard logs")
    
    args = parser.parse_args()
    
    # Training
    model = train_simple(
        total_timesteps=args.timesteps,
        grid_size=args.grid_size,
        difficulty_level=args.difficulty,
        save_path=args.save_path,
        log_dir=args.log_dir
    )
    
    print("🎉 Training completed! You can now:")
    print(f"   1. Use the GUI to load model: {args.save_path}")
    print("   2. Run layout generation with the trained model")
    print("   3. View tensorboard logs:")
    print(f"      tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()