"""
Relational Training Runner Module for AICRL GUI.

This module extends the existing training runner to support relational placement
training with curriculum learning and improved generalization.

Author: AICRL Team
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import threading
import time
from typing import Optional, Dict, Any

from modules.training_runner import TrainingRunnerModule
from RelationalLayoutEnv import RelationalLayoutEnv
from RelationalPolicy import RelationalActorCriticPolicy


class RelationalTrainingRunner(TrainingRunnerModule):
    """Extended training runner with relational placement support."""
    
    def __init__(self):
        super().__init__()
        
        # Relational-specific parameters
        self.use_relational = tk.BooleanVar(value=True)
        self.use_curriculum = tk.BooleanVar(value=False)
        self.difficulty_level = tk.IntVar(value=1)
        
    def create_gui(self, parent_frame):
        """Create GUI with relational options."""
        super().create_gui(parent_frame)
        
        # Add relational options section
        self.create_relational_section(parent_frame)
    
    def create_relational_section(self, parent):
        """Create relational training options section."""
        # Add relational frame after existing sections
        relational_frame = ttk.LabelFrame(parent, text="Relational Training Options", padding="10")
        relational_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        relational_frame.columnconfigure(1, weight=1)
        
        # Use relational placement
        relational_check = ttk.Checkbutton(
            relational_frame, 
            text="Use relational placement (recommended for generalization)",
            variable=self.use_relational
        )
        relational_check.grid(row=0, column=0, columnspan=3, sticky=(tk.W,), pady=(0, 10))
        
        # Curriculum learning
        curriculum_check = ttk.Checkbutton(
            relational_frame,
            text="Enable curriculum learning",
            variable=self.use_curriculum
        )
        curriculum_check.grid(row=1, column=0, columnspan=3, sticky=(tk.W,), pady=(0, 10))
        
        # Difficulty level
        ttk.Label(relational_frame, text="Difficulty Level:").grid(
            row=2, column=0, sticky=(tk.W,), padx=(0, 10)
        )
        
        difficulty_frame = ttk.Frame(relational_frame)
        difficulty_frame.grid(row=2, column=1, sticky=(tk.W,))
        
        ttk.Radiobutton(
            difficulty_frame, text="Easy (3-4 components)", 
            variable=self.difficulty_level, value=1
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            difficulty_frame, text="Medium (5-8 components)", 
            variable=self.difficulty_level, value=2
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            difficulty_frame, text="Hard (all components)", 
            variable=self.difficulty_level, value=3
        ).pack(anchor=tk.W)
        
        # Benefits explanation
        benefits_text = (
            "Relational placement benefits:\n"
            "• Better generalization across circuit sizes\n" 
            "• Learns spatial relationships instead of absolute positions\n"
            "• Curriculum learning gradually increases difficulty\n"
            "• More robust to different component counts"
        )
        
        benefits_label = ttk.Label(
            relational_frame, 
            text=benefits_text, 
            font=('Arial', 8),
            foreground='#7F8C8D',
            justify=tk.LEFT
        )
        benefits_label.grid(row=3, column=0, columnspan=3, sticky=(tk.W,), pady=(10, 0))
    
    def _run_training_thread(self):
        """Run training in separate thread with relational support."""
        try:
            self.is_training = True
            self.add_log_message("Starting relational training...")
            
            # Import required modules
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_util import make_vec_env
            from stable_baselines3.common.vec_env import VecNormalize
            
            # Determine which environment to use
            if self.use_relational.get():
                self.add_log_message("🚀 Using relational placement environment")
                env_class = RelationalLayoutEnv
                policy_class = RelationalActorCriticPolicy
                
                # Environment parameters
                env_kwargs = {
                    "grid_size": int(self.grid_size_var.get()),
                    "schematic_data": self.schematic_data,
                    "difficulty_level": self.difficulty_level.get()
                }
            else:
                self.add_log_message("Using traditional absolute placement environment")
                from AnalogICLayoutEnv import AnalogICLayoutEnv
                env_class = AnalogICLayoutEnv
                policy_class = "MultiInputPolicy"
                
                env_kwargs = {
                    "grid_size": int(self.grid_size_var.get()),
                    "schematic_data": self.schematic_data
                }
            
            # Create environment function
            def make_env():
                return env_class(**env_kwargs)
            
            # Create vectorized environment
            self.add_log_message("Creating training environment...")
            env = make_vec_env(make_env, n_envs=4)
            
            # Normalize environment for relational training
            if self.use_relational.get():
                env = VecNormalize(
                    env, 
                    norm_obs=True, 
                    norm_reward=True,
                    norm_obs_keys=['grid', 'placed_components', 'component_positions']
                )
                self.add_log_message("Environment normalization enabled")
            
            # Create model with appropriate policy
            self.add_log_message("Initializing model...")
            
            if self.use_relational.get():
                model = PPO(
                    policy=policy_class,
                    env=env,
                    learning_rate=self.training_params["learning_rate"],
                    n_steps=self.training_params["n_steps"],
                    batch_size=self.training_params["batch_size"],
                    gamma=self.training_params["gamma"],
                    gae_lambda=self.training_params["gae_lambda"],
                    ent_coef=self.training_params["ent_coef"],
                    verbose=1,
                    tensorboard_log="./relational_tensorboard/",
                    policy_kwargs={"features_extractor_kwargs": {"features_dim": 512}}
                )
                self.add_log_message("✓ Relational PPO model created")
            else:
                model = PPO(
                    policy_class,
                    env,
                    learning_rate=self.training_params["learning_rate"],
                    n_steps=self.training_params["n_steps"],
                    batch_size=self.training_params["batch_size"],
                    gamma=self.training_params["gamma"],
                    gae_lambda=self.training_params["gae_lambda"],
                    ent_coef=self.training_params["ent_coef"],
                    verbose=1,
                    tensorboard_log="./ppo_tensorboard/"
                )
                self.add_log_message("✓ Traditional PPO model created")
            
            # Training with curriculum learning (if enabled)
            total_timesteps = self.training_params["total_timesteps"]
            
            if self.use_relational.get() and self.use_curriculum.get():
                self.add_log_message("📚 Starting curriculum learning...")
                
                # Phase 1: Easy (30% of training)
                phase1_steps = int(total_timesteps * 0.3)
                self.add_log_message(f"Phase 1: Easy difficulty ({phase1_steps} steps)")
                
                model.learn(total_timesteps=phase1_steps, reset_num_timesteps=True)
                
                # Phase 2: Medium (40% of training)
                self.add_log_message("Switching to medium difficulty...")
                env.close()
                
                env_kwargs["difficulty_level"] = 2
                env = make_vec_env(lambda: env_class(**env_kwargs), n_envs=4)
                env = VecNormalize(
                    env, 
                    norm_obs=True, 
                    norm_reward=True,
                    norm_obs_keys=['grid', 'placed_components', 'component_positions']
                )
                model.set_env(env)
                
                phase2_steps = int(total_timesteps * 0.4)
                self.add_log_message(f"Phase 2: Medium difficulty ({phase2_steps} steps)")
                
                model.learn(total_timesteps=phase2_steps, reset_num_timesteps=False)
                
                # Phase 3: Hard (30% of training)
                self.add_log_message("Switching to hard difficulty...")
                env.close()
                
                env_kwargs["difficulty_level"] = 3
                env = make_vec_env(lambda: env_class(**env_kwargs), n_envs=4)
                env = VecNormalize(
                    env, 
                    norm_obs=True, 
                    norm_reward=True,
                    norm_obs_keys=['grid', 'placed_components', 'component_positions']
                )
                model.set_env(env)
                
                phase3_steps = total_timesteps - phase1_steps - phase2_steps
                self.add_log_message(f"Phase 3: Hard difficulty ({phase3_steps} steps)")
                
                model.learn(total_timesteps=phase3_steps, reset_num_timesteps=False)
                
                self.add_log_message("✓ Curriculum training completed")
                
            else:
                # Standard training
                self.add_log_message(f"Starting standard training ({total_timesteps} timesteps)")
                model.learn(total_timesteps=total_timesteps)
                self.add_log_message("✓ Standard training completed")
            
            # Save trained model
            timestamp = int(time.time())
            if self.use_relational.get():
                model_filename = f"relational_model_{timestamp}.zip"
            else:
                model_filename = f"traditional_model_{timestamp}.zip"
            
            model_path = f"./models/{model_filename}"
            os.makedirs("./models", exist_ok=True)
            
            model.save(model_path)
            self.add_log_message(f"✓ Model saved: {model_filename}")
            
            # Save environment normalization stats if used
            if self.use_relational.get():
                env_stats_path = model_path.replace(".zip", "_vecnormalize.pkl")
                env.save(env_stats_path)
                self.add_log_message("✓ Environment normalization stats saved")
            
            # Update available models
            self.discover_available_models()
            
            self.add_log_message("🎉 Training completed successfully!")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.add_log_message(error_msg)
            messagebox.showerror("Training Error", error_msg)
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_training = False
    
    def _run_generation_thread(self):
        """Run layout generation with relational support."""
        try:
            self.is_generating = True
            self.add_log_message("Starting relational layout generation...")
            
            # Import required modules
            from stable_baselines3 import PPO
            
            # Determine environment type
            if self.use_relational.get():
                from RelationalLayoutEnv import RelationalLayoutEnv
                env_class = RelationalLayoutEnv
                
                env_kwargs = {
                    "grid_size": int(self.grid_size_var.get()),
                    "schematic_data": self.schematic_data,
                    "difficulty_level": self.difficulty_level.get()
                }
                
                self.add_log_message("🎯 Using relational environment for generation")
            else:
                from AnalogICLayoutEnv import AnalogICLayoutEnv
                env_class = AnalogICLayoutEnv
                
                env_kwargs = {
                    "grid_size": int(self.grid_size_var.get()),
                    "schematic_data": self.schematic_data
                }
                
                self.add_log_message("Using traditional environment for generation")
            
            # Create environment
            env = env_class(**env_kwargs)
            
            # Load model
            model = self.current_model
            if not model:
                raise ValueError("No model loaded for generation")
            
            self.add_log_message("🎲 Starting layout generation...")
            
            # Generate multiple layouts and select best
            best_layout = None
            best_score = -float('inf')
            num_attempts = 3
            
            for attempt in range(num_attempts):
                self.add_log_message(f"Generation attempt {attempt + 1}/{num_attempts}...")
                
                obs, info = env.reset()
                total_reward = 0
                steps = 0
                
                while steps < 50:  # Maximum steps per episode
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                # Calculate final score
                completion_rate = len(env.placed_components) / len(env.components)
                final_score = total_reward + completion_rate * 100  # Bonus for completion
                
                self.add_log_message(f"  Attempt {attempt + 1}: {steps} steps, "
                                   f"score: {final_score:.2f}, "
                                   f"completion: {completion_rate:.1%}")
                
                if final_score > best_score:
                    best_score = final_score
                    best_layout = {
                        'env': env,
                        'total_reward': total_reward,
                        'steps': steps,
                        'completion_rate': completion_rate,
                        'attempt': attempt + 1
                    }
                    
                    # Save current state
                    best_placements = dict(env.placements)
                    best_grid = env.grid.copy()
            
            if best_layout:
                self.add_log_message(f"✓ Best layout from attempt {best_layout['attempt']} "
                                   f"(score: {best_score:.2f})")
                
                # Create output directory
                timestamp = int(time.time())
                output_dir = f"./output/relational_layout_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate visualizations
                self.add_log_message("Creating visualizations...")
                
                try:
                    # Use relational environment's render method
                    env.render(mode='human')
                    
                    # Save layout data
                    layout_data = {
                        "timestamp": timestamp,
                        "model_type": "relational" if self.use_relational.get() else "traditional",
                        "placements": {str(k): v for k, v in best_placements.items()},
                        "grid_size": env.grid_size,
                        "components": env.components,
                        "connections": env.connections,
                        "total_reward": best_layout['total_reward'],
                        "steps": best_layout['steps'],
                        "completion_rate": best_layout['completion_rate']
                    }
                    
                    with open(f"{output_dir}/layout_data.json", 'w') as f:
                        import json
                        json.dump(layout_data, f, indent=2, default=str)
                    
                    self.add_log_message("✓ Visualizations and data saved")
                    
                except Exception as e:
                    self.add_log_message(f"⚠️ Visualization error: {str(e)}")
                
                # Final success message
                self.add_log_message("🎉 Relational layout generation completed!")
                self.add_log_message(f"📁 Results saved to: {output_dir}")
                self.add_log_message(f"📊 Final completion: {best_layout['completion_rate']:.1%}")
                
            else:
                self.add_log_message("❌ No valid layout generated")
                
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            self.add_log_message(error_msg)
            messagebox.showerror("Generation Error", error_msg)
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_generating = False