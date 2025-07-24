"""
Training Runner Module for AICRL GUI.

This module handles the training and execution of AI models for IC layout generation.
Provides interfaces for model training, layout generation, and progress monitoring.

Author: AICRL Team
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import json
import threading
import queue
import subprocess
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path

# Add venv site-packages to Python path if not already available
def ensure_venv_access():
    """Ensure access to virtual environment packages."""
    try:
        import stable_baselines3
    except ImportError:
        # Try to add venv site-packages to path
        current_dir = Path(__file__).parent.parent.absolute()
        venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
        
        if venv_site_packages.exists():
            sys.path.insert(0, str(venv_site_packages))
            try:
                import stable_baselines3
                print(f"✓ Added venv path: {venv_site_packages}")
            except ImportError:
                raise ImportError("stable_baselines3 not found even after adding venv path")
        else:
            raise ImportError(f"Virtual environment not found at: {venv_site_packages}")

# Ensure venv access on module import
ensure_venv_access()


class TrainingProgressCallback:
    """Custom callback for tracking training progress."""
    
    def __init__(self, total_timesteps: int, training_runner, progress_callback: Optional[Callable] = None):
        """Initialize the progress callback."""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class SB3Callback(BaseCallback):
            def __init__(self, outer_total_timesteps, outer_training_runner, outer_progress_callback):
                super().__init__()
                self.total_timesteps = outer_total_timesteps
                self.training_runner = outer_training_runner
                self.progress_callback = outer_progress_callback
                self.last_log_step = 0
                self.log_interval = max(1000, outer_total_timesteps // 100)
            
            def _on_step(self) -> bool:
                """Called at each step during training."""
                current_step = self.num_timesteps
                progress = min((current_step / self.total_timesteps) * 100, 100)
                
                # Update GUI progress
                if self.training_runner.is_training:
                    self.training_runner.training_progress_var.set(progress)
                    
                    if self.progress_callback:
                        self.progress_callback(progress)
                    
                    # Log progress at intervals
                    if (current_step - self.last_log_step) >= self.log_interval:
                        self.training_runner.add_log_message(
                            f"Training progress: {progress:.1f}% ({current_step}/{self.total_timesteps} timesteps)"
                        )
                        self.last_log_step = current_step
                        
                        # Log episode rewards if available
                        if len(self.locals.get('ep_info_buffer', [])) > 0:
                            recent_rewards = [ep['r'] for ep in self.locals['ep_info_buffer'][-10:]]
                            if recent_rewards:
                                avg_reward = sum(recent_rewards) / len(recent_rewards)
                                self.training_runner.add_log_message(
                                    f"   Average episode reward (last 10): {avg_reward:.2f}"
                                )
                
                # Return True to continue training, False to stop
                return self.training_runner.is_training
        
        self.callback = SB3Callback(total_timesteps, training_runner, progress_callback)


class TrainingRunnerModule:
    """Module for handling AI model training and layout generation."""
    
    def __init__(self):
        """Initialize the training runner module."""
        self.is_training = False
        self.is_generating = False
        self.current_model = None
        self.loaded_model_path = None
        self.training_thread = None
        self.progress_queue = queue.Queue()
        self.available_models = []
        self.schematic_data = None
        
        # Model file selection (for browse functionality)
        self.model_file_var = tk.StringVar(value="No model selected")
        
        # Default training parameters
        self.training_params = {
            "total_timesteps": 100000,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "n_steps": 2048,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01
        }
        
        # Discover available models
        self.discover_available_models()
        
    def create_gui(self, parent_frame):
        """Create the GUI components for the training runner module."""
        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)
        
        # Main content frame
        content_frame = ttk.Frame(parent_frame)
        content_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(2, weight=1)
        
        # Model selection section
        self.create_model_section(content_frame)
        
        # Training parameters section
        self.create_parameters_section(content_frame)
        
        # Status and progress section
        self.create_status_section(content_frame)
        
    def create_model_section(self, parent):
        """Create the model selection section."""
        # Model frame
        model_frame = ttk.LabelFrame(parent, text="Model Configuration", padding="10")
        model_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(1, weight=1)
        
        # Model type selection
        ttk.Label(model_frame, text="Algorithm:").grid(row=0, column=0, sticky=(tk.W,), padx=(0, 10))
        self.algorithm_var = tk.StringVar(value="PPO")
        algorithm_combo = ttk.Combobox(model_frame, textvariable=self.algorithm_var,
                                      values=["PPO", "A2C", "DQN", "SAC"], state="readonly")
        algorithm_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Grid size
        ttk.Label(model_frame, text="Grid Size:").grid(row=0, column=2, sticky=(tk.W,), padx=(10, 10))
        self.grid_size_var = tk.StringVar(value="20")
        grid_size_spinbox = ttk.Spinbox(model_frame, from_=10, to=50, width=10,
                                       textvariable=self.grid_size_var)
        grid_size_spinbox.grid(row=0, column=3, sticky=(tk.W,))
        
        # Load existing model option
        self.load_existing_var = tk.BooleanVar(value=True)  # Default to using existing models
        load_check = ttk.Checkbutton(model_frame, text="Use existing trained model",
                                    variable=self.load_existing_var,
                                    command=self.toggle_model_loading)
        load_check.grid(row=1, column=0, columnspan=2, sticky=(tk.W,), pady=(10, 0))
        
        # Available models dropdown
        ttk.Label(model_frame, text="Model:").grid(row=1, column=2, sticky=(tk.W,), padx=(10, 10))
        self.model_selection_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_selection_var,
                                       values=self.get_model_names(), state="readonly", width=20)
        self.model_combo.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=(10, 0))
        if self.available_models:
            self.model_combo.set(self.available_models[0]["name"])
        
        # Model info display
        self.model_info_var = tk.StringVar(value="Select a model to see details")
        self.model_info_label = ttk.Label(model_frame, textvariable=self.model_info_var,
                                         foreground='#7F8C8D', font=('Arial', 8))
        self.model_info_label.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), 
                                  padx=(0, 0), pady=(5, 0))
        
        # Browse for other models button
        self.browse_model_btn = ttk.Button(model_frame, text="Browse Other...",
                                          command=self.browse_model_file,
                                          state=tk.DISABLED)
        self.browse_model_btn.grid(row=3, column=2, columnspan=2, sticky=(tk.W,), 
                                  padx=(10, 0), pady=(5, 0))
                                  
        # Bind model selection change
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selection_change)
        
    def create_parameters_section(self, parent):
        """Create the training parameters section."""
        # Parameters frame
        params_frame = ttk.LabelFrame(parent, text="Training Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create a grid for parameters
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        # Total timesteps
        ttk.Label(params_frame, text="Total Timesteps:").grid(row=0, column=0, sticky=(tk.W,), padx=(0, 10))
        self.timesteps_var = tk.StringVar(value=str(self.training_params["total_timesteps"]))
        timesteps_entry = ttk.Entry(params_frame, textvariable=self.timesteps_var, width=15)
        timesteps_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 20))
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=2, sticky=(tk.W,), padx=(0, 10))
        self.lr_var = tk.StringVar(value=str(self.training_params["learning_rate"]))
        lr_entry = ttk.Entry(params_frame, textvariable=self.lr_var, width=15)
        lr_entry.grid(row=0, column=3, sticky=(tk.W, tk.E))
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky=(tk.W,), padx=(0, 10))
        self.batch_size_var = tk.StringVar(value=str(self.training_params["batch_size"]))
        batch_size_entry = ttk.Entry(params_frame, textvariable=self.batch_size_var, width=15)
        batch_size_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 20), pady=(5, 0))
        
        # N steps
        ttk.Label(params_frame, text="N Steps:").grid(row=1, column=2, sticky=(tk.W,), padx=(0, 10))
        self.n_steps_var = tk.StringVar(value=str(self.training_params["n_steps"]))
        n_steps_entry = ttk.Entry(params_frame, textvariable=self.n_steps_var, width=15)
        n_steps_entry.grid(row=1, column=3, sticky=(tk.W, tk.E), pady=(5, 0))
        
    def create_status_section(self, parent):
        """Create the status and progress section."""
        # Status frame
        status_frame = ttk.LabelFrame(parent, text="Training Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(2, weight=1)
        
        # Current status
        self.training_status_var = tk.StringVar(value="Ready to train")
        status_label = ttk.Label(status_frame, textvariable=self.training_status_var,
                                font=('Arial', 10, 'bold'))
        status_label.grid(row=0, column=0, sticky=(tk.W,), pady=(0, 10))
        
        # Progress bar
        self.training_progress_var = tk.DoubleVar()
        self.training_progress_bar = ttk.Progressbar(status_frame, 
                                                    variable=self.training_progress_var,
                                                    maximum=100, length=300)
        self.training_progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Log output
        log_frame = ttk.Frame(status_frame)
        log_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.WORD, 
                               font=('Courier', 9), state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log scrollbar
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Initial log message
        self.add_log_message("Training module initialized. Ready to start.")
        
    def toggle_model_loading(self):
        """Toggle the model loading interface."""
        if self.load_existing_var.get():
            self.model_combo.config(state="readonly")
            self.browse_model_btn.config(state=tk.NORMAL)
            # Show model information for selected model
            if self.available_models:
                self.on_model_selection_change(None)
        else:
            self.model_combo.config(state=tk.DISABLED)
            self.browse_model_btn.config(state=tk.DISABLED)
            self.model_info_var.set("Training new model from scratch")
            
    def browse_model_file(self):
        """Browse for an existing model file."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[
                ("Model Files", "*.zip *.pkl"),
                ("All Files", "*.*")
            ],
            initialdir="./models"
        )
        
        if filename:
            self.model_file_var.set(filename)
            self.add_log_message(f"Selected model file: {os.path.basename(filename)}")
            
    def add_log_message(self, message: str):
        """Add a message to the log output."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def update_training_parameters(self):
        """Update training parameters from GUI values."""
        try:
            self.training_params["total_timesteps"] = int(self.timesteps_var.get())
            self.training_params["learning_rate"] = float(self.lr_var.get())
            self.training_params["batch_size"] = int(self.batch_size_var.get())
            self.training_params["n_steps"] = int(self.n_steps_var.get())
            return True
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
            return False
            
    def start_training(self, progress_callback: Optional[Callable] = None) -> bool:
        """Start the training process."""
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return False
            
        if not self.update_training_parameters():
            return False
            
        try:
            self.is_training = True
            self.training_status_var.set("Initializing training...")
            self.add_log_message("Starting training process...")
            
            # Start training in a separate thread
            self.training_thread = threading.Thread(
                target=self._run_training_thread,
                args=(progress_callback,),
                daemon=True
            )
            self.training_thread.start()
            
            return True
            
        except Exception as e:
            self.is_training = False
            self.training_status_var.set("Training failed")
            self.add_log_message(f"Error starting training: {str(e)}")
            return False
            
    def _run_training_thread(self, progress_callback: Optional[Callable] = None):
        """Run the training process in a separate thread."""
        try:
            # Import here to avoid GUI blocking
            from AnalogICLayoutEnv import AnalogICLayoutEnv
            from stable_baselines3 import PPO
            
            self.add_log_message("Creating environment...")
            grid_size = int(self.grid_size_var.get())
            
            # Pass schematic data to environment if available
            env = AnalogICLayoutEnv(grid_size=grid_size, schematic_data=self.schematic_data)
            
            self.add_log_message("Initializing model...")
            
            if self.load_existing_var.get() and self.model_selection_var.get():
                # Use existing model loading logic with compatibility checking
                selected_model_name = self.model_selection_var.get()
                model_info = next((m for m in self.available_models if m["name"] == selected_model_name), None)
                
                if model_info:
                    try:
                        # Use the load_selected_model method which includes compatibility checking
                        model = self._load_model_with_compatibility_check(model_info, env)
                        self.add_log_message(f"Model ready for training: {selected_model_name}")
                    except Exception as e:
                        self.add_log_message(f"Model loading failed: {str(e)}")
                        self.add_log_message("Creating new model for training...")
                        model = self._create_new_model(env)
                else:
                    # Fallback to creating new model
                    self.add_log_message("Selected model not found, creating new model...")
                    model = self._create_new_model(env)
            else:
                # Create new model
                model = self._create_new_model(env)
            
            self.training_status_var.set("Training in progress...")
            self.add_log_message("Starting model training...")
            
            # Actual PPO training with callbacks
            total_timesteps = self.training_params["total_timesteps"]
            
            # Create custom callback for progress tracking
            progress_callback_obj = TrainingProgressCallback(
                total_timesteps=total_timesteps,
                training_runner=self,
                progress_callback=progress_callback
            )
            
            self.add_log_message(f"Starting PPO training for {total_timesteps} timesteps...")
            self.add_log_message(f"Environment: {len(env.components)} components, grid size {env.grid_size}x{env.grid_size}")
            
            try:
                # Actual model training using stable-baselines3
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=progress_callback_obj.callback,
                    progress_bar=False  # We use our own progress tracking
                )
                
                if self.is_training:  # Training completed successfully
                    # Save the trained model
                    os.makedirs("./models", exist_ok=True)
                    timestamp = int(time.time())
                    
                    if self.schematic_data:
                        # Save as schematic-specific model
                        component_count = len(env.components)
                        model_path = f"./models/schematic_trained_{component_count}comp_{timestamp}.zip"
                    else:
                        # Save as general model
                        model_path = f"./models/trained_model_{timestamp}.zip"
                    
                    model.save(model_path)
                    
                    self.current_model = model
                    self.training_status_var.set("Training completed successfully!")
                    self.add_log_message(f"🎉 Training completed! Model saved to {model_path}")
                    self.training_progress_var.set(100)
                    
                    # Update available models list
                    self.discover_available_models()
                    
                    if progress_callback:
                        progress_callback(100)
                else:
                    self.training_status_var.set("Training cancelled")
                    self.add_log_message("Training was cancelled by user")
                    
            except Exception as training_error:
                self.training_status_var.set("Training failed")
                self.add_log_message(f"Training failed: {str(training_error)}")
                raise training_error
                
        except Exception as e:
            self.training_status_var.set("Training failed")
            self.add_log_message(f"Training error: {str(e)}")
            
        finally:
            self.is_training = False
            
    def stop_training(self):
        """Stop the current training process."""
        if self.is_training:
            self.is_training = False
            self.add_log_message("Stopping training...")
            self.training_status_var.set("Stopping training...")
            
    def generate_layout(self) -> bool:
        """Generate a layout using the trained model."""
        if self.is_generating:
            messagebox.showwarning("Warning", "Layout generation is already in progress.")
            return False
            
        # Check if we have a model to use
        if not self.current_model and self.load_existing_var.get():
            try:
                self.add_log_message("Loading selected model...")
                self.load_selected_model()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                return False
                
        if not self.current_model:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first or select an existing one.")
            return False
            
        try:
            self.is_generating = True
            self.add_log_message("Starting layout generation...")
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._run_generation_thread,
                daemon=True
            )
            generation_thread.start()
            
            return True
            
        except Exception as e:
            self.is_generating = False
            self.add_log_message(f"Error starting layout generation: {str(e)}")
            return False
            
    def _run_generation_thread(self):
        """Run the layout generation process in a separate thread."""
        try:
            # Import here to avoid GUI blocking
            from AnalogICLayoutEnv import AnalogICLayoutEnv
            from enhanced_visualizer import EnhancedLayoutVisualizer
            from layout_analyzer import LayoutAnalyzer
            
            self.add_log_message("Creating environment for generation...")
            grid_size = int(self.grid_size_var.get())
            
            # Pass schematic data to environment if available
            env = AnalogICLayoutEnv(grid_size=grid_size, schematic_data=self.schematic_data)
            
            # Use current loaded model
            model = self.current_model
                
            self.add_log_message(f"Using model from: {os.path.basename(self.loaded_model_path or 'memory')}")
            
            # If we have schematic data, try to incorporate it
            if self.schematic_data:
                self.add_log_message("Incorporating schematic constraints...")
                component_count = len(self.schematic_data.get("components", []))
                self.add_log_message(f"Target components from schematic: {component_count}")
                
            # Generate layout with multiple attempts for best result
            best_layout = None
            best_score = -float('inf')
            num_attempts = 3  # Generate multiple layouts and pick the best
            
            for attempt in range(num_attempts):
                self.add_log_message(f"Generation attempt {attempt + 1}/{num_attempts}...")
                
                # Reset environment
                obs, info = env.reset()
                done = False
                step = 0
                MAX_STEPS = 150  # Increased for more complex layouts
                total_reward = 0
                
                while not done and step < MAX_STEPS:
                    step += 1
                    action, _states = model.predict(obs, deterministic=False)  # Use stochastic for variety
                    obs, reward, done, truncated, info = env.step(action.item())
                    total_reward += reward
                    
                    if step % 25 == 0:
                        self.add_log_message(f"  Step {step}/{MAX_STEPS}, Total reward: {total_reward:.2f}")
                        
                # Evaluate this layout
                analyzer = LayoutAnalyzer(env)
                analysis = analyzer.analyze_layout()
                layout_score = analysis.get('compactness_score', 0) + analysis.get('connectivity_score', 0)
                
                self.add_log_message(f"  Attempt {attempt + 1}: {step} steps, score: {layout_score:.2f}")
                
                if layout_score > best_score:
                    best_score = layout_score
                    best_layout = {
                        'env': env,
                        'analysis': analysis,
                        'steps': step,
                        'reward': total_reward,
                        'attempt': attempt + 1
                    }
                    # Save current env state
                    best_placements = env.placements.copy()
                    
            if best_layout:
                self.add_log_message(f"Best layout from attempt {best_layout['attempt']} (score: {best_score:.2f})")
                
                # Restore best layout
                env.placements = best_placements
                
                # Create comprehensive output
                timestamp = int(time.time())
                output_dir = f"./output/layout_{timestamp}"
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate visualizations
                self.add_log_message("Creating comprehensive visualizations...")
                visualizer = EnhancedLayoutVisualizer(style='professional')
                
                # Main layout visualization
                layout_path = f"{output_dir}/layout_professional.png"
                try:
                    visualizer.render_layout_professional(env, save_path=layout_path, show_metrics=True)
                    self.add_log_message(f"✓ Professional layout saved: {layout_path}")
                except Exception as e:
                    self.add_log_message(f"⚠️ Professional layout error: {str(e)}")
                
                # Create simple final layout visualization (like training end visualization)
                simple_layout_path = f"{output_dir}/final_layout_simple.png"
                try:
                    self._create_simple_layout_visualization(env, simple_layout_path)
                    self.add_log_message(f"✓ Simple layout saved: {simple_layout_path}")
                except Exception as e:
                    self.add_log_message(f"⚠️ Simple layout error: {str(e)}")
                
                # Create training report for this generation
                report_path = f"{output_dir}/generation_report.txt"
                self._create_generation_report(best_layout, report_path)
                
                # Save layout data as JSON
                data_path = f"{output_dir}/layout_data.json"
                layout_data = {
                    "timestamp": timestamp,
                    "model_path": self.loaded_model_path,
                    "grid_size": grid_size,
                    "placements": {str(k): v for k, v in env.placements.items()},
                    "components": [comp for comp in env.components],
                    "connections": env.connections,
                    "analysis": best_layout['analysis'],
                    "generation_stats": {
                        "steps": best_layout['steps'],
                        "total_reward": best_layout['reward'],
                        "best_attempt": best_layout['attempt'],
                        "final_score": best_score
                    }
                }
                
                if self.schematic_data:
                    layout_data["schematic_info"] = {
                        "source_file": self.schematic_data.get("file", "unknown"),
                        "components": len(self.schematic_data.get("components", [])),
                        "format": self.schematic_data.get("format", "unknown")
                    }
                
                with open(data_path, 'w') as f:
                    json.dump(layout_data, f, indent=2, default=str)
                
                self.add_log_message(f"🎉 Layout generation completed successfully!")
                self.add_log_message(f"📁 Results saved to: {output_dir}/")
                self.add_log_message(f"📊 Quality score: {best_score:.2f}")
                self.add_log_message(f"🔧 Components placed: {len(env.placements)}")
                self.add_log_message(f"")
                self.add_log_message(f"📸 Generated images:")
                self.add_log_message(f"   • Professional: layout_professional.png")
                self.add_log_message(f"   • Simple view: final_layout_simple.png")
                self.add_log_message(f"📄 Report: generation_report.txt")
                self.add_log_message(f"💾 Data: layout_data.json")
                
                # Show key metrics
                analysis = best_layout['analysis']
                self.add_log_message(f"Metrics - Compactness: {analysis.get('compactness_score', 0):.1f}%, " +
                                   f"Connectivity: {analysis.get('connectivity_score', 0):.1f}%, " +
                                   f"Completion: {analysis.get('completion_rate', 0)*100:.1f}%")
                
            else:
                self.add_log_message("Layout generation failed - no valid layouts produced")
            
        except Exception as e:
            self.add_log_message(f"Layout generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_generating = False
            
    def _create_generation_report(self, layout_info: Dict[str, Any], report_path: str):
        """Create a detailed generation report."""
        try:
            with open(report_path, 'w') as f:
                f.write("Layout Generation Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Used: {os.path.basename(self.loaded_model_path or 'Unknown')}\n")
                f.write(f"Grid Size: {self.grid_size_var.get()}\n")
                f.write(f"Generation Steps: {layout_info['steps']}\n")
                f.write(f"Total Reward: {layout_info['reward']:.2f}\n")
                f.write(f"Best Attempt: {layout_info['attempt']}\n\n")
                
                if self.schematic_data:
                    f.write("Schematic Information:\n")
                    f.write("-" * 25 + "\n")
                    f.write(f"Source File: {self.schematic_data.get('file', 'Unknown')}\n")
                    f.write(f"Format: {self.schematic_data.get('format', 'Unknown')}\n")
                    f.write(f"Components: {len(self.schematic_data.get('components', []))}\n")
                    f.write(f"Nets: {len(self.schematic_data.get('nets', []))}\n\n")
                
                f.write("Layout Analysis:\n")
                f.write("-" * 20 + "\n")
                analysis = layout_info['analysis']
                for key, value in analysis.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                        
        except Exception as e:
            self.add_log_message(f"Error creating report: {str(e)}")
            
    def _create_simple_layout_visualization(self, env, save_path: str):
        """Create a simple layout visualization similar to training end visualization."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Set up the plot
        ax.set_xlim(-0.5, env.grid_size + 0.5)
        ax.set_ylim(-0.5, env.grid_size + 0.5)
        ax.set_aspect('equal')
        
        # Add grid
        ax.set_xticks(range(env.grid_size + 1))
        ax.set_yticks(range(env.grid_size + 1))
        ax.grid(True, alpha=0.3)
        
        # Color mapping for component types
        color_map = {
            'nfet': '#FF6B6B',      # Red
            'pfet': '#4ECDC4',      # Teal  
            'cap': '#45B7D1',       # Blue
            'resistor': '#FFA07A',  # Light salmon
            'inductor': '#98D8C8',  # Mint green
            'default': '#95A5A6'    # Gray
        }
        
        # Draw placed components
        for (x, y), comp_type in env.placements.items():
            color = color_map.get(comp_type, color_map['default'])
            
            # Create rectangle for component
            rect = patches.Rectangle((x, y), 1, 1, 
                                   linewidth=2, 
                                   edgecolor='black', 
                                   facecolor=color,
                                   alpha=0.8)
            ax.add_patch(rect)
            
            # Add component type label
            ax.text(x + 0.5, y + 0.5, comp_type, 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Add connections if they exist
        if hasattr(env, 'connections') and env.connections:
            for connection in env.connections:
                # Simple connection visualization - this would need to be adapted
                # based on your specific connection representation
                pass
        
        # Set title and labels
        ax.set_title(f'Final IC Layout - {len(env.placements)} Components Placed', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        
        # Add component count info
        component_counts = {}
        for comp_type in env.placements.values():
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        # Create legend
        legend_elements = []
        for comp_type, count in component_counts.items():
            color = color_map.get(comp_type, color_map['default'])
            legend_elements.append(patches.Patch(color=color, label=f'{comp_type}: {count}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Save the figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
            
    def get_training_status(self) -> Dict[str, Any]:
        """Get the current training status."""
        return {
            "is_training": self.is_training,
            "is_generating": self.is_generating,
            "current_model": self.current_model is not None,
            "available_models": len(self.available_models),
            "parameters": self.training_params.copy()
        }
        
    def discover_available_models(self):
        """Discover available trained models in the models directory."""
        self.available_models = []
        models_dir = "./models"
        
        if not os.path.exists(models_dir):
            return
            
        try:
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                
                if item.endswith('.zip'):
                    # Handle zipped models
                    model_info = self._analyze_zipped_model(item_path)
                    if model_info:
                        self.available_models.append(model_info)
                elif os.path.isdir(item_path) and not item.startswith('.'):
                    # Handle unzipped models
                    model_info = self._analyze_unzipped_model(item_path)
                    if model_info:
                        self.available_models.append(model_info)
                        
            # Sort by modification time (newest first)
            self.available_models.sort(key=lambda x: x.get("modified", 0), reverse=True)
            
        except Exception as e:
            print(f"Error discovering models: {e}")
            
    def _analyze_zipped_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a zipped model file."""
        try:
            stat = os.stat(model_path)
            name = os.path.basename(model_path)
            
            # Determine model type
            if "schematic_model" in name:
                # Extract component count from schematic model name
                import re
                match = re.search(r'(\d+)_components', name)
                component_count = match.group(1) if match else "Unknown"
                model_description = f"Schematic model ({component_count} components)"
            else:
                model_description = "General purpose model"
            
            return {
                "name": name,
                "path": model_path,
                "type": "zipped",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "status": "Ready to load",
                "description": model_description
            }
        except Exception:
            return None
            
    def _analyze_unzipped_model(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Analyze an unzipped model directory."""
        try:
            # Check for required model files
            required_files = ["policy.pth", "pytorch_variables.pth"]
            has_all_files = all(os.path.exists(os.path.join(model_path, f)) 
                              for f in required_files)
            
            if not has_all_files:
                return None
                
            stat = os.stat(model_path)
            name = os.path.basename(model_path)
            
            # Try to read system info if available
            info = {"algorithm": "PPO"}  # Default
            system_info_path = os.path.join(model_path, "system_info.txt")
            if os.path.exists(system_info_path):
                try:
                    with open(system_info_path, 'r') as f:
                        content = f.read()
                        if "Stable-Baselines3:" in content:
                            # Extract SB3 version info
                            info["sb3_version"] = content.split("Stable-Baselines3:")[1].split()[0]
                except Exception:
                    pass
                    
            return {
                "name": name,
                "path": model_path,
                "type": "unzipped",
                "size": self._get_dir_size(model_path),
                "modified": stat.st_mtime,
                "status": "Ready to load",
                "info": info
            }
        except Exception:
            return None
            
    def _get_dir_size(self, path: str) -> int:
        """Get the total size of a directory."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
        
    def get_model_names(self) -> list[str]:
        """Get a list of available model names for the dropdown."""
        return [model["name"] for model in self.available_models]
        
    def get_selected_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the currently selected model."""
        selected_name = self.model_selection_var.get()
        for model in self.available_models:
            if model["name"] == selected_name:
                return model
        return None
        
    def on_model_selection_change(self, event):
        """Handle model selection change in the dropdown."""
        model_info = self.get_selected_model_info()
        if model_info:
            size_mb = model_info["size"] / (1024 * 1024)
            modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(model_info["modified"]))
            status = model_info["status"]
            
            info_text = f"{status} | Size: {size_mb:.1f} MB | Modified: {modified}"
            if model_info.get("info", {}).get("sb3_version"):
                info_text += f" | SB3: {model_info['info']['sb3_version']}"
                
            self.model_info_var.set(info_text)
            self.add_log_message(f"Selected model: {model_info['name']}")
        else:
            self.model_info_var.set("Model information not available")
            
    def load_selected_model(self) -> Optional[Any]:
        """Load the currently selected model."""
        model_info = self.get_selected_model_info()
        if not model_info:
            raise ValueError("No model selected")
            
        model_path = model_info["path"]
        self.add_log_message(f"Loading model from: {model_path}")
        
        # Import here to avoid GUI blocking
        from stable_baselines3 import PPO
        from AnalogICLayoutEnv import AnalogICLayoutEnv
        
        # Create environment for model loading
        grid_size = int(self.grid_size_var.get())
        env = AnalogICLayoutEnv(grid_size=grid_size, schematic_data=self.schematic_data)
        
        try:
            # Check if model is compatible with current environment
            if not self._check_model_compatibility(model_path, env):
                # Model incompatible - create new model
                self.add_log_message(f"⚠️  Model incompatible with schematic (different component count)")
                self.add_log_message(f"🔄 Creating new model for {len(env.components)} components...")
                
                model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
                self.add_log_message(f"✓ New model created for schematic")
                
                # Save the new model
                new_model_path = f"./models/schematic_model_{len(env.components)}_components.zip"
                os.makedirs("./models", exist_ok=True)
                model.save(new_model_path)
                self.add_log_message(f"💾 New model saved: {new_model_path}")
                
            else:
                # Model compatible - load normally
                if model_info["type"] == "zipped":
                    # Load from zip file
                    model = PPO.load(model_path, env=env)
                else:
                    # Load from directory
                    model = PPO.load(model_path, env=env)
                    
                self.add_log_message(f"✓ Model loaded successfully: {model_info['name']}")
                
            self.current_model = model
            self.loaded_model_path = model_path
            return model
            
        except Exception as e:
            error_msg = f"Failed to load/create model: {str(e)}"
            self.add_log_message(error_msg)
            raise ValueError(error_msg)
            
    def _check_model_compatibility(self, model_path: str, env) -> bool:
        """Check if a model is compatible with the current environment."""
        try:
            # Import PPO for compatibility check
            from stable_baselines3 import PPO
            
            # Try to create a temporary model to check observation/action spaces
            temp_model = PPO.load(model_path)
            
            # Check action space compatibility
            model_action_space = temp_model.action_space
            env_action_space = env.action_space
            
            if model_action_space.n != env_action_space.n:
                self.add_log_message(f"   Action space mismatch: {model_action_space.n} vs {env_action_space.n}")
                return False
            
            # Check observation space compatibility  
            model_obs_space = temp_model.observation_space
            env_obs_space = env.observation_space
            
            # Check observation shape
            model_obs_shape = model_obs_space['observation'].shape
            env_obs_shape = env_obs_space['observation'].shape
            
            if model_obs_shape != env_obs_shape:
                self.add_log_message(f"   Observation shape mismatch: {model_obs_shape} vs {env_obs_shape}")
                return False
                
            # Check observation high values (max component ID)
            model_obs_high = model_obs_space['observation'].high[0]
            env_obs_high = env_obs_space['observation'].high[0]
            
            if model_obs_high != env_obs_high:
                self.add_log_message(f"   Max component ID mismatch: {model_obs_high} vs {env_obs_high}")
                return False
            
            # Check action mask shape
            model_mask_shape = model_obs_space['action_mask'].shape
            env_mask_shape = env_obs_space['action_mask'].shape
            
            if model_mask_shape != env_mask_shape:
                self.add_log_message(f"   Action mask mismatch: {model_mask_shape} vs {env_mask_shape}")
                return False
                
            self.add_log_message("✓ Model compatible with current environment")
            return True
            
        except Exception as e:
            self.add_log_message(f"   Compatibility check failed: {str(e)}")
            return False
            
    def _load_model_with_compatibility_check(self, model_info: Dict[str, Any], env) -> Any:
        """Load a model with compatibility checking."""
        from stable_baselines3 import PPO
        
        model_path = model_info["path"]
        
        # Check compatibility first
        if not self._check_model_compatibility(model_path, env):
            # Model incompatible - create new model
            self.add_log_message(f"⚠️  Model incompatible with schematic (different component count)")
            self.add_log_message(f"🔄 Creating new model for {len(env.components)} components...")
            
            model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")
            self.add_log_message(f"✓ New model created for schematic")
            
            # Save the new model
            new_model_path = f"./models/schematic_model_{len(env.components)}_components.zip"
            os.makedirs("./models", exist_ok=True)
            model.save(new_model_path)
            self.add_log_message(f"💾 New model saved: {new_model_path}")
            
        else:
            # Model compatible - load normally
            if model_info["type"] == "zipped":
                model = PPO.load(model_path, env=env)
            else:
                model = PPO.load(model_path, env=env)
                
            self.add_log_message(f"✓ Model loaded successfully: {model_info['name']}")
            
        return model
        
    def _create_new_model(self, env) -> Any:
        """Create a new PPO model with default parameters."""
        from stable_baselines3 import PPO
        
        tensorboard_log_dir = "./ppo_tensorboard/"
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=self.training_params["learning_rate"],
            n_steps=self.training_params["n_steps"],
            batch_size=self.training_params["batch_size"],
            gamma=self.training_params["gamma"],
            gae_lambda=self.training_params["gae_lambda"],
            ent_coef=self.training_params["ent_coef"]
        )
        self.add_log_message("✓ Created new PPO model")
        return model
            
    def set_schematic_data(self, schematic_data: Optional[Dict[str, Any]]):
        """Set the schematic data for layout generation."""
        self.schematic_data = schematic_data
        if schematic_data:
            if schematic_data.get("format") == "rl_compatible":
                # RL-compatible format with rich information
                stats = schematic_data.get("statistics", {})
                components = stats.get("total_components", 0)
                nets = stats.get("total_nets", 0)
                connections = stats.get("total_connections", 0)
                comp_types = stats.get("component_types", [])
                
                self.add_log_message(f"📋 Schematic data loaded (RL-compatible format)")
                self.add_log_message(f"   • Components: {components}")
                self.add_log_message(f"   • Nets: {nets}")
                self.add_log_message(f"   • Connections: {connections}")
                self.add_log_message(f"   • Types: {', '.join(comp_types)}")
                
                # Log constraints if available
                constraints = schematic_data.get("schematic_constraints", {})
                if constraints.get("symmetry_requirements", {}).get("symmetric_pairs"):
                    pairs = len(constraints["symmetry_requirements"]["symmetric_pairs"])
                    self.add_log_message(f"   • Symmetry pairs: {pairs}")
                    
            else:
                # Basic format
                components = len(schematic_data.get("components", []))
                self.add_log_message(f"Schematic data loaded: {components} components")
        else:
            self.add_log_message("No schematic data provided - using default configuration")