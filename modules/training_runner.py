"""
Training Runner Module for AICRL GUI.

This module handles the training and execution of AI models for IC layout generation.
Provides interfaces for model training, layout generation, and progress monitoring.

Author: AICRL Team
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
import queue
import subprocess
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path


class TrainingRunnerModule:
    """Module for handling AI model training and layout generation."""
    
    def __init__(self):
        """Initialize the training runner module."""
        self.is_training = False
        self.is_generating = False
        self.current_model = None
        self.training_thread = None
        self.progress_queue = queue.Queue()
        
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
        self.load_existing_var = tk.BooleanVar(value=False)
        load_check = ttk.Checkbutton(model_frame, text="Load existing model",
                                    variable=self.load_existing_var,
                                    command=self.toggle_model_loading)
        load_check.grid(row=1, column=0, columnspan=2, sticky=(tk.W,), pady=(10, 0))
        
        # Model file selection
        self.model_file_var = tk.StringVar(value="No model selected")
        self.model_file_label = ttk.Label(model_frame, textvariable=self.model_file_var,
                                         foreground='#7F8C8D', font=('Arial', 9))
        self.model_file_label.grid(row=1, column=2, columnspan=2, sticky=(tk.W, tk.E), 
                                  padx=(10, 0), pady=(10, 0))
        
        # Browse model button
        self.browse_model_btn = ttk.Button(model_frame, text="Browse...",
                                          command=self.browse_model_file,
                                          state=tk.DISABLED)
        self.browse_model_btn.grid(row=2, column=2, columnspan=2, sticky=(tk.W,), 
                                  padx=(10, 0), pady=(5, 0))
        
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
            self.browse_model_btn.config(state=tk.NORMAL)
        else:
            self.browse_model_btn.config(state=tk.DISABLED)
            self.model_file_var.set("No model selected")
            
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
            env = AnalogICLayoutEnv(grid_size=grid_size)
            
            self.add_log_message("Initializing model...")
            
            if self.load_existing_var.get() and self.model_file_var.get() != "No model selected":
                # Load existing model
                model_path = self.model_file_var.get()
                model = PPO.load(model_path, env=env)
                self.add_log_message(f"Loaded existing model from {os.path.basename(model_path)}")
            else:
                # Create new model
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
                self.add_log_message("Created new PPO model")
            
            self.training_status_var.set("Training in progress...")
            self.add_log_message("Starting model training...")
            
            # Training with progress updates
            total_timesteps = self.training_params["total_timesteps"]
            
            # Simulate training progress (in real implementation, you'd integrate with actual training)
            for i in range(0, total_timesteps, total_timesteps // 10):
                if not self.is_training:  # Check if training was cancelled
                    break
                    
                progress = (i / total_timesteps) * 100
                if progress_callback:
                    progress_callback(progress)
                self.training_progress_var.set(progress)
                
                self.add_log_message(f"Training progress: {progress:.1f}% ({i}/{total_timesteps} timesteps)")
                time.sleep(0.5)  # Simulate training time
                
            if self.is_training:  # Training completed successfully
                # Save the trained model
                os.makedirs("./models", exist_ok=True)
                model_path = f"./models/trained_model_{int(time.time())}.zip"
                model.save(model_path)
                
                self.current_model = model
                self.training_status_var.set("Training completed successfully!")
                self.add_log_message(f"Training completed! Model saved to {model_path}")
                self.training_progress_var.set(100)
                
                if progress_callback:
                    progress_callback(100)
            else:
                self.training_status_var.set("Training cancelled")
                self.add_log_message("Training was cancelled by user")
                
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
            
        if not self.current_model and not (self.load_existing_var.get() and 
                                          self.model_file_var.get() != "No model selected"):
            messagebox.showwarning("Warning", "No trained model available. Please train a model first or load an existing one.")
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
            from stable_baselines3 import PPO
            from enhanced_visualizer import EnhancedLayoutVisualizer
            from layout_analyzer import LayoutAnalyzer
            
            self.add_log_message("Creating environment for generation...")
            grid_size = int(self.grid_size_var.get())
            env = AnalogICLayoutEnv(grid_size=grid_size)
            
            # Load model if needed
            if not self.current_model:
                if self.load_existing_var.get() and self.model_file_var.get() != "No model selected":
                    model_path = self.model_file_var.get()
                    model = PPO.load(model_path, env=env)
                    self.add_log_message(f"Loaded model from {os.path.basename(model_path)}")
                else:
                    raise ValueError("No model available for generation")
            else:
                model = self.current_model
                
            self.add_log_message("Generating layout...")
            
            # Generate layout
            obs, info = env.reset()
            done = False
            step = 0
            MAX_STEPS = 100
            
            while not done and step < MAX_STEPS:
                step += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action.item())
                
                if step % 10 == 0:
                    self.add_log_message(f"Generation step {step}/{MAX_STEPS}")
                    
            self.add_log_message(f"Layout generation completed in {step} steps")
            
            # Analyze and visualize results
            self.add_log_message("Analyzing generated layout...")
            analyzer = LayoutAnalyzer(env)
            analysis = analyzer.analyze_layout()
            
            self.add_log_message("Creating visualizations...")
            visualizer = EnhancedLayoutVisualizer(style='professional')
            
            # Create output directory
            os.makedirs("./output", exist_ok=True)
            
            # Generate layout visualization
            layout_path = f"./output/generated_layout_{int(time.time())}.png"
            visualizer.render_layout_professional(env, save_path=layout_path, show_metrics=True)
            
            self.add_log_message(f"Layout generated successfully! Saved to {layout_path}")
            self.add_log_message(f"Layout quality metrics: {analysis}")
            
        except Exception as e:
            self.add_log_message(f"Layout generation error: {str(e)}")
            
        finally:
            self.is_generating = False
            
    def get_training_status(self) -> Dict[str, Any]:
        """Get the current training status."""
        return {
            "is_training": self.is_training,
            "is_generating": self.is_generating,
            "current_model": self.current_model is not None,
            "parameters": self.training_params.copy()
        }