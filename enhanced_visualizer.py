import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from matplotlib.gridspec import GridSpec
import pandas as pd
from datetime import datetime
import os

class EnhancedLayoutVisualizer:
    """Professional visualization system for IC layout training and analysis."""
    
    def __init__(self, style='professional'):
        """Initialize the visualizer with professional styling."""
        self.setup_style(style)
        self.colors = {
            'nfet': '#FF6B6B',      # Coral red
            'pfet': '#4ECDC4',      # Teal
            'cap': '#45B7D1',       # Blue
            'resistor': '#FFA07A',  # Light salmon
            'inductor': '#98D8C8',  # Mint green
            'default': '#95A5A6'    # Gray
        }
        self.training_history = []
        
    def setup_style(self, style='professional'):
        """Setup matplotlib style for professional appearance."""
        if SEABORN_AVAILABLE and style == 'seaborn':
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')
        else:
            plt.style.use('default')
        
        # Professional color scheme
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#F8F9FA',
            'axes.edgecolor': '#DEE2E6',
            'axes.linewidth': 1.2,
            'axes.grid': True,
            'grid.color': '#E9ECEF',
            'grid.alpha': 0.7,
            'grid.linewidth': 0.8,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def render_layout_professional(self, env, save_path=None, show_metrics=True):
        """Render the IC layout with professional styling and comprehensive information."""
        if not env.placements:
            print("No components placed yet.")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], width_ratios=[2, 1, 1])
        
        # Main layout plot
        ax_main = fig.add_subplot(gs[0, :2])
        self._render_main_layout(ax_main, env)
        
        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 2])
        if show_metrics:
            self._render_metrics_panel(ax_metrics, env)
        
        # Component legend and info
        ax_legend = fig.add_subplot(gs[1, :])
        self._render_component_info(ax_legend, env)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            
        if save_path:
            plt.close(fig)
        else:
            plt.show()
        return fig
    
    def _render_main_layout(self, ax, env):
        """Render the main IC layout with enhanced styling."""
        ax.set_xlim(-0.5, env.grid_size + 0.5)
        ax.set_ylim(-0.5, env.grid_size + 0.5)
        
        # Enhanced grid
        ax.set_xticks(np.arange(0, env.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, env.grid_size + 1, 1))
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        # Add major grid lines
        major_ticks = np.arange(0, env.grid_size + 1, 5)
        ax.set_xticks(major_ticks, minor=False)
        ax.set_yticks(major_ticks, minor=False)
        ax.grid(True, which='major', alpha=0.6, linewidth=1.0)
        
        ax.set_aspect('equal')
        ax.set_title("Analog IC Layout - Professional View", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("X Coordinate", fontsize=12)
        ax.set_ylabel("Y Coordinate", fontsize=12)
        
        # Render components with enhanced styling
        for cid, (x, y) in env.placements.items():
            comp = next(c for c in env.components if c["id"] == cid)
            self._render_enhanced_component(ax, comp, x, y, env.grid_size)
        
        # Add connections as lines
        self._render_connections(ax, env)
        
        # Add border
        border = patches.Rectangle((0, 0), env.grid_size, env.grid_size, 
                                 linewidth=3, edgecolor='#2C3E50', 
                                 facecolor='none', alpha=0.8)
        ax.add_patch(border)
    
    def _render_enhanced_component(self, ax, comp, x, y, grid_size):
        """Render a single component with enhanced professional styling."""
        w, h = comp["width"], comp["height"]
        
        # Get color based on component type
        color = self.colors.get(comp.get("type", "default"), self.colors["default"])
        
        # Flip y-coordinate for proper display
        display_y = grid_size - x - h
        
        # Create fancy rounded rectangle
        fancy_box = FancyBboxPatch(
            (y, display_y), w, h,
            boxstyle="round,pad=0.05",
            linewidth=2,
            edgecolor='#2C3E50',
            facecolor=color,
            alpha=0.8,
            mutation_scale=0.1
        )
        ax.add_patch(fancy_box)
        
        # Add component name with better typography
        text_x = y + w/2
        text_y = display_y + h/2
        
        # Component name
        ax.text(text_x, text_y + 0.15, comp["name"], 
                ha='center', va='center', 
                fontsize=9, fontweight='bold', 
                color='white', 
                path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
        
        # Component ID
        ax.text(text_x, text_y - 0.15, f"ID: {comp['id']}", 
                ha='center', va='center', 
                fontsize=7, style='italic',
                color='white',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
        
        # Add shadow effect
        shadow = FancyBboxPatch(
            (y + 0.05, display_y - 0.05), w, h,
            boxstyle="round,pad=0.05",
            linewidth=0,
            facecolor='black',
            alpha=0.2,
            mutation_scale=0.1,
            zorder=0
        )
        ax.add_patch(shadow)
    
    def _render_connections(self, ax, env):
        """Render connection lines between components."""
        for cid1, cid2 in env.connections:
            if cid1 in env.placements and cid2 in env.placements:
                comp1 = next(c for c in env.components if c["id"] == cid1)
                comp2 = next(c for c in env.components if c["id"] == cid2)
                
                x1, y1 = env.placements[cid1]
                x2, y2 = env.placements[cid2]
                
                # Calculate centers
                center1_x = y1 + comp1["width"] / 2
                center1_y = env.grid_size - x1 - comp1["height"] / 2
                center2_x = y2 + comp2["width"] / 2
                center2_y = env.grid_size - x2 - comp2["height"] / 2
                
                # Draw connection line with arrow
                ax.annotate('', xy=(center2_x, center2_y), xytext=(center1_x, center1_y),
                           arrowprops=dict(arrowstyle='<->', color='#E74C3C', 
                                         lw=2, alpha=0.7, connectionstyle="arc3,rad=0.1"))
    
    def _render_metrics_panel(self, ax, env):
        """Render a metrics panel showing layout quality scores."""
        from layout_analyzer import LayoutAnalyzer
        
        analyzer = LayoutAnalyzer(env)
        metrics = analyzer.analyze_layout()
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Layout Metrics', ha='center', va='top', 
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        # Metrics with color coding
        metric_names = ['Compactness', 'Symmetry', 'Connectivity', 'Completion']
        metric_values = [
            metrics.get('compactness_score', 0),
            metrics.get('symmetry_score', 0),
            metrics.get('connectivity_score', 0),
            metrics.get('completion_rate', 0) * 100
        ]
        
        colors = ['#27AE60', '#3498DB', '#9B59B6', '#F39C12']
        
        y_pos = 0.8
        for name, value, color in zip(metric_names, metric_values, colors):
            # Score bar
            bar_width = value / 100 * 0.6
            ax.barh(y_pos, bar_width, height=0.08, left=0.35, 
                   color=color, alpha=0.7, edgecolor='white', linewidth=1)
            
            # Labels
            ax.text(0.02, y_pos, name, ha='left', va='center', 
                   fontsize=10, fontweight='bold')
            ax.text(0.98, y_pos, f'{value:.1f}%', ha='right', va='center', 
                   fontsize=10, fontweight='bold')
            
            y_pos -= 0.15
        
        # Overall grade
        overall = np.mean(metric_values[:3]) * (metrics.get('completion_rate', 0))
        grade_color = '#27AE60' if overall >= 80 else '#F39C12' if overall >= 60 else '#E74C3C'
        grade = 'A' if overall >= 80 else 'B' if overall >= 60 else 'C' if overall >= 40 else 'D'
        
        ax.text(0.5, 0.15, f'Overall Grade: {grade}', ha='center', va='center',
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=grade_color, alpha=0.8))
        ax.text(0.5, 0.05, f'Score: {overall:.1f}/100', ha='center', va='center',
                fontsize=10, style='italic')
    
    def _render_component_info(self, ax, env):
        """Render component information and legend."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Title
        ax.text(0.02, 0.9, 'Component Information', fontsize=12, fontweight='bold')
        
        # Component table
        x_positions = [0.02, 0.25, 0.4, 0.55, 0.7, 0.85]
        headers = ['Name', 'Type', 'Size', 'Position', 'Status', 'Group']
        
        # Headers
        for i, header in enumerate(headers):
            ax.text(x_positions[i], 0.8, header, fontsize=9, fontweight='bold')
        
        # Component data
        y_pos = 0.7
        for comp in env.components:
            color = self.colors.get(comp.get("type", "default"), self.colors["default"])
            
            # Color indicator
            rect = patches.Rectangle((x_positions[0]-0.01, y_pos-0.03), 0.015, 0.06, 
                                   facecolor=color, alpha=0.8)
            ax.add_patch(rect)
            
            # Component info
            ax.text(x_positions[0], y_pos, comp["name"], fontsize=8)
            ax.text(x_positions[1], y_pos, comp.get("type", "N/A"), fontsize=8)
            ax.text(x_positions[2], y_pos, f"{comp['width']}×{comp['height']}", fontsize=8)
            
            if comp["id"] in env.placements:
                x, y = env.placements[comp["id"]]
                ax.text(x_positions[3], y_pos, f"({x},{y})", fontsize=8)
                ax.text(x_positions[4], y_pos, "Placed", fontsize=8, color='green')
            else:
                ax.text(x_positions[3], y_pos, "N/A", fontsize=8)
                ax.text(x_positions[4], y_pos, "Pending", fontsize=8, color='red')
            
            ax.text(x_positions[5], y_pos, comp.get("match_group", "None"), fontsize=8)
            
            y_pos -= 0.08
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.98, 0.02, f"Generated: {timestamp}", ha='right', va='bottom',
                fontsize=8, style='italic', alpha=0.7)
    
    def plot_training_progress(self, tensorboard_log_dir=None, save_path=None):
        """Create professional training progress visualization."""
        if tensorboard_log_dir and os.path.exists(tensorboard_log_dir):
            # Try to read tensorboard logs
            self._load_tensorboard_data(tensorboard_log_dir)
        
        if not self.training_history:
            print("No training data available for visualization.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Reward progression
        steps = [d['step'] for d in self.training_history if 'reward' in d]
        rewards = [d['reward'] for d in self.training_history if 'reward' in d]
        
        ax1.plot(steps, rewards, linewidth=2, color='#3498DB', alpha=0.8)
        ax1.fill_between(steps, rewards, alpha=0.3, color='#3498DB')
        ax1.set_title('Reward Progression', fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Episode Reward')
        ax1.grid(True, alpha=0.3)
        
        # Add moving average
        if len(rewards) > 10:
            ma_window = min(100, len(rewards) // 10)
            ma_rewards = pd.Series(rewards).rolling(window=ma_window).mean().values
            ax1.plot(steps, ma_rewards, linewidth=3, color='#E74C3C', 
                    label=f'Moving Average ({ma_window})')
            ax1.legend()
        
        # Loss progression (if available)
        if any('loss' in d for d in self.training_history):
            losses = [d.get('loss', 0) for d in self.training_history]
            ax2.plot(steps, losses, linewidth=2, color='#E74C3C')
            ax2.set_title('Training Loss', fontweight='bold')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Loss data not available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Training Loss', fontweight='bold')
        
        # Performance metrics
        if len(self.training_history) > 0:
            recent_rewards = rewards[-min(100, len(rewards)):]
            
            ax3.hist(recent_rewards, bins=20, alpha=0.7, color='#27AE60', edgecolor='black')
            ax3.axvline(np.mean(recent_rewards), color='#E74C3C', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
            ax3.set_title('Recent Reward Distribution', fontweight='bold')
            ax3.set_xlabel('Episode Reward')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Training statistics
        ax4.axis('off')
        improvement = ((rewards[-1] / rewards[0] - 1) * 100) if len(rewards) > 1 else 0
        best_reward = max(rewards) if rewards else 0
        avg_reward = np.mean(rewards) if rewards else 0
        final_reward = rewards[-1] if rewards else 0
        
        stats_text = f"""Training Statistics:
• Total Episodes: {len(self.training_history)}
• Best Reward: {best_reward:.2f}
• Average Reward: {avg_reward:.2f}
• Final Reward: {final_reward:.2f}
• Improvement: {improvement:.1f}%"""
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
        return fig
    
    def _load_tensorboard_data(self, log_dir):
        """Load training data from tensorboard logs."""
        try:
            import tensorflow as tf
            from tensorflow.python.summary.summary_iterator import summary_iterator
            
            # Find all event files in subdirectories (PPO creates subdirectories like PPO_1/, PPO_2/, etc.)
            all_event_files = []
            
            if os.path.exists(log_dir):
                # Check if there are subdirectories with event files
                for item in os.listdir(log_dir):
                    item_path = os.path.join(log_dir, item)
                    if os.path.isdir(item_path):
                        # Look for event files in subdirectory
                        try:
                            subdir_files = os.listdir(item_path)
                            for f in subdir_files:
                                if f.startswith('events.out.tfevents'):
                                    file_path = os.path.join(item_path, f)
                                    # Get modification time for sorting
                                    mtime = os.path.getmtime(file_path)
                                    all_event_files.append((mtime, file_path))
                        except (OSError, PermissionError):
                            continue
                    else:
                        # Also check for event files directly in log_dir
                        if item.startswith('events.out.tfevents'):
                            file_path = os.path.join(log_dir, item)
                            mtime = os.path.getmtime(file_path)
                            all_event_files.append((mtime, file_path))
            
            if not all_event_files:
                print(f"No tensorboard event files found in {log_dir}")
                return
            
            # Sort by modification time and get the most recent
            all_event_files.sort(key=lambda x: x[0], reverse=True)
            latest_file_path = all_event_files[0][1]
            
            print(f"Loading tensorboard data from: {latest_file_path}")
            
            self.training_history = []
            for summary in summary_iterator(latest_file_path):
                for value in summary.summary.value:
                    # Look for various reward metrics that PPO might log
                    if value.tag in ['rollout/ep_rew_mean', 'rollout/ep_rew_std', 'episode_reward', 'mean_reward']:
                        self.training_history.append({
                            'step': summary.step,
                            'reward': value.simple_value,
                            'tag': value.tag
                        })
                    # Also capture loss data if available
                    elif value.tag in ['train/loss', 'loss', 'policy_loss', 'value_loss']:
                        # Find existing entry for this step or create new one
                        existing_entry = None
                        for entry in self.training_history:
                            if entry['step'] == summary.step:
                                existing_entry = entry
                                break
                        
                        if existing_entry:
                            existing_entry['loss'] = value.simple_value
                        else:
                            self.training_history.append({
                                'step': summary.step,
                                'loss': value.simple_value,
                                'tag': value.tag
                            })
            
            # Sort by step
            self.training_history.sort(key=lambda x: x['step'])
            
            print(f"Loaded {len(self.training_history)} training data points")
            if self.training_history:
                print(f"Step range: {self.training_history[0]['step']} to {self.training_history[-1]['step']}")
                
        except ImportError:
            print("TensorFlow not available for reading tensorboard logs.")
            print("Install with: pip install tensorflow")
            print("Continuing without tensorboard data...")
        except Exception as e:
            print(f"Error reading tensorboard logs: {e}")
            print("Continuing without tensorboard data...")
    
    def create_training_report(self, env, save_dir="./reports"):
        """Generate a comprehensive training report with multiple visualizations."""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Layout visualization
        layout_path = os.path.join(save_dir, f"layout_analysis_{timestamp}.png")
        self.render_layout_professional(env, save_path=layout_path)
        
        # Training progress
        progress_path = os.path.join(save_dir, f"training_progress_{timestamp}.png")
        self.plot_training_progress(save_path=progress_path)
        
        print(f"Training report generated in: {save_dir}")
        print(f"- Layout analysis: {layout_path}")
        print(f"- Training progress: {progress_path}")