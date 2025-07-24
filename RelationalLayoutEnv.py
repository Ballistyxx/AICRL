"""
Relational Layout Environment for Analog IC Design.

This environment uses relational placement actions instead of absolute coordinates,
enabling better generalization across different circuit sizes and topologies.

Author: AICRL Team
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Any

try:
    from config import DEFAULT_COMPONENTS, DEFAULT_CONNECTIONS, REWARD_WEIGHTS
except ImportError:
    # Fallback if config file is not available
    DEFAULT_COMPONENTS = [
        {"name": "nfet_d1", "id": 1, "width": 2, "height": 2, "color": "red", "can_overlap": False, "type": "nfet", "match_group": "diff_pair"},
        {"name": "nfet_d2", "id": 2, "width": 2, "height": 2, "color": "red", "can_overlap": False, "type": "nfet", "match_group": "diff_pair"},
        {"name": "pfet_m1", "id": 3, "width": 2, "height": 2, "color": "blue", "can_overlap": False, "type": "pfet", "match_group": "current_mirror"},
        {"name": "pfet_m2", "id": 4, "width": 2, "height": 2, "color": "blue", "can_overlap": False, "type": "pfet", "match_group": "current_mirror"},
        {"name": "cap", "id": 5, "width": 1, "height": 1, "color": "green", "can_overlap": True, "type": "cap", "match_group": None},
    ]
    DEFAULT_CONNECTIONS = [(1, 3), (2, 4)]
    REWARD_WEIGHTS = {
        "component_completion": 10.0,
        "valid_placement": 1.0,
        "symmetry": 20.0,
        "compactness": 5.0,
        "connectivity": 8.0,
        "invalid_action": -50.0,
        "collision": -20.0
    }


class SpatialRelation(IntEnum):
    """Enum for spatial relations between components."""
    LEFT_OF = 0
    RIGHT_OF = 1
    ABOVE = 2
    BELOW = 3
    MIRRORED_HORIZONTALLY = 4
    MIRRORED_VERTICALLY = 5
    ADJACENT = 6


class Orientation(IntEnum):
    """Enum for component orientations."""
    DEGREE_0 = 0
    DEGREE_90 = 1
    DEGREE_180 = 2
    DEGREE_270 = 3


class RelationalLayoutEnv(gym.Env):
    """
    Relational Layout Environment for Analog IC Design.
    
    Uses relational placement actions: (target_component_id, spatial_relation, orientation)
    instead of absolute (x, y) coordinates for better generalization.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=20, schematic_data=None, difficulty_level=1):
        super().__init__()
        
        self.grid_size = grid_size
        self.schematic_data = schematic_data
        self.difficulty_level = difficulty_level
        
        # Component metadata and connections
        if schematic_data:
            self.components = self._generate_components_from_schematic(schematic_data, difficulty_level)
            self.connections = self._define_connections_from_schematic(schematic_data)
        else:
            self.components = self._generate_components(difficulty_level)
            self.connections = self._define_connections()
            
        self.num_components = len(self.components)
        
        # Relational Action Space: (target_component_id, spatial_relation, orientation)
        self.action_space = spaces.MultiDiscrete([
            self.num_components,  # target_component_id (0 to num_components-1)
            len(SpatialRelation),  # spatial_relation (0 to 6)
            len(Orientation)       # orientation (0 to 3)
        ])
        
        # Enhanced observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=self.num_components,
                shape=(grid_size, grid_size), dtype=np.int8
            ),
            "placed_components": spaces.Box(
                low=0, high=1,
                shape=(self.num_components,), dtype=np.int8
            ),
            "component_positions": spaces.Box(
                low=-1, high=grid_size,  # -1 indicates not placed
                shape=(self.num_components, 2), dtype=np.int8
            ),
            "next_component_id": spaces.Discrete(self.num_components),
            "action_mask": spaces.Box(
                low=0, high=1, 
                shape=(self.num_components * len(SpatialRelation) * len(Orientation),), 
                dtype=np.int8
            )
        })
        
        self.reset()
        
    def _generate_components(self, difficulty_level=1):
        """Generate components based on difficulty level."""
        base_components = DEFAULT_COMPONENTS.copy()
        
        if difficulty_level == 1:
            # Easy: 3-4 components
            return base_components[:4]
        elif difficulty_level == 2:
            # Medium: 5-8 components  
            return base_components[:min(8, len(base_components))]
        else:
            # Hard: all components
            return base_components
            
    def _generate_components_from_schematic(self, schematic_data, difficulty_level=1):
        """Generate components from schematic data with difficulty scaling."""
        if schematic_data.get("format") == "rl_compatible":
            components = schematic_data.get("components", [])
            
            # Apply difficulty scaling
            if difficulty_level == 1:
                components = components[:min(4, len(components))]
            elif difficulty_level == 2:
                components = components[:min(8, len(components))]
            
            # Ensure components have all required fields
            processed_components = []
            for i, comp in enumerate(components):
                processed_comp = {
                    "name": comp.get("name", f"comp_{i+1}"),
                    "id": i + 1,  # Reindex for consistency
                    "width": comp.get("width", 2),
                    "height": comp.get("height", 2),
                    "color": comp.get("color", "gray"),
                    "can_overlap": comp.get("can_overlap", False),
                    "type": comp.get("type", "unknown"),
                    "match_group": comp.get("match_group"),
                    "nets": comp.get("nets", []),
                    "parameters": comp.get("parameters", {}),
                    "spice_model": comp.get("spice_model", "")
                }
                processed_components.append(processed_comp)
                
            return processed_components
        else:
            return self._generate_components(difficulty_level)
            
    def _define_connections(self):
        """Define connections based on components."""
        if len(self.components) <= 4:
            return [(1, 2), (2, 3)]
        else:
            # More complex connections for larger circuits
            connections = []
            for i in range(1, len(self.components)):
                connections.append((i, i+1))
            return connections
            
    def _define_connections_from_schematic(self, schematic_data):
        """Generate connections from schematic data."""
        if schematic_data.get("format") == "rl_compatible":
            connections = schematic_data.get("connections", [])
            # Adjust connection IDs to match reindexed components
            adjusted_connections = []
            component_id_map = {comp.get("id", i+1): i+1 for i, comp in enumerate(schematic_data.get("components", []))}
            
            for conn in connections:
                if len(conn) == 2:
                    old_id1, old_id2 = conn
                    new_id1 = component_id_map.get(old_id1, old_id1)
                    new_id2 = component_id_map.get(old_id2, old_id2)
                    if new_id1 <= len(self.components) and new_id2 <= len(self.components):
                        adjusted_connections.append((new_id1, new_id2))
                        
            return adjusted_connections
        else:
            return self._define_connections()
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Initialize grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.placements = {}  # component_id -> (x, y, orientation)
        self.placed_components = set()
        self.placement_order = []  # Track order of placement
        
        # Initialize with first component placed at center
        if self.components:
            first_comp = self.components[0]
            center_x = self.grid_size // 2
            center_y = self.grid_size // 2
            
            # Place first component at center
            self._place_component_at_position(
                first_comp["id"], center_x, center_y, Orientation.DEGREE_0
            )
            
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute a relational placement action."""
        target_comp_idx, spatial_relation, orientation = action
        
        # Validate action
        if not self._is_valid_action(target_comp_idx, spatial_relation, orientation):
            return (
                self._get_obs(), 
                REWARD_WEIGHTS["invalid_action"], 
                True, 
                False, 
                {"error": "Invalid action"}
            )
        
        # Get next component to place
        next_comp = self._get_next_component()
        if next_comp is None:
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {"error": "All components already placed"}
            )
        
        # Get target component position
        target_comp_id = self.components[target_comp_idx]["id"]
        if target_comp_id not in self.placements:
            return (
                self._get_obs(),
                REWARD_WEIGHTS["invalid_action"],
                True,
                False,
                {"error": "Target component not yet placed"}
            )
        
        # Calculate placement position based on relation
        new_x, new_y = self._calculate_relational_position(
            target_comp_id, next_comp, spatial_relation
        )
        
        # Validate placement
        if not self._is_valid_placement(next_comp, new_x, new_y):
            return (
                self._get_obs(),
                REWARD_WEIGHTS["collision"],
                False,
                False,
                {"error": "Invalid placement - collision or out of bounds"}
            )
        
        # Place component
        success = self._place_component_at_position(
            next_comp["id"], new_x, new_y, Orientation(orientation)
        )
        
        if not success:
            return (
                self._get_obs(),
                REWARD_WEIGHTS["collision"],
                False,
                False,
                {"error": "Placement failed"}
            )
        
        # Calculate reward
        reward = self._calculate_relational_reward(
            next_comp, target_comp_id, spatial_relation
        )
        
        # Check if done
        done = len(self.placed_components) == self.num_components
        if done:
            reward += self._calculate_completion_reward()
        
        # Log action for diagnostics
        self._log_action(next_comp, target_comp_id, spatial_relation, orientation)
        
        return self._get_obs(), reward, done, False, {}
    
    def _is_valid_action(self, target_comp_idx, spatial_relation, orientation):
        """Check if the action is valid."""
        # Check bounds
        if (target_comp_idx >= self.num_components or 
            spatial_relation >= len(SpatialRelation) or
            orientation >= len(Orientation)):
            return False
        
        # Check if target component is placed
        target_comp_id = self.components[target_comp_idx]["id"]
        if target_comp_id not in self.placed_components:
            return False
            
        # Check if we have a component to place
        next_comp = self._get_next_component()
        return next_comp is not None
    
    def _get_next_component(self):
        """Get the next component to place."""
        for comp in self.components:
            if comp["id"] not in self.placed_components:
                return comp
        return None
    
    def _calculate_relational_position(self, target_comp_id, new_comp, spatial_relation):
        """Calculate position based on spatial relation to target component."""
        target_x, target_y, _ = self.placements[target_comp_id]
        target_comp = next(c for c in self.components if c["id"] == target_comp_id)
        
        # Base spacing (minimum distance between components)
        spacing = 1
        
        relation = SpatialRelation(spatial_relation)
        
        if relation == SpatialRelation.LEFT_OF:
            new_x = target_x - new_comp["width"] - spacing
            new_y = target_y
        elif relation == SpatialRelation.RIGHT_OF:
            new_x = target_x + target_comp["width"] + spacing
            new_y = target_y
        elif relation == SpatialRelation.ABOVE:
            new_x = target_x
            new_y = target_y - new_comp["height"] - spacing
        elif relation == SpatialRelation.BELOW:
            new_x = target_x
            new_y = target_y + target_comp["height"] + spacing
        elif relation == SpatialRelation.MIRRORED_HORIZONTALLY:
            # Mirror horizontally with some spacing
            center_y = target_y + target_comp["height"] // 2
            new_x = target_x + target_comp["width"] + spacing
            new_y = center_y - new_comp["height"] // 2
        elif relation == SpatialRelation.MIRRORED_VERTICALLY:
            # Mirror vertically with some spacing
            center_x = target_x + target_comp["width"] // 2
            new_x = center_x - new_comp["width"] // 2
            new_y = target_y + target_comp["height"] + spacing
        elif relation == SpatialRelation.ADJACENT:
            # Place adjacent (right side by default)
            new_x = target_x + target_comp["width"]
            new_y = target_y
        else:
            # Fallback
            new_x = target_x + 1
            new_y = target_y + 1
            
        return int(new_x), int(new_y)
    
    def _is_valid_placement(self, component, x, y):
        """Check if placement is valid (no collision, within bounds)."""
        w, h = component["width"], component["height"]
        
        # Check bounds
        if x < 0 or y < 0 or x + w > self.grid_size or y + h > self.grid_size:
            return False
        
        # Check overlap for non-overlapping components
        if not component["can_overlap"]:
            region = self.grid[x:x+w, y:y+h]
            if np.any(region != 0):
                return False
                
        return True
    
    def _place_component_at_position(self, comp_id, x, y, orientation):
        """Place component at specific position with orientation."""
        component = next(c for c in self.components if c["id"] == comp_id)
        w, h = component["width"], component["height"]
        
        # Apply orientation (simplified - just store the orientation)
        if orientation in [Orientation.DEGREE_90, Orientation.DEGREE_270]:
            w, h = h, w  # Swap dimensions for 90/270 degree rotations
            
        if not self._is_valid_placement(component, x, y):
            return False
        
        # Place on grid
        self.grid[x:x+w, y:y+h] = comp_id
        self.placements[comp_id] = (x, y, orientation)
        self.placed_components.add(comp_id)
        self.placement_order.append(comp_id)
        
        return True
    
    def _calculate_relational_reward(self, placed_comp, target_comp_id, spatial_relation):
        """Calculate reward for relational placement."""
        reward = REWARD_WEIGHTS["valid_placement"]
        
        # Reward for component completion
        completion_ratio = len(self.placed_components) / self.num_components
        reward += REWARD_WEIGHTS["component_completion"] * completion_ratio
        
        # Symmetry reward
        symmetry_score = self._evaluate_symmetry()
        reward += REWARD_WEIGHTS["symmetry"] * symmetry_score
        
        # Connectivity reward
        connectivity_score = self._evaluate_connectivity()
        reward += REWARD_WEIGHTS["connectivity"] * connectivity_score
        
        # Compactness reward
        compactness_score = self._evaluate_compactness()
        reward += REWARD_WEIGHTS["compactness"] * compactness_score
        
        return reward
    
    def _calculate_completion_reward(self):
        """Calculate final reward when all components are placed."""
        final_reward = 0.0
        
        # Large completion bonus
        final_reward += REWARD_WEIGHTS["component_completion"] * 5.0
        
        # Final layout quality assessment
        final_symmetry = self._evaluate_symmetry()
        final_connectivity = self._evaluate_connectivity()
        final_compactness = self._evaluate_compactness()
        
        final_reward += (
            REWARD_WEIGHTS["symmetry"] * final_symmetry * 2.0 +
            REWARD_WEIGHTS["connectivity"] * final_connectivity * 2.0 +
            REWARD_WEIGHTS["compactness"] * final_compactness * 2.0
        )
        
        return final_reward
    
    def _evaluate_symmetry(self):
        """Evaluate symmetry of current layout."""
        if len(self.placed_components) < 2:
            return 0.0
        
        # Find components with same match_group
        match_groups = {}
        for comp in self.components:
            if comp["id"] in self.placed_components and comp["match_group"]:
                group = comp["match_group"]
                if group not in match_groups:
                    match_groups[group] = []
                match_groups[group].append(comp["id"])
        
        symmetry_score = 0.0
        total_groups = 0
        
        for group, comp_ids in match_groups.items():
            if len(comp_ids) >= 2:
                total_groups += 1
                # Calculate symmetry for this group
                positions = [self.placements[cid][:2] for cid in comp_ids]
                
                # Simple symmetry check: components should be roughly symmetric
                if len(positions) == 2:
                    pos1, pos2 = positions
                    # Check for horizontal or vertical symmetry
                    center_x = self.grid_size // 2
                    center_y = self.grid_size // 2
                    
                    # Horizontal symmetry check
                    if abs((pos1[0] - center_x) + (pos2[0] - center_x)) < 2:
                        symmetry_score += 0.5
                    
                    # Vertical symmetry check
                    if abs((pos1[1] - center_y) + (pos2[1] - center_y)) < 2:
                        symmetry_score += 0.5
        
        return symmetry_score / max(total_groups, 1)
    
    def _evaluate_connectivity(self):
        """Evaluate how well connected components are placed."""
        if not self.connections:
            return 1.0  # No connections to evaluate
        
        total_distance = 0.0
        valid_connections = 0
        
        for comp1_id, comp2_id in self.connections:
            if comp1_id in self.placed_components and comp2_id in self.placed_components:
                pos1 = self.placements[comp1_id][:2]
                pos2 = self.placements[comp2_id][:2]
                
                # Manhattan distance
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                total_distance += distance
                valid_connections += 1
        
        if valid_connections == 0:
            return 0.5
        
        avg_distance = total_distance / valid_connections
        # Normalize and invert (closer is better)
        max_possible_distance = self.grid_size * 2
        connectivity_score = 1.0 - (avg_distance / max_possible_distance)
        
        return max(0.0, min(1.0, connectivity_score))
    
    def _evaluate_compactness(self):
        """Evaluate layout compactness."""
        if len(self.placed_components) < 2:
            return 1.0
        
        # Calculate bounding box of all placed components
        min_x = min_y = self.grid_size
        max_x = max_y = 0
        
        for comp_id in self.placed_components:
            comp = next(c for c in self.components if c["id"] == comp_id)
            x, y, _ = self.placements[comp_id]
            
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + comp["width"])
            max_y = max(max_y, y + comp["height"])
        
        # Calculate area efficiency
        bounding_area = (max_x - min_x) * (max_y - min_y)
        total_component_area = sum(
            comp["width"] * comp["height"] 
            for comp in self.components 
            if comp["id"] in self.placed_components
        )
        
        if bounding_area == 0:
            return 1.0
        
        area_efficiency = total_component_area / bounding_area
        return min(1.0, area_efficiency)
    
    def _get_obs(self):
        """Get current observation."""
        # Component placement status
        placed_mask = np.zeros(self.num_components, dtype=np.int8)
        for i, comp in enumerate(self.components):
            if comp["id"] in self.placed_components:
                placed_mask[i] = 1
        
        # Component positions (-1 for not placed)
        positions = np.full((self.num_components, 2), -1, dtype=np.int8)
        for i, comp in enumerate(self.components):
            if comp["id"] in self.placements:
                x, y, _ = self.placements[comp["id"]]
                positions[i] = [x, y]
        
        # Next component to place
        next_comp = self._get_next_component()
        next_comp_id = next_comp["id"] - 1 if next_comp else 0  # 0-indexed
        
        # Action mask
        action_mask = self._get_action_mask()
        
        return {
            "grid": self.grid.copy(),
            "placed_components": placed_mask,
            "component_positions": positions,
            "next_component_id": next_comp_id,
            "action_mask": action_mask
        }
    
    def _get_action_mask(self):
        """Generate action mask for valid actions."""
        mask_size = self.num_components * len(SpatialRelation) * len(Orientation)
        mask = np.zeros(mask_size, dtype=np.int8)
        
        next_comp = self._get_next_component()
        if next_comp is None:
            return mask  # No valid actions if all components placed
        
        for target_idx in range(self.num_components):
            target_comp_id = self.components[target_idx]["id"]
            
            # Only allow actions targeting already placed components
            if target_comp_id in self.placed_components:
                for relation in range(len(SpatialRelation)):
                    for orientation in range(len(Orientation)):
                        # Calculate if this action would result in valid placement
                        try:
                            new_x, new_y = self._calculate_relational_position(
                                target_comp_id, next_comp, relation
                            )
                            if self._is_valid_placement(next_comp, new_x, new_y):
                                action_idx = (
                                    target_idx * len(SpatialRelation) * len(Orientation) +
                                    relation * len(Orientation) +
                                    orientation
                                )
                                mask[action_idx] = 1
                        except:
                            continue  # Invalid action
        
        return mask
    
    def _log_action(self, placed_comp, target_comp_id, spatial_relation, orientation):
        """Log the relational action for diagnostics."""
        target_comp = next(c for c in self.components if c["id"] == target_comp_id)
        relation_name = SpatialRelation(spatial_relation).name.lower().replace('_', '-')
        orientation_deg = orientation * 90
        
        print(f"Placed {placed_comp['name']} {relation_name} {target_comp['name']}, {orientation_deg}° rotation")
    
    def render(self, mode='human'):
        """Render the current layout."""
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render layout for human viewing."""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        ax.set_xlim(-0.5, self.grid_size + 0.5)
        ax.set_ylim(-0.5, self.grid_size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw components
        for comp_id in self.placed_components:
            comp = next(c for c in self.components if c["id"] == comp_id)
            x, y, orientation = self.placements[comp_id]
            
            # Draw component rectangle
            rect = patches.Rectangle(
                (x, y), comp["width"], comp["height"],
                linewidth=2, edgecolor='black', 
                facecolor=comp["color"], alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add component label
            ax.text(x + comp["width"]/2, y + comp["height"]/2, 
                   comp["name"], ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        ax.set_title(f'Relational Layout - {len(self.placed_components)}/{self.num_components} Components')
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self):
        """Render layout as RGB array."""
        # Implementation for RGB array rendering
        fig, ax = plt.subplots(figsize=(8, 8))
        self._render_human()  # Reuse human rendering logic
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf