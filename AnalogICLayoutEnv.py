import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        "compactness": 0.1,
        "symmetry": 20.0,
        "connectivity": 0.5,
        "valid_placement": 1.0,
        "incremental_connectivity": 2.0,
        "incremental_symmetry": 1.0,
    }
    DEFAULT_COMPONENTS = [
        {"name": "nfet_d1", "id": 1, "width": 2, "height": 2, "color": "red", "can_overlap": False, "type": "nfet", "match_group": "diff_pair"},
        {"name": "nfet_d2", "id": 2, "width": 2, "height": 2, "color": "red", "can_overlap": False, "type": "nfet", "match_group": "diff_pair"},
        {"name": "pfet_m1", "id": 3, "width": 2, "height": 2, "color": "blue", "can_overlap": False, "type": "pfet", "match_group": "current_mirror"},
        {"name": "pfet_m2", "id": 4, "width": 2, "height": 2, "color": "blue", "can_overlap": False, "type": "pfet", "match_group": "current_mirror"},
        {"name": "cap", "id": 5, "width": 1, "height": 1, "color": "green", "can_overlap": True, "type": "cap", "match_group": None},
    ]
    DEFAULT_CONNECTIONS = [(1, 3), (2, 4)]
    REWARD_WEIGHTS = {"compactness": 0.1, "symmetry": 20.0, "connectivity": 0.5, "valid_placement": 1.0, "incremental_connectivity": 2.0, "incremental_symmetry": 1.0}

class AnalogICLayoutEnv(gym.Env):
    """
    A custom Gym environment for Analog IC Layout.

    The environment is a grid where the agent places components. The goal is to
    achieve a layout that is compact, symmetric, and respects connectivity constraints.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_size=20, schematic_data=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.schematic_data = schematic_data
        
        # Component metadata and connections
        if schematic_data:
            self.components = self._generate_components_from_schematic(schematic_data)
            self.connections = self._define_connections_from_schematic(schematic_data)
        else:
            self.components = self._generate_components()
            self.connections = self._define_connections()
            
        self.num_components = len(self.components)
        
        # Action: place component_id at (x, y)
        self.action_space = spaces.Discrete(self.num_components * grid_size * grid_size)

        # Define observation space: grid with cell values representing component IDs
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=0, high=self.num_components,
                shape=(grid_size, grid_size), dtype=np.int8
            ),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int8)
        })
        
        self.reset()

    def _generate_components(self):
        """Defines the components to be placed in the layout."""
        return DEFAULT_COMPONENTS.copy()

    def _define_connections(self):
        """Defines which components should be close to each other (by ID)."""
        return DEFAULT_CONNECTIONS.copy()
        
    def _generate_components_from_schematic(self, schematic_data):
        """Generate components from schematic data."""
        try:
            if schematic_data.get("format") == "rl_compatible":
                # Use the RL-compatible format directly
                components = schematic_data.get("components", [])
                # Ensure components have all required fields
                processed_components = []
                for comp in components:
                    processed_comp = {
                        "name": comp.get("name", f"comp_{comp.get('id', 'unknown')}"),
                        "id": comp.get("id", len(processed_components) + 1),
                        "width": comp.get("width", 2),
                        "height": comp.get("height", 2),
                        "color": comp.get("color", "gray"),
                        "can_overlap": comp.get("can_overlap", False),
                        "type": comp.get("type", "unknown"),
                        "match_group": comp.get("match_group"),
                        # Additional fields for constraint handling
                        "nets": comp.get("nets", []),
                        "parameters": comp.get("parameters", {}),
                        "spice_model": comp.get("spice_model", "")
                    }
                    processed_components.append(processed_comp)
                    
                return processed_components
            else:
                # Convert other formats to RL format
                return self._convert_raw_schematic_to_components(schematic_data)
                
        except Exception as e:
            print(f"Error generating components from schematic: {e}")
            # Fallback to default components
            return DEFAULT_COMPONENTS.copy()
            
    def _define_connections_from_schematic(self, schematic_data):
        """Generate connections from schematic data."""
        try:
            if schematic_data.get("format") == "rl_compatible":
                return schematic_data.get("connections", [])
            else:
                # Generate connections from raw schematic
                return self._convert_raw_schematic_to_connections(schematic_data)
                
        except Exception as e:
            print(f"Error generating connections from schematic: {e}")
            # Fallback to default connections
            return DEFAULT_CONNECTIONS.copy()
            
    def _convert_raw_schematic_to_components(self, schematic_data):
        """Convert raw schematic data to component format."""
        components = []
        component_id_counter = 1
        
        raw_components = schematic_data.get("components", [])
        for raw_comp in raw_components:
            # Basic conversion logic
            comp_type = raw_comp.get("type", "unknown")
            
            # Map component types
            if comp_type == "mosfet":
                rl_type = "nfet"  # Default, should be refined based on model
                color = "red"
                size = (2, 2)
            elif comp_type == "capacitor":
                rl_type = "cap"
                color = "green" 
                size = (1, 1)
            elif comp_type == "resistor":
                rl_type = "resistor"
                color = "orange"
                size = (1, 2)
            else:
                rl_type = "unknown"
                color = "gray"
                size = (2, 2)
                
            component = {
                "name": raw_comp.get("instance_name", f"comp_{component_id_counter}"),
                "id": component_id_counter,
                "width": size[0],
                "height": size[1],
                "color": color,
                "can_overlap": rl_type in ["cap", "resistor"],
                "type": rl_type,
                "match_group": None
            }
            
            components.append(component)
            component_id_counter += 1
            
        return components if components else DEFAULT_COMPONENTS.copy()
        
    def _convert_raw_schematic_to_connections(self, schematic_data):
        """Convert raw schematic data to connections."""
        # Basic connection generation from nets
        connections = []
        nets = schematic_data.get("nets", [])
        components = schematic_data.get("components", [])
        
        # This is a simplified approach - in practice would need more sophisticated net analysis
        if len(components) >= 2:
            # Connect adjacent components as a simple heuristic
            for i in range(len(components) - 1):
                connections.append((i + 1, i + 2))
                
        return connections if connections else DEFAULT_CONNECTIONS.copy()

    def _get_action_mask(self):
        """Generates a mask of valid actions."""
        mask = np.ones(self.action_space.n, dtype=np.int8)
        for i, comp in enumerate(self.components):
            if comp["id"] in self.placed_cids:
                start_idx = i * (self.grid_size * self.grid_size)
                end_idx = (i + 1) * (self.grid_size * self.grid_size)
                mask[start_idx:end_idx] = 0
        return mask

    def _get_obs(self):
        """Returns the current observation dictionary."""
        return {
            "observation": self.grid.copy(),
            "action_mask": self._get_action_mask()
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.placements = {}  # cid -> (x, y)
        self.placed_cids = set()
        return self._get_obs(), {}

    def step(self, action):
        """Executes one time step within the environment."""
        comp_idx = action // (self.grid_size * self.grid_size)
        
        # This should not happen with a valid policy, but as a safeguard:
        if comp_idx >= self.num_components:
            return self._get_obs(), -100, True, False, {"error": "Invalid component index"}

        comp = self.components[comp_idx]
        cid = comp["id"]

        if cid in self.placed_cids:
            # This should not happen with a correct action mask, but as a safeguard
            return self._get_obs(), -100, True, False, {"error": "Agent chose an already placed component"}

        flat_pos = action % (self.grid_size * self.grid_size)
        x, y = divmod(flat_pos, self.grid_size)
        
        w, h = comp["width"], comp["height"]

        # Check bounds
        if x + h > self.grid_size or y + w > self.grid_size:
            return self._get_obs(), -10.0, False, False, {}

        # Check overlap for non-overlapping components
        region = self.grid[x:x+h, y:y+w]
        if not comp["can_overlap"] and np.any(region != 0):
            return self._get_obs(), -10.0, False, False, {}

        # Place component
        self.grid[x:x+h, y:y+w] = cid
        self.placements[cid] = (x, y)
        self.placed_cids.add(cid)

        done = len(self.placed_cids) == self.num_components
        
        # Calculate intermediate rewards for better learning
        reward = REWARD_WEIGHTS["valid_placement"]  # Base reward for valid placement
        
        # Add incremental reward based on current layout quality
        if len(self.placed_cids) > 1:
            reward += self._calculate_incremental_reward()
        
        # Final comprehensive reward when all components are placed
        if done:
            reward += self._calculate_reward()

        return self._get_obs(), reward, done, False, {}

    def _calculate_reward(self):
        """Calculates the reward for the current layout."""
        if not self.placements:
            return 0

        reward = 0
        
        # 1. Compactness Reward (penalize large bounding box)
        min_x, max_x = self.grid_size, 0
        min_y, max_y = self.grid_size, 0
        
        for cid, (px, py) in self.placements.items():
            comp = next(c for c in self.components if c["id"] == cid)
            w, h = comp["width"], comp["height"]
            min_x = min(min_x, px)
            max_x = max(max_x, px + h)
            min_y = min(min_y, py)
            max_y = max(max_y, py + w)
            
        compactness_penalty = (max_x - min_x) * (max_y - min_y)
        reward -= REWARD_WEIGHTS["compactness"] * compactness_penalty

        # 2. Symmetry Reward
        match_groups = {}
        for comp in self.components:
            if comp["match_group"]:
                if comp["match_group"] not in match_groups:
                    match_groups[comp["match_group"]] = []
                match_groups[comp["match_group"]].append(comp["id"])

        for group, cids in match_groups.items():
            if len(cids) == 2 and all(c in self.placements for c in cids):
                cid1, cid2 = cids
                comp1 = next(c for c in self.components if c["id"] == cid1)
                comp2 = next(c for c in self.components if c["id"] == cid2)
                x1, y1 = self.placements[cid1]
                x2, y2 = self.placements[cid2]

                # Check for vertical symmetry around the center
                is_symmetric = (x1 == x2 and 
                                (y1 + comp1["width"] / 2) == (self.grid_size - (y2 + comp2["width"] / 2)))
                if is_symmetric:
                    reward += REWARD_WEIGHTS["symmetry"]

        # 3. Connectivity Reward (penalize distance between connected components)
        for cid1, cid2 in self.connections:
            if cid1 in self.placements and cid2 in self.placements:
                x1, y1 = self.placements[cid1]
                x2, y2 = self.placements[cid2]
                comp1 = next(c for c in self.components if c["id"] == cid1)
                comp2 = next(c for c in self.components if c["id"] == cid2)
                
                center1 = (x1 + comp1["height"] / 2, y1 + comp1["width"] / 2)
                center2 = (x2 + comp2["height"] / 2, y2 + comp2["width"] / 2)
                
                # Manhattan distance
                dist = abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])
                reward -= REWARD_WEIGHTS["connectivity"] * dist

        return reward

    def _calculate_incremental_reward(self):
        """Calculate incremental rewards during placement for better learning."""
        reward = 0
        
        # Connectivity reward for each placement
        for cid1, cid2 in self.connections:
            if cid1 in self.placements and cid2 in self.placements:
                x1, y1 = self.placements[cid1]
                x2, y2 = self.placements[cid2]
                comp1 = next(c for c in self.components if c["id"] == cid1)
                comp2 = next(c for c in self.components if c["id"] == cid2)
                
                center1 = (x1 + comp1["height"] / 2, y1 + comp1["width"] / 2)
                center2 = (x2 + comp2["height"] / 2, y2 + comp2["width"] / 2)
                
                # Reward close placement of connected components
                # TODO: make connected components closer based on continuous function, rather than discrete value
                dist = abs(center1[0] - center2[0]) + abs(center1[1] - center2[1])
                if dist <= 4:  # Close enough
                    reward += REWARD_WEIGHTS["incremental_connectivity"]
                elif dist <= 8:  # Moderately close
                    reward += REWARD_WEIGHTS["incremental_connectivity"] / 2
                    
        # Partial symmetry rewards
        match_groups = {}
        for comp in self.components:
            if comp["match_group"]:
                if comp["match_group"] not in match_groups:
                    match_groups[comp["match_group"]] = []
                match_groups[comp["match_group"]].append(comp["id"])

        for group, cids in match_groups.items():
            if len(cids) == 2 and all(c in self.placements for c in cids):
                cid1, cid2 = cids
                x1, y1 = self.placements[cid1]
                x2, y2 = self.placements[cid2]
                
                # Reward if they are at same x level (step toward symmetry)
                if x1 == x2:
                    reward += REWARD_WEIGHTS["incremental_symmetry"]
                    
        return reward

    def render(self, mode="human"):
        """Renders the environment."""
        if not self.placements:
            print("No components placed yet.")
            return

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(0, self.grid_size + 1, 1))
        ax.set_yticks(np.arange(0, self.grid_size + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title("Analog IC Layout")

        for cid, (x, y) in self.placements.items():
            comp = next(c for c in self.components if c["id"] == cid)
            w, h = comp["width"], comp["height"]
            color = comp["color"]
            
            # matplotlib origin is bottom-left, our grid origin is top-left.
            # Rect: (y, grid_size - x - h)
            rect = patches.Rectangle((y, self.grid_size - x - h), w, h, 
                                     linewidth=1.5, edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            ax.text(y + w/2, self.grid_size - x - h/2, comp["name"], 
                    ha='center', va='center', fontsize=8, weight='bold')

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img

