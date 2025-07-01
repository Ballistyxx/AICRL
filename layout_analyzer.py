import numpy as np
import matplotlib.pyplot as plt
from AnalogICLayoutEnv import AnalogICLayoutEnv

class LayoutAnalyzer:
    """Analyzes and evaluates IC layout quality."""
    
    def __init__(self, env):
        self.env = env
        
    def analyze_layout(self):
        """Comprehensive layout analysis."""
        if not self.env.placements:
            print("No components placed to analyze.")
            return {}
            
        metrics = {
            'total_components': len(self.env.placed_cids),
            'expected_components': self.env.num_components,
            'completion_rate': len(self.env.placed_cids) / self.env.num_components,
            'compactness_score': self._calculate_compactness(),
            'symmetry_score': self._calculate_symmetry(),
            'connectivity_score': self._calculate_connectivity(),
            'area_efficiency': self._calculate_area_efficiency()
        }
        
        self._print_analysis(metrics)
        return metrics
    
    def _calculate_compactness(self):
        """Calculate how compact the layout is (0-100 scale)."""
        if not self.env.placements:
            return 0
            
        min_x, max_x = self.env.grid_size, 0
        min_y, max_y = self.env.grid_size, 0
        
        for cid, (px, py) in self.env.placements.items():
            comp = next(c for c in self.env.components if c["id"] == cid)
            w, h = comp["width"], comp["height"]
            min_x = min(min_x, px)
            max_x = max(max_x, px + h)
            min_y = min(min_y, py)
            max_y = max(max_y, py + w)
        
        bounding_area = (max_x - min_x) * (max_y - min_y)
        max_possible_area = self.env.grid_size * self.env.grid_size
        
        return max(0, 100 * (1 - bounding_area / max_possible_area))
    
    def _calculate_symmetry(self):
        """Calculate symmetry score (0-100 scale)."""
        match_groups = {}
        for comp in self.env.components:
            if comp["match_group"]:
                if comp["match_group"] not in match_groups:
                    match_groups[comp["match_group"]] = []
                match_groups[comp["match_group"]].append(comp["id"])
        
        total_groups = len(match_groups)
        symmetric_groups = 0
        
        for group, cids in match_groups.items():
            if len(cids) == 2 and all(c in self.env.placements for c in cids):
                cid1, cid2 = cids
                x1, y1 = self.env.placements[cid1]
                x2, y2 = self.env.placements[cid2]
                
                # Check for various types of symmetry
                if x1 == x2:  # Horizontal alignment
                    symmetric_groups += 0.5
                
                # Perfect vertical symmetry around center
                center_y = self.env.grid_size / 2
                comp1 = next(c for c in self.env.components if c["id"] == cid1)
                if abs((y1 + comp1["width"]/2) - center_y) == abs((y2 + comp1["width"]/2) - center_y):
                    symmetric_groups += 0.5
        
        return (symmetric_groups / max(1, total_groups)) * 100 if total_groups > 0 else 0
    
    def _calculate_connectivity(self):
        """Calculate connectivity score (0-100 scale)."""
        if not self.env.connections:
            return 100  # No connections to optimize
            
        total_score = 0
        for cid1, cid2 in self.env.connections:
            if cid1 in self.env.placements and cid2 in self.env.placements:
                x1, y1 = self.env.placements[cid1]
                x2, y2 = self.env.placements[cid2]
                
                # Manhattan distance
                dist = abs(x1 - x2) + abs(y1 - y2)
                max_dist = 2 * self.env.grid_size  # Maximum possible distance
                
                # Score: closer is better
                score = max(0, 100 * (1 - dist / max_dist))
                total_score += score
        
        return total_score / len(self.env.connections)
    
    def _calculate_area_efficiency(self):
        """Calculate how efficiently the area is used."""
        total_component_area = 0
        for cid in self.env.placed_cids:
            comp = next(c for c in self.env.components if c["id"] == cid)
            total_component_area += comp["width"] * comp["height"]
        
        grid_area = self.env.grid_size * self.env.grid_size
        return (total_component_area / grid_area) * 100
    
    def _print_analysis(self, metrics):
        """Print formatted analysis results."""
        print("\n" + "="*50)
        print("           LAYOUT ANALYSIS REPORT")
        print("="*50)
        print(f"Components Placed: {metrics['total_components']}/{metrics['expected_components']}")
        print(f"Completion Rate:   {metrics['completion_rate']:.1%}")
        print(f"Compactness Score: {metrics['compactness_score']:.1f}/100")
        print(f"Symmetry Score:    {metrics['symmetry_score']:.1f}/100")
        print(f"Connectivity Score:{metrics['connectivity_score']:.1f}/100")
        print(f"Area Efficiency:   {metrics['area_efficiency']:.1f}%")
        print("="*50)
        
        # Overall grade
        overall = np.mean([metrics['compactness_score'], metrics['symmetry_score'], 
                          metrics['connectivity_score']]) * metrics['completion_rate']
        grade = 'A' if overall >= 80 else 'B' if overall >= 60 else 'C' if overall >= 40 else 'D'
        print(f"Overall Grade:     {grade} ({overall:.1f}/100)")
        print("="*50)

def evaluate_layout_with_analysis(env_or_model_path):
    """Helper function to quickly evaluate and analyze a layout."""
    if isinstance(env_or_model_path, str):
        # Load model and run
        from stable_baselines3 import PPO
        model = PPO.load(env_or_model_path)
        env = AnalogICLayoutEnv(grid_size=20)
        
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action.item())
    else:
        # Use provided environment
        env = env_or_model_path
    
    analyzer = LayoutAnalyzer(env)
    metrics = analyzer.analyze_layout()
    env.render(mode='human')
    
    return metrics
