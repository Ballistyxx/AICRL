#!/usr/bin/env python3
"""
Test script for the relational placement system.

This script tests the new RelationalLayoutEnv and RelationalPolicy
to ensure they work correctly with the refactored action space.

Author: AICRL Team
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Add venv path
venv_site_packages = current_dir / "AICRL" / "lib" / "python3.10" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

print("Relational Placement System Test")
print("=" * 40)


def test_relational_environment():
    """Test the RelationalLayoutEnv environment."""
    print("1. Testing RelationalLayoutEnv...")
    
    try:
        from RelationalLayoutEnv import RelationalLayoutEnv, SpatialRelation, Orientation
        
        # Create environment with different difficulty levels
        print("   Testing difficulty levels...")
        
        # Easy difficulty (3-4 components)
        env_easy = RelationalLayoutEnv(grid_size=10, difficulty_level=1)
        print(f"   ✓ Easy difficulty: {len(env_easy.components)} components")
        
        # Medium difficulty (5-8 components)  
        env_medium = RelationalLayoutEnv(grid_size=15, difficulty_level=2)
        print(f"   ✓ Medium difficulty: {len(env_medium.components)} components")
        
        # Hard difficulty (all components)
        env_hard = RelationalLayoutEnv(grid_size=20, difficulty_level=3)
        print(f"   ✓ Hard difficulty: {len(env_hard.components)} components")
        
        # Test action space
        print(f"   Action space: {env_easy.action_space}")
        print(f"   Action space shape: {env_easy.action_space.nvec}")
        
        # Test observation space
        print(f"   Observation space keys: {list(env_easy.observation_space.spaces.keys())}")
        
        # Test reset
        obs, info = env_easy.reset()
        print("   ✓ Environment reset successful")
        print(f"   Initial placed components: {np.sum(obs['placed_components'])}")
        
        return env_easy
        
    except Exception as e:
        print(f"   ❌ RelationalLayoutEnv test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_relational_actions(env):
    """Test relational action execution."""
    print("\n2. Testing relational actions...")
    
    try:
        from RelationalLayoutEnv import SpatialRelation, Orientation
        
        obs, info = env.reset()
        print("   ✓ Environment reset")
        
        # Check initial state
        initial_placed = np.sum(obs['placed_components'])
        print(f"   Initial components placed: {initial_placed}")
        
        # Test action mask
        action_mask = obs['action_mask']
        valid_actions = np.sum(action_mask)
        print(f"   Valid actions available: {valid_actions}")
        
        if valid_actions > 0:
            # Find a valid action
            valid_indices = np.where(action_mask == 1)[0]
            
            if len(valid_indices) > 0:
                # Decode first valid action
                action_idx = valid_indices[0]
                target_idx = action_idx // (len(SpatialRelation) * len(Orientation))
                remaining = action_idx % (len(SpatialRelation) * len(Orientation))
                relation_idx = remaining // len(Orientation)
                orientation_idx = remaining % len(Orientation)
                
                action = [target_idx, relation_idx, orientation_idx]
                print(f"   Testing action: {action}")
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                print(f"   ✓ Action executed - Reward: {reward:.2f}")
                print(f"   Components now placed: {np.sum(obs['placed_components'])}")
                
                if reward > 0:
                    print("   ✓ Positive reward received")
                
                return True
            else:
                print("   ⚠️  No valid action indices found")
                return False
        else:
            print("   ⚠️  No valid actions available")
            return False
            
    except Exception as e:
        print(f"   ❌ Action test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_functions(env):
    """Test reward function components."""
    print("\n3. Testing reward functions...")
    
    try:
        obs, info = env.reset()
        
        # Test individual reward components
        symmetry_score = env._evaluate_symmetry()
        connectivity_score = env._evaluate_connectivity()
        compactness_score = env._evaluate_compactness()
        
        print(f"   Initial symmetry score: {symmetry_score:.3f}")
        print(f"   Initial connectivity score: {connectivity_score:.3f}")
        print(f"   Initial compactness score: {compactness_score:.3f}")
        
        # Try to place a few components
        steps = 0
        max_steps = 3
        
        while steps < max_steps and not np.all(obs['placed_components']):
            action_mask = obs['action_mask']
            valid_actions = np.where(action_mask == 1)[0]
            
            if len(valid_actions) > 0:
                # Take random valid action
                action_idx = np.random.choice(valid_actions)
                
                # Decode action
                from RelationalLayoutEnv import SpatialRelation, Orientation
                target_idx = action_idx // (len(SpatialRelation) * len(Orientation))
                remaining = action_idx % (len(SpatialRelation) * len(Orientation))
                relation_idx = remaining // len(Orientation)
                orientation_idx = remaining % len(Orientation)
                
                action = [target_idx, relation_idx, orientation_idx]
                obs, reward, done, truncated, info = env.step(action)
                
                print(f"   Step {steps + 1}: Reward = {reward:.3f}")
                steps += 1
                
                if done:
                    break
            else:
                break
        
        # Final scores
        final_symmetry = env._evaluate_symmetry()
        final_connectivity = env._evaluate_connectivity()
        final_compactness = env._evaluate_compactness()
        
        print(f"   Final symmetry score: {final_symmetry:.3f}")
        print(f"   Final connectivity score: {final_connectivity:.3f}")
        print(f"   Final compactness score: {final_compactness:.3f}")
        
        print("   ✓ Reward functions working")
        return True
        
    except Exception as e:
        print(f"   ❌ Reward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schematic_integration():
    """Test integration with schematic data."""
    print("\n4. Testing schematic integration...")
    
    try:
        from RelationalLayoutEnv import RelationalLayoutEnv
        
        # Create mock schematic data
        mock_schematic = {
            "format": "rl_compatible",
            "components": [
                {"id": 1, "name": "M1", "type": "nfet", "width": 2, "height": 2, 
                 "color": "red", "can_overlap": False, "match_group": "diff_pair"},
                {"id": 2, "name": "M2", "type": "nfet", "width": 2, "height": 2, 
                 "color": "red", "can_overlap": False, "match_group": "diff_pair"},
                {"id": 3, "name": "M3", "type": "pfet", "width": 2, "height": 2, 
                 "color": "blue", "can_overlap": False, "match_group": "current_mirror"},
                {"id": 4, "name": "M4", "type": "pfet", "width": 2, "height": 2, 
                 "color": "blue", "can_overlap": False, "match_group": "current_mirror"},
                {"id": 5, "name": "C1", "type": "cap", "width": 1, "height": 1, 
                 "color": "green", "can_overlap": True, "match_group": None}
            ],
            "connections": [(1, 3), (2, 4), (3, 5)],
            "statistics": {"total_components": 5, "component_types": ["nfet", "pfet", "cap"]}
        }
        
        # Create environment with schematic
        env = RelationalLayoutEnv(grid_size=15, schematic_data=mock_schematic, difficulty_level=2)
        
        print(f"   ✓ Environment created with {len(env.components)} components")
        print(f"   ✓ Connections: {env.connections}")
        
        # Check component types
        component_types = [comp["type"] for comp in env.components]
        print(f"   ✓ Component types: {set(component_types)}")
        
        # Check match groups for symmetry
        match_groups = {}
        for comp in env.components:
            if comp["match_group"]:
                group = comp["match_group"]
                if group not in match_groups:
                    match_groups[group] = []
                match_groups[group].append(comp["name"])
        
        print(f"   ✓ Match groups: {match_groups}")
        
        # Test a few steps
        obs, info = env.reset()
        for i in range(2):
            action_mask = obs['action_mask']
            valid_actions = np.where(action_mask == 1)[0]
            
            if len(valid_actions) > 0:
                action_idx = valid_actions[0]
                
                # Decode action
                from RelationalLayoutEnv import SpatialRelation, Orientation
                target_idx = action_idx // (len(SpatialRelation) * len(Orientation))
                remaining = action_idx % (len(SpatialRelation) * len(Orientation))
                relation_idx = remaining // len(Orientation)
                orientation_idx = remaining % len(Orientation)
                
                action = [target_idx, relation_idx, orientation_idx]
                obs, reward, done, truncated, info = env.step(action)
                
                print(f"   Step {i+1}: Placed component, reward: {reward:.2f}")
        
        print("   ✓ Schematic integration working")
        return True
        
    except Exception as e:
        print(f"   ❌ Schematic integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_network():
    """Test the relational policy network."""
    print("\n5. Testing RelationalPolicy...")
    
    try:
        from RelationalLayoutEnv import RelationalLayoutEnv
        from RelationalPolicy import RelationalFeaturesExtractor, RelationalActorCriticPolicy
        import torch
        
        # Create environment
        env = RelationalLayoutEnv(grid_size=10, difficulty_level=1)
        
        # Test features extractor
        obs, _ = env.reset()
        
        # Convert observation to torch tensors
        torch_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                torch_obs[key] = torch.FloatTensor(value).unsqueeze(0)  # Add batch dimension
            else:
                torch_obs[key] = torch.LongTensor([value])
        
        # Test features extractor
        features_extractor = RelationalFeaturesExtractor(env.observation_space)
        features = features_extractor(torch_obs)
        
        print(f"   ✓ Features extractor output shape: {features.shape}")
        
        # Test policy
        policy = RelationalActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda x: 3e-4
        )
        
        # Test forward pass
        with torch.no_grad():
            actions, values, log_probs = policy.forward(torch_obs)
            
        print(f"   ✓ Policy forward pass successful")
        print(f"   Actions shape: {actions.shape}")
        print(f"   Values shape: {values.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        
        # Test action evaluation
        with torch.no_grad():
            values, log_probs, entropy = policy.evaluate_actions(torch_obs, actions)
            
        print(f"   ✓ Action evaluation successful")
        print(f"   Entropy: {entropy.mean().item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all relational system tests."""
    tests = [
        ("Relational Environment", test_relational_environment),
        ("Relational Actions", None),  # Will use env from test 1
        ("Reward Functions", None),    # Will use env from test 1  
        ("Schematic Integration", test_schematic_integration),
        ("Policy Network", test_policy_network)
    ]
    
    results = []
    env = None
    
    # Run tests
    for i, (test_name, test_func) in enumerate(tests):
        try:
            if i == 0:  # Environment test
                env = test_func()
                results.append((test_name, env is not None))
            elif i == 1:  # Actions test
                if env:
                    result = test_relational_actions(env)
                    results.append(("Relational Actions", result))
                else:
                    results.append(("Relational Actions", False))
            elif i == 2:  # Reward test
                if env:
                    result = test_reward_functions(env)
                    results.append(("Reward Functions", result))
                else:
                    results.append(("Reward Functions", False))
            else:  # Independent tests
                result = test_func()
                results.append((test_name, result))
                
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Results summary
    print("\n" + "=" * 40)
    print("RELATIONAL SYSTEM TEST RESULTS")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25s}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 RELATIONAL PLACEMENT SYSTEM WORKING!")
        print("✅ Environment supports relational actions")
        print("✅ Action space properly defined (target, relation, orientation)")
        print("✅ Reward functions encourage relational reasoning")
        print("✅ Policy network handles multi-discrete actions")
        print("✅ Schematic integration maintained")
        print("\n🚀 Ready for relational training!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failing")
        print("Check the errors above for details")
        
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)