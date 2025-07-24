"""
Custom policy network for relational placement actions.

This module implements a policy network that outputs relational placement actions
instead of absolute coordinates, enabling better generalization.

Author: AICRL Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Dict, List, Tuple, Type, Union


class RelationalFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for relational observations.
    
    Processes grid, component positions, and relational information.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        # Calculate total feature dimension
        super().__init__(observation_space, features_dim)
        
        # Get observation space components
        grid_space = observation_space.spaces["grid"]
        placed_components_space = observation_space.spaces["placed_components"]
        component_positions_space = observation_space.spaces["component_positions"]
        
        self.grid_size = grid_space.shape[0]
        self.num_components = placed_components_space.shape[0]
        
        # Grid CNN for spatial features
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        )
        
        # Component state encoder
        self.component_encoder = nn.Sequential(
            nn.Linear(self.num_components + self.num_components * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Calculate combined feature dimensions dynamically
        # This will be set after the first forward pass
        self.relation_net = None
        self._features_dim = features_dim
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Extract grid features
        grid = observations["grid"].float()
        batch_size = grid.shape[0]
        
        # Add channel dimension for CNN
        grid = grid.unsqueeze(1)  # (batch, 1, grid_size, grid_size)
        grid_features = self.grid_cnn(grid)
        grid_features = grid_features.view(batch_size, -1)  # Flatten
        
        # Extract component features
        placed_components = observations["placed_components"].float()
        component_positions = observations["component_positions"].float()
        component_positions = component_positions.view(batch_size, -1)  # Flatten positions
        
        # Combine component information
        component_info = torch.cat([placed_components, component_positions], dim=1)
        component_features = self.component_encoder(component_info)
        
        # Next component information
        next_component_id = observations["next_component_id"].float()
        # Ensure consistent dimensions for batched data
        if next_component_id.dim() == 0:
            # Single scalar value
            next_component_id = next_component_id.unsqueeze(0).unsqueeze(1)
        elif next_component_id.dim() == 1:
            # Batch of scalars - add feature dimension
            next_component_id = next_component_id.unsqueeze(1)
        elif next_component_id.dim() == 3:
            # Already has batch and feature dims but might be squeezed wrong
            next_component_id = next_component_id.view(batch_size, -1)
        
        # Debug shapes if there's still a mismatch
        try:
            # Combine all features
            combined = torch.cat([grid_features, component_features, next_component_id], dim=1)
        except RuntimeError as e:
            print(f"Tensor shape error in feature combination:")
            print(f"  grid_features shape: {grid_features.shape}")
            print(f"  component_features shape: {component_features.shape}")
            print(f"  next_component_id shape: {next_component_id.shape}")
            print(f"  batch_size: {batch_size}")
            raise e
        
        # Create relation network dynamically on first forward pass
        if self.relation_net is None:
            combined_dim = combined.shape[1]
            self.relation_net = nn.Sequential(
                nn.Linear(combined_dim, self._features_dim),
                nn.ReLU(),
                nn.Linear(self._features_dim, self._features_dim),
                nn.ReLU()
            ).to(combined.device)
        
        features = self.relation_net(combined)
        
        return features


class RelationalActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for relational placement.
    
    Outputs three separate action heads:
    1. Target component selection (attention over placed components)
    2. Spatial relation selection
    3. Orientation selection
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs
    ):
        # Use custom features extractor
        kwargs["features_extractor_class"] = RelationalFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 512}
        
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Get action space dimensions
        self.num_components = action_space.nvec[0]
        self.num_relations = action_space.nvec[1]
        self.num_orientations = action_space.nvec[2]
        
        # Custom action heads
        self.target_head = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_components)
        )
        
        self.relation_head = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_relations)
        )
        
        self.orientation_head = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_orientations)
        )
        
        # Enhanced value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def _get_constructor_parameters(self):
        """Get parameters for reconstructing the policy."""
        data = super()._get_constructor_parameters()
        return data
    
    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass through the policy network.
        
        Returns actions and values.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get action logits from each head
        target_logits = self.target_head(features)
        relation_logits = self.relation_head(features)
        orientation_logits = self.orientation_head(features)
        
        # Apply action mask to target selection
        if "action_mask" in obs:
            action_mask = obs["action_mask"]
            # Reshape mask to match our multi-discrete structure
            mask_reshaped = action_mask.view(-1, self.num_components, self.num_relations, self.num_orientations)
            
            # For target selection, any valid action for this target makes it selectable
            target_mask = torch.any(torch.any(mask_reshaped, dim=3), dim=2).float()
            target_logits = target_logits + torch.log(target_mask + 1e-8)
        
        # Sample actions
        target_dist = torch.distributions.Categorical(logits=target_logits)
        relation_dist = torch.distributions.Categorical(logits=relation_logits)
        orientation_dist = torch.distributions.Categorical(logits=orientation_logits)
        
        if deterministic:
            target_action = torch.argmax(target_logits, dim=1)
            relation_action = torch.argmax(relation_logits, dim=1)
            orientation_action = torch.argmax(orientation_logits, dim=1)
        else:
            target_action = target_dist.sample()
            relation_action = relation_dist.sample()
            orientation_action = orientation_dist.sample()
        
        # Combine actions
        actions = torch.stack([target_action, relation_action, orientation_action], dim=1)
        
        # Calculate log probabilities
        target_log_prob = target_dist.log_prob(target_action)
        relation_log_prob = relation_dist.log_prob(relation_action)
        orientation_log_prob = orientation_dist.log_prob(orientation_action)
        log_prob = target_log_prob + relation_log_prob + orientation_log_prob
        
        # Calculate values
        values = self.value_net(features)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions and return values, log probabilities, and entropy.
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Get action logits
        target_logits = self.target_head(features)
        relation_logits = self.relation_head(features)
        orientation_logits = self.orientation_head(features)
        
        # Apply action mask
        if "action_mask" in obs:
            action_mask = obs["action_mask"]
            mask_reshaped = action_mask.view(-1, self.num_components, self.num_relations, self.num_orientations)
            target_mask = torch.any(torch.any(mask_reshaped, dim=3), dim=2).float()
            target_logits = target_logits + torch.log(target_mask + 1e-8)
        
        # Create distributions
        target_dist = torch.distributions.Categorical(logits=target_logits)
        relation_dist = torch.distributions.Categorical(logits=relation_logits)
        orientation_dist = torch.distributions.Categorical(logits=orientation_logits)
        
        # Extract individual actions
        target_actions = actions[:, 0]
        relation_actions = actions[:, 1]
        orientation_actions = actions[:, 2]
        
        # Calculate log probabilities
        target_log_prob = target_dist.log_prob(target_actions)
        relation_log_prob = relation_dist.log_prob(relation_actions)
        orientation_log_prob = orientation_dist.log_prob(orientation_actions)
        log_prob = target_log_prob + relation_log_prob + orientation_log_prob
        
        # Calculate entropy
        entropy = target_dist.entropy() + relation_dist.entropy() + orientation_dist.entropy()
        
        # Calculate values
        values = self.value_net(features)
        
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        """
        Get value estimates for observations.
        """
        features = self.extract_features(obs)
        return self.value_net(features)


class RelationalPPO:
    """
    Wrapper class to use RelationalActorCriticPolicy with PPO.
    """
    
    @staticmethod
    def create_policy(env):
        """Create a relational policy for the given environment."""
        from stable_baselines3 import PPO
        
        # Create PPO with custom policy
        model = PPO(
            policy=RelationalActorCriticPolicy,
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./relational_tensorboard/"
        )
        
        return model


# Helper function to create attention mechanism for target selection
class ComponentAttention(nn.Module):
    """
    Attention mechanism for selecting target components.
    """
    
    def __init__(self, feature_dim, num_components):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_components = num_components
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, query_features, component_features, placed_mask):
        """
        Apply attention over placed components.
        
        Args:
            query_features: Features for current state [batch, feature_dim]
            component_features: Features for each component [batch, num_components, feature_dim]
            placed_mask: Mask for placed components [batch, num_components]
        """
        batch_size = query_features.shape[0]
        
        # Compute attention
        Q = self.query(query_features).unsqueeze(1)  # [batch, 1, feature_dim]
        K = self.key(component_features)  # [batch, num_components, feature_dim]
        V = self.value(component_features)  # [batch, num_components, feature_dim]
        
        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2))  # [batch, 1, num_components]
        scores = scores.squeeze(1)  # [batch, num_components]
        
        # Apply placed mask (only attend to placed components)
        scores = scores + torch.log(placed_mask + 1e-8)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=1)
        
        return attention_weights