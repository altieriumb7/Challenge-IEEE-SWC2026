import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Compute global Adjacency Matrix for the 100 nodes
df = pd.read_csv('smart_city_traffic_mobility.csv')
hourly_agg = df.groupby(['latitude', 'longitude', 'hour']).agg({
    'emission_estimate': 'mean',
    'heavy_vehicle_count': 'mean',
    'nearby_school': 'max',
    'nearby_hospital': 'max',
    'average_wait_time': 'mean',
    'vehicle_count': 'mean',
    'public_transport_count': 'mean',
    'lanes': 'first',
    'speed_limit': 'first',
    'queue_length': 'mean',
    'emergency_vehicle_detected': 'max',
    'accident_reported': 'max'
}).reset_index()

coords = hourly_agg[hourly_agg['hour'] == 8][['latitude', 'longitude']].values
tri = Delaunay(coords)
num_nodes = len(coords)
adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for simplex in tri.simplices:
    for i in range(3):
        for j in range(i+1, 3):
            u, v = simplex[i], simplex[j]
            adj[u, v] = 1.0
            adj[v, u] = 1.0

adj_tilde = adj + np.eye(num_nodes)
deg = np.sum(adj_tilde, axis=1)
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
D_inv_sqrt = np.diag(deg_inv_sqrt)
norm_adj = np.dot(np.dot(D_inv_sqrt, adj_tilde), D_inv_sqrt)

global_norm_adj = torch.FloatTensor(norm_adj)

class PurePyTorchGCNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super(PurePyTorchGCNExtractor, self).__init__(observation_space, features_dim)
        self.node_features = 12
        self.num_nodes = observation_space.shape[0] // self.node_features
        
        self.weight1 = nn.Parameter(torch.FloatTensor(self.node_features, 32))
        self.bias1 = nn.Parameter(torch.FloatTensor(32))
        
        self.weight2 = nn.Parameter(torch.FloatTensor(32, 64))
        self.bias2 = nn.Parameter(torch.FloatTensor(64))
        
        nn.init.xavier_uniform_(self.weight1)
        nn.init.zeros_(self.bias1)
        nn.init.xavier_uniform_(self.weight2)
        nn.init.zeros_(self.bias2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        x = observations.view(batch_size, self.num_nodes, self.node_features)
        
        adj = global_norm_adj.to(observations.device)
        
        ax = torch.matmul(adj, x) 
        out1 = F.relu(torch.matmul(ax, self.weight1) + self.bias1)
        
        ax2 = torch.matmul(adj, out1)
        out2 = F.relu(torch.matmul(ax2, self.weight2) + self.bias2)
        
        graph_embedding = torch.mean(out2, dim=1)
        return graph_embedding
