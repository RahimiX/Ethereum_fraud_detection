"""
Graph Attention Network (GAT) Model for Ethereum Fraud Detection
This model uses multi-head attention to learn from transaction graph structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


class GATFraudDetector(nn.Module):
    """
    Graph Attention Network for fraud detection on Ethereum transaction graphs.
    
    Uses multi-head attention to capture complex patterns in transaction networks.
    """
    
    def __init__(self, in_channels, hidden_channels=128, out_channels=2, 
                 num_heads=8, num_layers=3, dropout=0.5, use_batch_norm=True):
        """
        Initialize GAT model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden channels per head
            out_channels: Number of output classes (2 for binary classification)
            num_heads: Number of attention heads in each layer
            num_layers: Number of GAT layers
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super(GATFraudDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer: in_channels -> hidden_channels
        self.convs.append(
            GATConv(
                in_channels, 
                hidden_channels, 
                heads=num_heads, 
                dropout=dropout,
                concat=True,
                edge_dim=None  # Can be extended to use edge features
            )
        )
        
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels * num_heads))
        
        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=None
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels * num_heads))
        
        # Output layer: hidden_channels -> out_channels
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_channels * num_heads,
                    out_channels,
                    heads=1,  # Single head for final layer
                    dropout=dropout,
                    concat=False,
                    edge_dim=None
                )
            )
        else:
            # Single layer case
            self.convs[0] = GATConv(
                in_channels,
                out_channels,
                heads=1,
                dropout=dropout,
                concat=False,
                edge_dim=None
            )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
        
        Returns:
            Logits [num_nodes, out_channels]
        """
        # Apply GAT layers with residual connections and batch norm
        for i, conv in enumerate(self.convs):
            # Store residual
            residual = x if i > 0 and x.shape[1] == conv.out_channels else None
            
            # Apply convolution
            x = conv(x, edge_index, edge_attr)
            
            # Apply batch norm if enabled
            if self.use_batch_norm and i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if residual is not None and x.shape[1] == residual.shape[1]:
                x = x + residual
        
        return x
    
    def get_attention_weights(self, x, edge_index, layer_idx=0):
        """
        Get attention weights for visualization.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity
            layer_idx: Which layer to get attention from
        
        Returns:
            Attention weights
        """
        if layer_idx >= len(self.convs):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        self.eval()
        with torch.no_grad():
            # Forward through layers up to layer_idx
            for i in range(layer_idx):
                x = self.convs[i](x, edge_index)
                if self.use_batch_norm and i < len(self.convs) - 1:
                    x = self.batch_norms[i](x)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
            
            # Get attention from target layer
            # Note: This requires modifying the GATConv to return attention
            # For now, we'll just return the node representations
            return x


class GraphSAGEFraudDetector(nn.Module):
    """
    GraphSAGE model as an alternative to GAT for fraud detection.
    Uses neighborhood sampling and aggregation.
    """
    
    def __init__(self, in_channels, hidden_channels=128, out_channels=2,
                 num_layers=3, dropout=0.5, use_batch_norm=True):
        """Initialize GraphSAGE model."""
        super(GraphSAGEFraudDetector, self).__init__()
        
        from torch_geometric.nn import SAGEConv
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass."""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if self.use_batch_norm and i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class TemporalGATFraudDetector(nn.Module):
    """
    Temporal Graph Attention Network (TGAT) for fraud detection.
    
    This model incorporates temporal information from edge timestamps into the
    attention mechanism, allowing it to learn time-aware patterns in transaction graphs.
    Key features:
    - Time encoding for timestamps
    - Edge-timestamp aware attention
    - Temporal decay in attention weights
    """
    
    def __init__(self, in_channels, hidden_channels=128, out_channels=2,
                 num_heads=8, num_layers=3, dropout=0.5, use_batch_norm=True,
                 time_dim=64, edge_dim=5):
        """
        Initialize Temporal GAT model.
        
        Args:
            in_channels: Number of input node features
            hidden_channels: Number of hidden channels per head
            out_channels: Number of output classes (2 for binary classification)
            num_heads: Number of attention heads in each layer
            num_layers: Number of GAT layers
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            time_dim: Dimension for time encoding
            edge_dim: Dimension of edge features (including timestamp)
        """
        super(TemporalGATFraudDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        
        # Time encoding layer (sinusoidal encoding)
        self.time_encoder = TimeEncoder(time_dim)
        
        # Project edge features (including time encoding) to attention dimension
        self.edge_proj = nn.Linear(edge_dim + time_dim, num_heads)
        
        # Build GAT layers with edge features support
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # First layer: in_channels -> hidden_channels
        self.convs.append(
            TemporalGATLayer(
                in_channels,
                hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                time_dim=time_dim,
                edge_dim=edge_dim
            )
        )
        
        if use_batch_norm:
            self.batch_norms.append(BatchNorm(hidden_channels * num_heads))
        
        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(
                TemporalGATLayer(
                    hidden_channels * num_heads,
                    hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    time_dim=time_dim,
                    edge_dim=edge_dim
                )
            )
            if use_batch_norm:
                self.batch_norms.append(BatchNorm(hidden_channels * num_heads))
        
        # Output layer: hidden_channels -> out_channels
        if num_layers > 1:
            self.convs.append(
                TemporalGATLayer(
                    hidden_channels * num_heads,
                    out_channels,
                    num_heads=1,  # Single head for final layer
                    dropout=dropout,
                    time_dim=time_dim,
                    edge_dim=edge_dim
                )
            )
        else:
            # Single layer case
            self.convs[0] = TemporalGATLayer(
                in_channels,
                out_channels,
                num_heads=1,
                dropout=dropout,
                time_dim=time_dim,
                edge_dim=edge_dim
            )
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with temporal attention.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (must include timestamp at index 1)
        
        Returns:
            Logits [num_nodes, out_channels]
        """
        # Extract timestamps from edge attributes
        if edge_attr is not None:
            # Assume timestamp is at index 1 in edge features
            timestamps = edge_attr[:, 1:2]  # Keep as [num_edges, 1] for broadcasting
        else:
            # If no edge attributes, use zeros (fallback)
            timestamps = torch.zeros(edge_index.size(1), 1, device=x.device)
        
        # Apply temporal GAT layers
        for i, conv in enumerate(self.convs):
            # Store residual
            residual = x if i > 0 and x.shape[1] == conv.out_channels else None
            
            # Apply temporal convolution
            x = conv(x, edge_index, edge_attr, timestamps)
            
            # Apply batch norm if enabled
            if self.use_batch_norm and i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
            
            # Apply activation (except for last layer)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (if dimensions match)
            if residual is not None and x.shape[1] == residual.shape[1]:
                x = x + residual
        
        return x


class TimeEncoder(nn.Module):
    """
    Sinusoidal time encoding for timestamps.
    Transforms timestamps into learnable temporal representations.
    """
    
    def __init__(self, time_dim):
        """
        Initialize time encoder.
        
        Args:
            time_dim: Dimension of time encoding
        """
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        
        # Learnable frequency parameters
        self.freq = nn.Parameter(torch.randn(time_dim // 2))
        
    def forward(self, timestamps):
        """
        Encode timestamps using sinusoidal encoding.
        
        Args:
            timestamps: Tensor of shape [num_edges, 1] or [num_edges]
        
        Returns:
            Time encoding [num_edges, time_dim]
        """
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(1)
        
        # Normalize timestamps (assuming they're in reasonable range)
        # Use log scale for better numerical stability
        timestamps_norm = torch.log1p(timestamps + 1e-6)
        
        # Create frequency components
        # Use different frequencies for different dimensions
        angles = timestamps_norm * self.freq.unsqueeze(0)  # [num_edges, time_dim//2]
        
        # Sinusoidal encoding: [sin(angle), cos(angle)]
        time_encoding = torch.zeros(timestamps.size(0), self.time_dim, device=timestamps.device)
        time_encoding[:, 0::2] = torch.sin(angles)
        time_encoding[:, 1::2] = torch.cos(angles)
        
        return time_encoding


class TemporalGATLayer(nn.Module):
    """
    Single layer of Temporal GAT with edge-timestamp aware attention.
    Uses efficient edge-wise attention computation similar to PyG's GATConv.
    """
    
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.5,
                 time_dim=64, edge_dim=5):
        """
        Initialize temporal GAT layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension per head
            num_heads: Number of attention heads
            dropout: Dropout probability
            time_dim: Dimension of time encoding
            edge_dim: Dimension of edge features
        """
        super(TemporalGATLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Linear transformations for query, key, value
        self.lin_q = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.lin_k = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.lin_v = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        # Time encoder
        self.time_encoder = TimeEncoder(time_dim)
        
        # Temporal attention bias (learnable)
        self.temporal_bias = nn.Parameter(torch.randn(num_heads))
        
        # Edge feature projection (including time encoding)
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim + time_dim, num_heads)
        else:
            self.edge_proj = nn.Linear(time_dim, num_heads)
        
        # Attention weight parameter for temporal decay
        self.temporal_weight = nn.Parameter(torch.ones(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        if hasattr(self, 'edge_proj'):
            nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.normal_(self.temporal_bias, std=0.1)
        nn.init.ones_(self.temporal_weight)
    
    def forward(self, x, edge_index, edge_attr=None, timestamps=None):
        """
        Forward pass with temporal attention.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            timestamps: Timestamps [num_edges, 1] or [num_edges]
        
        Returns:
            Updated node features [num_nodes, out_channels * num_heads]
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Compute Q, K, V
        q = self.lin_q(x).view(num_nodes, self.num_heads, self.out_channels)  # [N, H, F]
        k = self.lin_k(x).view(num_nodes, self.num_heads, self.out_channels)  # [N, H, F]
        v = self.lin_v(x).view(num_nodes, self.num_heads, self.out_channels)  # [N, H, F]
        
        # Extract source and target nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Compute attention scores: Q^T * K
        q_src = q[src]  # [E, H, F]
        k_dst = k[dst]  # [E, H, F]
        
        # Basic attention: (Q^T * K) / sqrt(d)
        attn = (q_src * k_dst).sum(dim=-1) / math.sqrt(self.out_channels)  # [E, H]
        
        # Add temporal encoding to attention
        if timestamps is not None:
            # Ensure timestamps is 2D
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(1)
            
            # Encode time
            time_enc = self.time_encoder(timestamps)  # [E, time_dim]
            
            # Compute temporal attention bias
            # Normalize timestamps for temporal decay (more recent = higher attention)
            max_time = timestamps.max() if timestamps.numel() > 0 else 1.0
            min_time = timestamps.min() if timestamps.numel() > 0 else 0.0
            if max_time > min_time:
                time_norm = (timestamps - min_time) / (max_time - min_time + 1e-6)
            else:
                time_norm = torch.ones_like(timestamps)
            
            # Temporal bias: recent edges get boost
            temporal_boost = self.temporal_bias.unsqueeze(0) * time_norm  # [E, 1]
            attn = attn + temporal_boost.squeeze(-1)  # [E, H]
        else:
            time_enc = None
        
        # Add edge feature bias
        if edge_attr is not None:
            # Encode time from edge attributes if not already encoded
            if time_enc is None:
                if edge_attr.size(1) > 1:
                    timestamps = edge_attr[:, 1:2]
                    time_enc = self.time_encoder(timestamps)
                else:
                    time_enc = torch.zeros(edge_attr.size(0), self.time_encoder.time_dim, device=edge_attr.device)
            
            # Combine edge features with time encoding
            edge_time = torch.cat([edge_attr, time_enc], dim=-1)  # [E, edge_dim + time_dim]
            edge_bias = self.edge_proj(edge_time)  # [E, H]
            attn = attn + edge_bias
        
        # Apply LeakyReLU activation
        attn = F.leaky_relu(attn, negative_slope=0.2)  # [E, H]
        
        # Softmax over neighbors for each head (efficient implementation)
        # Group by destination node and apply softmax
        out = torch.zeros(num_nodes, self.num_heads, self.out_channels, device=x.device, dtype=x.dtype)
        
        # Use scatter to aggregate attention-weighted values
        # First, compute attention weights (softmax per destination node)
        attn_exp = attn.exp()  # [E, H]
        
        # Normalize attention weights per destination node
        attn_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.num_heads), attn_exp)
        attn_normalized = attn_exp / (attn_sum[dst] + 1e-8)  # [E, H]
        
        # Apply attention to values and aggregate
        v_src = v[src]  # [E, H, F]
        weighted_v = attn_normalized.unsqueeze(-1) * v_src  # [E, H, F]
        
        # Aggregate to destination nodes
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, self.out_channels), weighted_v)
        
        # Reshape output
        out = out.reshape(num_nodes, -1)  # [N, H*F]
        
        # Apply dropout
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out


def create_model(model_type='tgn', in_channels=19, **kwargs):
    """
    Factory function to create a model.
    
    Args:
        model_type: 'gat' or 'sage'
        in_channels: Number of input features
        **kwargs: Additional model arguments
    
    Returns:
        Initialized model
    """
    if model_type.lower() == 'tgn' or model_type.lower() == 'tgat':
        return TemporalGATFraudDetector(in_channels=in_channels, **kwargs)
    elif model_type.lower() == 'gat':
        return GATFraudDetector(in_channels=in_channels, **kwargs)
    elif model_type.lower() == 'sage':
        return GraphSAGEFraudDetector(in_channels=in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'tgn', 'gat', or 'sage'")


if __name__ == '__main__':
    # Test model creation
    print("Testing GAT model...")
    model = GATFraudDetector(in_channels=19, hidden_channels=64, out_channels=2, num_heads=4)
    
    # Create dummy data
    num_nodes = 100
    num_edges = 500
    x = torch.randn(num_nodes, 19)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Forward pass
    out = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Model created successfully!")

