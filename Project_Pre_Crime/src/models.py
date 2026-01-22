"""
Graph Neural Network Models for Pre-Crime Prediction

This module implements:
- GraphSAGE: For neighborhood embedding aggregation
- GAT (Graph Attention Network): For crime prediction with attention mechanism
- RedGAN: Combines GraphSAGE generator with GAT discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for generating node embeddings through neighborhood aggregation
    
    This acts as the generator in the RedGAN architecture, creating embeddings
    that capture the structural and feature information of the graph.
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 32,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        """
        Initialize GraphSAGE model
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
        """
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.batch_norms.append(nn.BatchNorm1d(out_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layers
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        x = self.batch_norms[-1](x)
        
        return x


class GAT(nn.Module):
    """
    Graph Attention Network for crime prediction
    
    This acts as the discriminator in the RedGAN architecture, using attention
    mechanisms to predict crime risk and identify anomalies.
    The attention mechanism helps the model focus on important relationships
    and interactions that contribute to criminal behavior.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,  # Crime risk score
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.5):
        """
        Initialize GAT model
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension (1 for risk score)
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                     heads=heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer (single head for final prediction)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, 
                                 heads=1, concat=False, dropout=dropout))
        
        # Additional layers for risk prediction
        self.risk_predictor = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Output risk score between 0 and 1
        )
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through GAT layers
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (risk_scores, attention_weights)
            - risk_scores: [num_nodes, 1] crime risk predictions
            - attention_weights: Optional attention weights if return_attention=True
        """
        attention_weights = []
        
        for i, conv in enumerate(self.convs[:-1]):
            if return_attention:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)
            
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        if return_attention:
            x, (edge_idx, attn) = self.convs[-1](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, attn))
        else:
            x = self.convs[-1](x, edge_index)
        
        # Predict risk scores
        risk_scores = self.risk_predictor(x)
        
        if return_attention:
            return risk_scores, attention_weights
        else:
            return risk_scores, None


class RedGAN(nn.Module):
    """
    RedGAN: Risk Estimation and Detection GAN
    
    Combines GraphSAGE (Generator) and GAT (Discriminator) for crime prediction.
    
    Architecture:
    1. GraphSAGE generates node embeddings from graph structure
    2. GAT uses attention to predict crime risk (discriminator)
    3. The model learns to identify "Red Balls" (high-risk individuals/patterns)
    
    Training uses adversarial learning where:
    - Generator tries to create realistic embeddings
    - Discriminator tries to distinguish high-risk from low-risk patterns
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 embedding_dim: int = 32,
                 num_sage_layers: int = 2,
                 num_gat_layers: int = 2,
                 gat_heads: int = 4,
                 dropout: float = 0.5):
        """
        Initialize RedGAN model
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Hidden dimension for both models
            embedding_dim: Dimension of GraphSAGE embeddings
            num_sage_layers: Number of GraphSAGE layers
            num_gat_layers: Number of GAT layers
            gat_heads: Number of attention heads in GAT
            dropout: Dropout probability
        """
        super(RedGAN, self).__init__()
        
        # Generator: GraphSAGE for embedding generation
        self.generator = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_sage_layers,
            dropout=dropout
        )
        
        # Discriminator: GAT for risk prediction
        self.discriminator = GAT(
            in_channels=embedding_dim,  # Takes GraphSAGE embeddings as input
            hidden_channels=hidden_channels,
            out_channels=embedding_dim,
            num_layers=num_gat_layers,
            heads=gat_heads,
            dropout=dropout
        )
        
        # Anomaly detection threshold (learned during training)
        self.register_buffer('anomaly_threshold', torch.tensor(0.5))
        
    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor,
                return_attention: bool = False) -> dict:
        """
        Forward pass through RedGAN
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - embeddings: GraphSAGE node embeddings
            - risk_scores: GAT risk predictions
            - red_balls: Binary mask for high-risk nodes
            - attention_weights: Optional attention weights
        """
        # Generate embeddings
        embeddings = self.generator(x, edge_index)
        
        # Predict risk scores
        risk_scores, attention_weights = self.discriminator(
            embeddings, edge_index, return_attention=return_attention
        )
        
        # Identify "Red Balls" (high-risk individuals)
        red_balls = (risk_scores > self.anomaly_threshold).squeeze()
        
        return {
            'embeddings': embeddings,
            'risk_scores': risk_scores,
            'red_balls': red_balls,
            'attention_weights': attention_weights
        }
    
    def detect_anomalies(self, 
                        x: torch.Tensor, 
                        edge_index: torch.Tensor,
                        threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalous high-risk nodes in the graph
        
        Args:
            x: Node feature matrix
            edge_index: Graph edge indices
            threshold: Custom threshold (uses model's threshold if None)
            
        Returns:
            Tuple of (risk_scores, anomaly_mask)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x, edge_index)
            risk_scores = output['risk_scores']
            
            if threshold is not None:
                anomaly_mask = (risk_scores > threshold).squeeze()
            else:
                anomaly_mask = output['red_balls']
            
        return risk_scores, anomaly_mask
    
    def update_threshold(self, new_threshold: float):
        """Update anomaly detection threshold"""
        self.anomaly_threshold = torch.tensor(new_threshold)


def compute_loss(output: dict, 
                target_risk: torch.Tensor,
                lambda_gen: float = 0.5,
                lambda_disc: float = 0.5) -> Tuple[torch.Tensor, dict]:
    """
    Compute combined loss for RedGAN training
    
    Args:
        output: Model output dictionary
        target_risk: Target risk scores (from risk_seed evolution)
        lambda_gen: Weight for generator loss
        lambda_disc: Weight for discriminator loss
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    risk_scores = output['risk_scores'].squeeze()
    
    # Discriminator loss: MSE between predicted and actual risk
    disc_loss = F.mse_loss(risk_scores, target_risk)
    
    # Generator loss: Encourage diverse embeddings
    embeddings = output['embeddings']
    # Embedding diversity loss (minimize correlation between embeddings)
    embedding_mean = embeddings.mean(dim=0, keepdim=True)
    gen_loss = F.mse_loss(embeddings, embedding_mean)
    
    # Total loss
    total_loss = lambda_disc * disc_loss + lambda_gen * gen_loss
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'discriminator_loss': disc_loss.item(),
        'generator_loss': gen_loss.item()
    }
    
    return total_loss, loss_dict


if __name__ == "__main__":
    # Test the models
    print("Testing RedGAN model...")
    
    # Create dummy data
    num_nodes = 50
    num_features = 2
    num_edges = 100
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    target_risk = torch.rand(num_nodes)  # Simulated risk_seed values
    
    # Initialize model
    model = RedGAN(
        in_channels=num_features,
        hidden_channels=64,
        embedding_dim=32,
        num_sage_layers=2,
        num_gat_layers=2,
        gat_heads=4
    )
    
    # Forward pass
    output = model(x, edge_index, return_attention=True)
    
    print(f"Embeddings shape: {output['embeddings'].shape}")
    print(f"Risk scores shape: {output['risk_scores'].shape}")
    print(f"Number of red balls: {output['red_balls'].sum().item()}")
    
    # Compute loss
    loss, loss_dict = compute_loss(output, target_risk)
    print(f"\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test anomaly detection
    risk_scores, anomalies = model.detect_anomalies(x, edge_index, threshold=0.5)
    print(f"\nAnomalies detected: {anomalies.sum().item()}/{num_nodes}")
    
    print("\nModel test complete!")
