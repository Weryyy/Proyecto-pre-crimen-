"""
Training Pipeline for Pre-Crime Prediction System

This module implements:
- Training loop for RedGAN model
- Anomaly detection and "Red Balls" identification
- Risk seed evolution over time
- Model evaluation and visualization
"""

import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from models import RedGAN, compute_loss
from connector import Neo4jConnector


class PreCrimeTrainer:
    """
    Trainer for the Pre-Crime Prediction System
    
    Handles training, evaluation, and anomaly detection using RedGAN
    """
    
    def __init__(self,
                 model: RedGAN,
                 connector: Neo4jConnector,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize trainer
        
        Args:
            model: RedGAN model instance
            connector: Neo4j connector for data access
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.connector = connector
        self.device = device
        
        # Optimizers for generator and discriminator
        self.optimizer_gen = optim.Adam(
            self.model.generator.parameters(),
            lr=learning_rate
        )
        self.optimizer_disc = optim.Adam(
            self.model.discriminator.parameters(),
            lr=learning_rate
        )
        self.optimizer_full = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'disc_loss': [],
            'gen_loss': [],
            'red_balls_count': [],
            'avg_risk': []
        }
        
    def evolve_graph_risks(self, data, time_step: float = 1.0):
        """
        Evolve risk seeds for all nodes in the graph
        
        Args:
            data: PyTorch Geometric Data object
            time_step: Time elapsed (in days)
            
        Returns:
            Updated risk seeds tensor
        """
        # Get current risks from node features (assuming risk is at index 1)
        current_risks = data.x[:, 1].cpu().numpy()
        
        # Get interaction counts for each node
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.x.shape[0]
        interaction_counts = np.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            interaction_counts[edge_index[0, i]] += 1
        
        # Evolve each node's risk
        new_risks = []
        for i in range(num_nodes):
            # Simulate location crime rate (could be enhanced with actual location data)
            avg_crime_rate = np.random.beta(2, 8)  # Random but realistic
            
            new_risk = self.connector.evolve_risk_seed(
                current_risk=current_risks[i],
                interactions=int(interaction_counts[i]),
                location_crime_rate=avg_crime_rate,
                time_delta=time_step
            )
            new_risks.append(new_risk)
        
        return torch.tensor(new_risks, dtype=torch.float)
    
    def train_epoch(self, data, target_risks: torch.Tensor, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            data: Training graph data
            target_risks: Target risk values from risk_seed evolution
            epoch: Current epoch number
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        
        # Move data to device
        data = data.to(self.device)
        target_risks = target_risks.to(self.device)
        
        # Forward pass
        output = self.model(data.x, data.edge_index)
        
        # Compute loss
        loss, loss_dict = compute_loss(output, target_risks)
        
        # Backward pass
        self.optimizer_full.zero_grad()
        loss.backward()
        self.optimizer_full.step()
        
        # Track red balls
        red_balls_count = output['red_balls'].sum().item()
        avg_risk = output['risk_scores'].mean().item()
        
        loss_dict['red_balls_count'] = red_balls_count
        loss_dict['avg_risk'] = avg_risk
        
        return loss_dict
    
    def train(self,
             num_epochs: int = 100,
             time_evolution_interval: int = 10,
             eval_interval: int = 10,
             save_interval: int = 20):
        """
        Main training loop with risk evolution
        
        Args:
            num_epochs: Number of training epochs
            time_evolution_interval: Epochs between risk seed evolution
            eval_interval: Epochs between evaluation
            save_interval: Epochs between checkpoint saves
        """
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print("-" * 50)
        
        # Extract initial graph data
        data = self.connector.extract_subgraph()
        target_risks = data.x[:, 1].clone()  # Initial risk seeds
        
        for epoch in range(1, num_epochs + 1):
            # Evolve risk seeds periodically
            if epoch % time_evolution_interval == 0:
                print(f"\n[Epoch {epoch}] Evolving risk seeds...")
                target_risks = self.evolve_graph_risks(data, time_step=1.0)
                # Update node features with new risks
                data.x[:, 1] = target_risks
            
            # Train one epoch
            metrics = self.train_epoch(data, target_risks, epoch)
            
            # Store history
            self.history['train_loss'].append(metrics['total_loss'])
            self.history['disc_loss'].append(metrics['discriminator_loss'])
            self.history['gen_loss'].append(metrics['generator_loss'])
            self.history['red_balls_count'].append(metrics['red_balls_count'])
            self.history['avg_risk'].append(metrics['avg_risk'])
            
            # Print progress
            if epoch % eval_interval == 0:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"  Total Loss: {metrics['total_loss']:.4f}")
                print(f"  Disc Loss: {metrics['discriminator_loss']:.4f}")
                print(f"  Gen Loss: {metrics['generator_loss']:.4f}")
                print(f"  Red Balls: {metrics['red_balls_count']:.0f}")
                print(f"  Avg Risk: {metrics['avg_risk']:.4f}")
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print("=" * 50)
        
        # Final evaluation
        self.evaluate(data, target_risks)
    
    def evaluate(self, data, target_risks: torch.Tensor):
        """
        Evaluate model and detect anomalies
        
        Args:
            data: Evaluation graph data
            target_risks: Target risk values
        """
        print("\nEvaluating model...")
        self.model.eval()
        
        data = data.to(self.device)
        target_risks = target_risks.to(self.device)
        
        with torch.no_grad():
            output = self.model(data.x, data.edge_index, return_attention=True)
            
            # Compute metrics
            predicted_risks = output['risk_scores'].squeeze()
            mse = torch.mean((predicted_risks - target_risks) ** 2).item()
            mae = torch.mean(torch.abs(predicted_risks - target_risks)).item()
            
            # Anomaly detection
            red_balls = output['red_balls']
            num_red_balls = red_balls.sum().item()
            
            print(f"\nEvaluation Results:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Red Balls Detected: {num_red_balls}/{data.x.shape[0]}")
            print(f"  Red Ball Percentage: {100 * num_red_balls / data.x.shape[0]:.2f}%")
            
            # Identify top risk individuals
            top_k = min(10, data.x.shape[0])
            top_risks, top_indices = torch.topk(predicted_risks, top_k)
            
            print(f"\nTop {top_k} High-Risk Nodes:")
            for i, (idx, risk) in enumerate(zip(top_indices, top_risks)):
                actual_risk = target_risks[idx].item()
                print(f"  {i+1}. Node {idx.item()}: "
                      f"Predicted={risk.item():.4f}, Actual={actual_risk:.4f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'num_red_balls': num_red_balls,
            'red_ball_percentage': 100 * num_red_balls / data.x.shape[0]
        }
    
    def detect_red_balls(self, data, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Detect and report red balls (high-risk individuals)
        
        Args:
            data: Graph data
            threshold: Risk threshold (uses model default if None)
            
        Returns:
            DataFrame with red ball information
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            risk_scores, anomaly_mask = self.model.detect_anomalies(
                data.x, data.edge_index, threshold
            )
        
        # Create DataFrame
        results = []
        for i, is_red_ball in enumerate(anomaly_mask):
            if is_red_ball:
                results.append({
                    'node_id': i,
                    'predicted_risk': risk_scores[i].item(),
                    'actual_risk': data.x[i, 1].item(),
                    'age_normalized': data.x[i, 0].item()
                })
        
        return pd.DataFrame(results)
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_full_state_dict': self.optimizer_full.state_dict(),
            'optimizer_gen_state_dict': self.optimizer_gen.state_dict(),
            'optimizer_disc_state_dict': self.optimizer_disc.state_dict(),
            'history': self.history,
            'anomaly_threshold': self.model.anomaly_threshold.item()
        }, checkpoint_path)
        
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_full.load_state_dict(checkpoint['optimizer_full_state_dict'])
        self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
        self.history = checkpoint['history']
        self.model.update_threshold(checkpoint['anomaly_threshold'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
    
    def save_history(self, filename: str = 'training_history.json'):
        """Save training history to JSON file"""
        history_path = self.checkpoint_dir / filename
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("Pre-Crime Prediction System - Training Pipeline")
    print("=" * 50)
    
    # Configuration
    config = {
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'password',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 100,
        'learning_rate': 0.001,
        'hidden_channels': 64,
        'embedding_dim': 32,
        'num_sage_layers': 2,
        'num_gat_layers': 2,
        'gat_heads': 4,
        'dropout': 0.5,
        'checkpoint_dir': 'checkpoints'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize connector
    print("Connecting to Neo4j...")
    connector = Neo4jConnector(
        uri=config['neo4j_uri'],
        user=config['neo4j_user'],
        password=config['neo4j_password']
    )
    
    try:
        # Setup database and generate data
        print("\nSetting up database...")
        connector.setup_schema("../scripts/setup_db.cypher")
        
        print("\nGenerating synthetic data...")
        connector.generate_synthetic_data(
            num_citizens=100,
            num_locations=20,
            num_interactions=200,
            num_movements=300
        )
        
        # Extract graph data
        print("\nExtracting graph data...")
        data = connector.extract_subgraph()
        print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Node features: {data.x.shape}")
        
        # Initialize model
        print("\nInitializing RedGAN model...")
        model = RedGAN(
            in_channels=data.x.shape[1],
            hidden_channels=config['hidden_channels'],
            embedding_dim=config['embedding_dim'],
            num_sage_layers=config['num_sage_layers'],
            num_gat_layers=config['num_gat_layers'],
            gat_heads=config['gat_heads'],
            dropout=config['dropout']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Initialize trainer
        trainer = PreCrimeTrainer(
            model=model,
            connector=connector,
            device=config['device'],
            learning_rate=config['learning_rate'],
            checkpoint_dir=config['checkpoint_dir']
        )
        
        # Train model
        trainer.train(
            num_epochs=config['num_epochs'],
            time_evolution_interval=10,
            eval_interval=10,
            save_interval=20
        )
        
        # Save training history
        trainer.save_history()
        
        # Detect red balls
        print("\nDetecting Red Balls...")
        red_balls_df = trainer.detect_red_balls(data, threshold=0.5)
        print(f"\nRed Balls Summary:")
        print(red_balls_df.to_string())
        
        # Save red balls to CSV
        red_balls_path = Path(config['checkpoint_dir']) / 'red_balls.csv'
        red_balls_df.to_csv(red_balls_path, index=False)
        print(f"\nRed balls saved to {red_balls_path}")
        
        # Query high-risk citizens from database
        print("\nQuerying high-risk citizens from database...")
        high_risk_citizens = connector.get_high_risk_citizens(threshold=0.3)
        print(f"\nHigh-Risk Citizens (from database):")
        print(high_risk_citizens.head(10).to_string())
        
    finally:
        connector.close()
        print("\nDatabase connection closed.")
    
    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
