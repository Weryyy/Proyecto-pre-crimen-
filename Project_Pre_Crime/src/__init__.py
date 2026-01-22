"""
Pre-Crime Prediction System

A graph neural network-based system for predicting criminal behavior
using Neo4j, GraphSAGE, and GAT.
"""

from .connector import Neo4jConnector
from .models import RedGAN, GraphSAGE, GAT, compute_loss
from .train import PreCrimeTrainer

__version__ = "0.1.0"
__all__ = [
    'Neo4jConnector',
    'RedGAN',
    'GraphSAGE',
    'GAT',
    'compute_loss',
    'PreCrimeTrainer'
]
