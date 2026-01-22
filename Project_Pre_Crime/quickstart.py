#!/usr/bin/env python3
"""
Quick Start Script for Pre-Crime Prediction System

This script verifies that all dependencies are installed and provides
a simple test of the core functionality without requiring Neo4j.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Pre-Crime Prediction System - Quick Start")
print("=" * 70)

# Check Python version
print("\n1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
print("   ✓ Python version OK")

# Check dependencies
print("\n2. Checking dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'torch_geometric': 'PyTorch Geometric',
    'neo4j': 'Neo4j Driver',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'faker': 'Faker',
    'scipy': 'SciPy',
    'sklearn': 'Scikit-learn',
    'matplotlib': 'Matplotlib'
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} not found")
        missing_deps.append(module)

if missing_deps:
    print(f"\n   ERROR: Missing dependencies: {', '.join(missing_deps)}")
    print(f"   Install with: pip install -r requirements.txt")
    sys.exit(1)

print("\n   ✓ All dependencies installed")

# Test core modules
print("\n3. Testing core modules...")
try:
    import torch
    import numpy as np
    from faker import Faker
    
    # Test Beta distribution
    print("   Testing Beta distribution for risk_seed...")
    np.random.seed(42)
    risks = np.random.beta(2, 5, 100)
    print(f"   ✓ Generated {len(risks)} risk seeds")
    print(f"     Mean: {risks.mean():.3f}, Std: {risks.std():.3f}")
    
    # Test Faker
    print("   Testing Faker for data generation...")
    fake = Faker()
    Faker.seed(42)
    name = fake.name()
    print(f"   ✓ Generated sample name: {name}")
    
    # Test PyTorch
    print("   Testing PyTorch...")
    x = torch.randn(10, 2)
    print(f"   ✓ Created tensor of shape {x.shape}")
    
    # Test torch_geometric
    print("   Testing PyTorch Geometric...")
    from torch_geometric.data import Data
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    data = Data(x=x[:3], edge_index=edge_index)
    print(f"   ✓ Created graph with {data.num_nodes} nodes, {data.num_edges} edges")
    
except Exception as e:
    print(f"   ❌ Error during testing: {e}")
    sys.exit(1)

print("\n   ✓ Core modules working correctly")

# Test project modules (without Neo4j)
print("\n4. Testing project modules...")
try:
    # Add src to path
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))
    
    # Test models
    print("   Testing models.py...")
    from models import RedGAN
    model = RedGAN(in_channels=2, hidden_channels=16, embedding_dim=8)
    print(f"   ✓ RedGAN model initialized")
    
    # Test forward pass
    x = torch.randn(5, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    output = model(x, edge_index)
    print(f"   ✓ Forward pass successful")
    print(f"     Output: {output['risk_scores'].shape[0]} risk scores")
    
except Exception as e:
    print(f"   ❌ Error testing project modules: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n   ✓ Project modules working correctly")

# Summary
print("\n" + "=" * 70)
print("✓ Quick Start Complete - All checks passed!")
print("=" * 70)
print("\nNext steps:")
print("1. Setup Neo4j database (see README.md)")
print("2. Run example: python examples/demo_risk_evolution.py")
print("3. Run full pipeline: python src/train.py")
print("\nFor more information, see Project_Pre_Crime/README.md")
print("=" * 70)
