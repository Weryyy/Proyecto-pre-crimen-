# Pre-Crime Prediction System

A graph neural network-based system for predicting criminal behavior using Neo4j, GraphSAGE, and GAT (Graph Attention Networks).

🐳 **Docker Ready**: Fully containerized setup for easy deployment anywhere!  
📊 **3D Visualization**: Interactive browser-based 3D graph visualization!

## Quick Start with Docker

```bash
# Setup completo en 3 comandos
make build      # Construir contenedores
make setup      # Configurar base de datos
make visualize  # Ver visualización 3D en http://localhost:8050
```

Ver [DOCKER.md](DOCKER.md) para guía completa de Docker.

## Overview

This project implements a pre-crime prediction system inspired by minority report concepts, using:
- **Neo4j** for graph database storage
- **GraphSAGE** for neighborhood embedding generation
- **GAT (Graph Attention Networks)** for attention-based crime risk prediction
- **RedGAN** architecture combining generator and discriminator
- **Beta distribution** for realistic risk seed evolution
- **Faker** for synthetic data generation
- **Docker** for containerized deployment
- **Plotly Dash** for interactive 3D visualization

## Project Structure

```
Project_Pre_Crime/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── Makefile               # Convenient commands
├── requirements.txt        # Python dependencies
├── scripts/
│   └── setup_db.cypher    # Neo4j schema definition
├── src/
│   ├── connector.py       # Neo4j interaction and data generation
│   ├── models.py          # GraphSAGE and GAT models (RedGAN)
│   └── train.py           # Training pipeline
└── visualization/
    └── dashboard.py       # 3D interactive dashboard
```

## Features

### 1. Docker Containerization
- **Fully containerized**: Run anywhere with Docker
- **Docker Compose**: Neo4j + App + Jupyter in one command
- **No manual setup**: Everything automated
- **Persistent data**: Volumes for checkpoints and data

### 2. 3D Interactive Visualization
- **Browser-based**: View at http://localhost:8050
- **Real-time 3D**: Rotate, zoom, explore the graph
- **Multiple layouts**: t-SNE, PCA, Spring layout
- **Color-coded risk**: Visual risk levels
- **Red Balls highlight**: High-risk individuals clearly marked
- **Interactive controls**: Adjust threshold, toggle edges

### 3. Graph Database Schema
- **Citizen nodes**: Individuals with risk scores
- **Location nodes**: Places with crime rates
- **INTERACTS_WITH relationships**: Social connections
- **MOVES_TO relationships**: Location visits

### 4. Risk Seed Evolution
- Uses Beta distribution for realistic risk modeling
- Evolves based on:
  - Social interactions
  - Location crime rates
  - Time progression
  - Current risk level

### 5. RedGAN Architecture
- **Generator (GraphSAGE)**: Creates node embeddings from graph structure
- **Discriminator (GAT)**: Predicts crime risk using attention mechanisms
- **Red Balls Detection**: Identifies high-risk individuals

### 6. Synthetic Data Generation
- Realistic citizen profiles using Faker
- Location data with geographical coordinates
- Social interaction networks
- Movement patterns

## Installation

### Option 1: Docker (Recommended) 🐳

**Prerequisites**: Docker and Docker Compose installed

```bash
# Clone repository
git clone <repo-url>
cd Project_Pre_Crime

# Quick start
make demo

# Or step by step
make build      # Build containers
make setup      # Setup database and generate data
make visualize  # Start 3D visualization
```

Access services:
- **3D Visualization**: http://localhost:8050
- **Neo4j Browser**: http://localhost:7474 (neo4j/precrime2024)
- **Jupyter**: http://localhost:8888

See [DOCKER.md](DOCKER.md) for complete Docker guide.

### Option 2: Manual Installation

**Prerequisites**:
- Python 3.8+
- Neo4j 5.0+ database
- CUDA (optional, for GPU training)

**Setup**:

1. Install Python dependencies:
```bash
cd Project_Pre_Crime
pip install -r requirements.txt
```

2. Setup Neo4j database:
- Install Neo4j from https://neo4j.com/download/
- Start Neo4j service
- Update connection credentials in `src/train.py`

3. Initialize database schema:
```bash
# From Neo4j browser or using cypher-shell
cypher-shell < scripts/setup_db.cypher
```

## Usage

### Quick Start

Run the complete pipeline:
```bash
cd src
python train.py
```

This will:
1. Connect to Neo4j
2. Setup database schema
3. Generate synthetic data (100 citizens, 20 locations)
4. Train RedGAN model
5. Detect "Red Balls" (high-risk individuals)
6. Save results and checkpoints

### Individual Components

#### 1. Data Generation Only
```python
from connector import Neo4jConnector

connector = Neo4jConnector()
connector.setup_schema("../scripts/setup_db.cypher")
connector.generate_synthetic_data(
    num_citizens=100,
    num_locations=20,
    num_interactions=200,
    num_movements=300
)
```

#### 2. Model Testing
```python
from models import RedGAN
import torch

model = RedGAN(in_channels=2, hidden_channels=64, embedding_dim=32)
x = torch.randn(50, 2)
edge_index = torch.randint(0, 50, (2, 100))

output = model(x, edge_index)
print(f"Risk scores: {output['risk_scores']}")
print(f"Red balls: {output['red_balls'].sum()} detected")
```

#### 3. Custom Training
```python
from train import PreCrimeTrainer
from models import RedGAN
from connector import Neo4jConnector

connector = Neo4jConnector()
data = connector.extract_subgraph()

model = RedGAN(in_channels=2)
trainer = PreCrimeTrainer(model, connector)
trainer.train(num_epochs=50)
```

## Configuration

Key parameters in `train.py`:
- `num_epochs`: Training epochs (default: 100)
- `learning_rate`: Learning rate (default: 0.001)
- `hidden_channels`: Hidden layer size (default: 64)
- `embedding_dim`: Embedding dimension (default: 32)
- `gat_heads`: Attention heads (default: 4)

## Risk Seed Evolution

The risk seed evolves using Beta distribution:
```python
# Initial risk generation
risk = np.random.beta(alpha=2, beta=5)  # Skewed towards lower risk

# Evolution over time
new_risk = evolve_risk_seed(
    current_risk=risk,
    interactions=num_interactions,
    location_crime_rate=crime_rate,
    time_delta=days_elapsed
)
```

## Model Architecture

### GraphSAGE (Generator)
- Aggregates neighborhood information
- Creates rich node embeddings
- Multiple layers with batch normalization
- Dropout for regularization

### GAT (Discriminator)
- Multi-head attention mechanism
- Focuses on important relationships
- Predicts crime risk scores
- Identifies anomalous patterns

### RedGAN Integration
```
Input Graph → GraphSAGE → Embeddings → GAT → Risk Scores → Red Balls
```

## Output

The system produces:
1. **Model checkpoints**: Saved in `checkpoints/` directory
2. **Training history**: Loss curves and metrics
3. **Red balls CSV**: High-risk individuals detected
4. **Risk predictions**: For all nodes in the graph

## Database Queries

Example Cypher queries:

```cypher
// Find high-risk citizens
MATCH (c:Citizen)
WHERE c.risk_seed > 0.5
RETURN c.name, c.risk_seed
ORDER BY c.risk_seed DESC

// Find citizens in high-crime areas
MATCH (c:Citizen)-[:MOVES_TO]->(l:Location)
WHERE l.crime_rate > 0.3
RETURN c.name, l.name, l.crime_rate

// Find high-risk interaction networks
MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
WHERE c1.risk_seed > 0.5 AND c2.risk_seed > 0.5
RETURN c1.name, c2.name, r.interaction_type
```

## Future Enhancements

Planned features:
- Temporal graph neural networks
- Real-time risk updates
- Geographic visualization
- Multiple crime types
- Intervention strategies
- Privacy-preserving techniques

## License

This is an educational project for research purposes.

## Ethical Considerations

This system is designed for:
- Research and education
- Understanding graph neural networks
- Synthetic data analysis

**Not intended for**:
- Real-world crime prediction
- Individual profiling
- Discriminatory practices

Always ensure ethical use of predictive systems and respect privacy rights.
