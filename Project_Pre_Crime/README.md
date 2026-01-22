# Pre-Crime Prediction System

A graph neural network-based system for predicting criminal behavior using Neo4j, GraphSAGE, and GAT (Graph Attention Networks).

🐳 **Docker Ready**: Fully containerized setup for easy deployment anywhere!  
📊 **3D Visualization**: Interactive browser-based 3D graph visualization!  
🗺️ **NEW: HoloViz Dashboards**: Real-time map visualization with Panel, PyDeck, and Mapbox!  
⚡ **NEW: Fast Analytics**: DuckDB integration for lightning-fast data queries!

## Quick Start with Docker

```bash
# Setup completo en 3 comandos
make build      # Construir contenedores
make setup      # Configurar base de datos
make visualize  # Ver visualización 3D en http://localhost:8050
```

Ver [DOCKER.md](DOCKER.md) para guía completa de Docker.

## NEW: Multiple Dashboard Options 🎯

This system now offers **3 different dashboard interfaces**:

### 1. Original 3D Graph Visualization (Plotly)
```bash
python launch_dashboard.py --plotly
# or
python visualization/dashboard.py
```
Access at: http://localhost:8050

### 2. Panel HoloViz Dashboard
```bash
python launch_dashboard.py --panel
# or
python visualization/panel_dashboard.py
```
Access at: http://localhost:5006

### 3. Integrated Dashboard (Recommended) ⭐
```bash
python launch_dashboard.py --integrated
# or
python visualization/integrated_dashboard.py
```
Access at: http://localhost:5007

**Features:**
- 🗺️ PyDeck 3D maps with hexagon aggregation
- 📊 HoloViews interactive visualizations
- ⚡ DuckDB fast SQL queries
- 🌍 GeoViews geographic analysis
- 📈 Real-time risk analysis
- 🎨 Multiple visualization tabs

See [HOLOVIZ_INTEGRATION.md](HOLOVIZ_INTEGRATION.md) for complete documentation.

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
- **🆕 Panel + HoloViz** for modern interactive dashboards
- **🆕 PyDeck** for 3D geospatial map visualization
- **🆕 DuckDB** for high-performance analytics
- **🆕 Geopandas** for geospatial data operations

## Project Structure

```
Project_Pre_Crime/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Multi-container orchestration
├── Makefile               # Convenient commands
├── requirements.txt        # Python dependencies (50+ packages)
├── launch_dashboard.py     # 🆕 Dashboard launcher with menu
├── HOLOVIZ_INTEGRATION.md  # 🆕 HoloViz documentation
├── scripts/
│   └── setup_db.cypher    # Neo4j schema definition
├── src/
│   ├── connector.py       # Neo4j interaction and data generation
│   ├── models.py          # GraphSAGE and GAT models (RedGAN)
│   └── train.py           # Training pipeline
├── visualization/
│   ├── dashboard.py       # Original 3D interactive dashboard
│   ├── mapbox.py          # 🆕 Mapbox/PyDeck map visualization
│   ├── panel_dashboard.py # 🆕 Panel HoloViz dashboard
│   └── integrated_dashboard.py # 🆕 Comprehensive integrated dashboard
└── examples/
    ├── duckdb_analysis.py # 🆕 DuckDB data processing examples
    └── ...
```

## Features

### 1. Docker Containerization
- **Fully containerized**: Run anywhere with Docker
- **Docker Compose**: Neo4j + App + Jupyter in one command
- **No manual setup**: Everything automated
- **Persistent data**: Volumes for checkpoints and data

### 2. Multiple Visualization Options

#### Original 3D Interactive Visualization (Plotly)
- **Browser-based**: View at http://localhost:8050
- **Real-time 3D**: Rotate, zoom, explore the graph
- **Multiple layouts**: t-SNE, PCA, Spring layout
- **Color-coded risk**: Visual risk levels
- **Red Balls highlight**: High-risk individuals clearly marked
- **Interactive controls**: Adjust threshold, toggle edges

#### 🆕 Panel HoloViz Dashboard
- **Modern interface**: Clean, professional design
- **HoloViews plots**: Declarative visualizations
- **Multiple tabs**: Organized information display
- **GeoViews maps**: Geographic data visualization
- **Interactive tables**: Sortable, filterable data

#### 🆕 Integrated Dashboard (Recommended)
- **PyDeck 3D maps**: Stunning geospatial visualization
- **Hexagon aggregation**: Large-scale data rendering
- **DuckDB queries**: Lightning-fast analytics
- **Real-time updates**: Refresh from Neo4j
- **Multiple views**: Map, analysis, demographics, data table
- **Statistics cards**: Key metrics at a glance

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
- **🆕 Panel Dashboard**: http://localhost:5006
- **🆕 Integrated Dashboard**: http://localhost:5007

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

**Note**: The new HoloViz dashboards require additional packages (50+ dependencies).
Installation may take 5-10 minutes. If you encounter issues, install in smaller groups:

```bash
# Core dependencies first
pip install torch torch-geometric neo4j pandas numpy

# Visualization basics
pip install plotly dash jupyter

# HoloViz ecosystem
pip install panel holoviews hvplot geoviews datashader bokeh

# Geospatial
pip install geopandas pydeck shapely folium

# Data processing
pip install duckdb ibis-framework polars pyarrow

# ML optimization (optional)
pip install optuna xgboost lightgbm catboost
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
- ~~Geographic visualization~~ ✅ **IMPLEMENTED** (HoloViz + PyDeck)
- Multiple crime types
- Intervention strategies
- Privacy-preserving techniques

## Technology Stack

### Core ML & Data
- PyTorch 2.0+ for deep learning
- PyTorch Geometric for GNNs
- Neo4j 5.12+ for graph database
- Pandas, NumPy for data processing
- Scikit-learn for ML utilities

### Visualization & Dashboards
- **Plotly Dash** - Original 3D graph visualization
- **Panel** - Modern dashboard framework
- **HoloViews** - Declarative visualization
- **hvPlot** - High-level plotting
- **GeoViews** - Geographic visualization
- **PyDeck** - 3D deck.gl maps
- **Datashader** - Big data rendering
- **Bokeh** - Interactive plots

### Geospatial
- Geopandas for geospatial operations
- Shapely for geometric operations
- Folium for leaflet maps
- Mapbox for base maps

### Data Processing & Analytics
- **DuckDB** - Fast in-memory SQL
- **Ibis** - Dataframe abstraction
- **Polars** - High-performance dataframes
- **PyArrow** - Columnar data format
- **Xarray** - N-dimensional arrays

### Machine Learning Optimization
- Optuna for hyperparameter tuning
- XGBoost for gradient boosting
- LightGBM for efficient training
- CatBoost for categorical features

### Infrastructure
- Docker & Docker Compose
- Jupyter notebooks
- NetworkX for graph analysis

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
