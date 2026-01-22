# 🗺️ HoloViz Dashboard Integration Guide

## Overview

The Pre-Crime Prediction System has been enhanced with comprehensive HoloViz ecosystem integration, providing multiple dashboard options for real-time data visualization and analysis.

## New Components

### 1. Mapbox Integration (`visualization/mapbox.py`)

Real-time map visualization using Mapbox, PyDeck, and geopandas.

**Features:**
- Geographic visualization of citizens and locations
- Risk-based color coding
- Interactive markers and tooltips
- Hexagon aggregation for large-scale data
- Arc visualization for movement patterns
- Heatmap overlay

**Key Technologies:**
- PyDeck for 3D deck.gl maps
- Geopandas for geospatial data
- Shapely for geometric operations
- Mapbox for base maps

### 2. Panel HoloViz Dashboard (`visualization/panel_dashboard.py`)

Modern dashboard using the HoloViz ecosystem.

**Features:**
- Interactive Panel-based interface
- HoloViews visualizations
- GeoViews for geographic analysis
- Real-time data updates
- Interactive controls and filters
- Multi-tab layout

**Key Technologies:**
- Panel for dashboard framework
- HoloViews for declarative viz
- hvPlot for interactive plots
- GeoViews for geographic data
- Bokeh for rendering

### 3. Integrated Dashboard (`visualization/integrated_dashboard.py`)

Comprehensive dashboard combining all features.

**Features:**
- PyDeck 3D maps with hexagon aggregation
- DuckDB for fast SQL queries
- Multiple visualization tabs
- Interactive risk analysis
- Geographic distribution analysis
- Real-time filtering
- Statistics cards with key metrics

**Key Technologies:**
- Panel + HoloViz ecosystem
- PyDeck for 3D maps
- DuckDB for analytics
- Datashader for big data
- GeoViews for maps

### 4. Advanced Data Processing (`examples/duckdb_analysis.py`)

Data processing and analytics using DuckDB and Ibis.

**Features:**
- Fast SQL queries on large datasets
- Risk pattern analysis
- Interaction network analysis
- Movement pattern detection
- High-risk cluster identification
- Time series analysis
- Export to Parquet format

**Key Technologies:**
- DuckDB for in-memory analytics
- Ibis for dataframe abstraction
- Parquet for efficient storage
- Pandas for data manipulation

## Installation

### Update Dependencies

```bash
cd Project_Pre_Crime
pip install -r requirements.txt
```

New dependencies include:
- **HoloViz Ecosystem**: panel, holoviews, hvplot, geoviews, datashader, colorcet
- **Geospatial**: geopandas, mapbox, pydeck, folium, shapely
- **Data Processing**: duckdb, ibis-framework, polars, pyarrow, xarray
- **ML Optimization**: optuna, lightgbm, xgboost, catboost
- **Network Analysis**: networkx

## Usage

### Quick Start - Interactive Launcher

```bash
python launch_dashboard.py
```

This will show an interactive menu with all dashboard options.

### Launch Specific Dashboards

#### 1. Original Plotly Dashboard
```bash
python launch_dashboard.py --plotly
# or
python visualization/dashboard.py
```
Access at: http://localhost:8050

#### 2. Panel HoloViz Dashboard
```bash
python launch_dashboard.py --panel
# or
python visualization/panel_dashboard.py
```
Access at: http://localhost:5006

#### 3. Integrated Dashboard (Recommended)
```bash
python launch_dashboard.py --integrated
# or
python visualization/integrated_dashboard.py
```
Access at: http://localhost:5007

#### 4. Test Mapbox Integration
```bash
python launch_dashboard.py --mapbox
# or
python visualization/mapbox.py
```
Generates HTML file in `examples/mapbox_test.html`

### Command-Line Options

```bash
# Show all available dashboards
python launch_dashboard.py --all

# Launch specific dashboard
python launch_dashboard.py --integrated

# Show help
python launch_dashboard.py --help
```

## Dashboard Features

### Integrated Dashboard (Port 5007)

#### Main Tabs:

**🗺️ 3D Map Tab**
- PyDeck 3D map with hexagon aggregation
- Interactive zoom, pan, and rotation
- Risk-based elevation and coloring
- Citizen markers with tooltips
- Real-time updates

**📊 Risk Analysis Tab**
- Risk score distribution histogram
- Risk by occupation bar chart
- Age vs risk scatter plot
- Statistical summaries

**🌍 Geographic Tab**
- GeoViews map with tile layers
- Points colored by risk score
- Location markers
- Interactive hover information

**📋 Data Table Tab**
- Sortable, filterable table
- Top 100 high-risk citizens
- Pagination support
- Export capabilities

#### Sidebar Controls:

- **Refresh Data**: Reload from Neo4j
- **Risk Threshold**: Filter display (0-1)
- **Filter by Occupation**: Multi-select filter
- **Show Hexagon Aggregation**: Toggle 3D hexagons
- **Dashboard Info**: Stats and metadata

#### Statistics Cards:

- Total Citizens count
- High Risk (>0.7) count
- Medium Risk (0.5-0.7) count
- Low Risk (<0.5) count

### Panel HoloViz Dashboard (Port 5006)

Features similar layout with emphasis on HoloViews-based visualizations:
- GeoViews map integration
- hvPlot interactive charts
- Tabulator data tables
- Panel indicators and gauges

### Original Plotly Dashboard (Port 8050)

The original 3D graph visualization:
- t-SNE/PCA/Spring layouts
- 3D interactive graph
- Red Ball detection
- Network connections

## Data Processing

### Using DuckDB for Analytics

```python
from examples.duckdb_analysis import DataProcessor

# Initialize processor
processor = DataProcessor(connector)

# Load data from Neo4j
data = processor.load_data_from_neo4j()
processor.setup_duckdb_tables(data)

# Run analyses
risk_patterns = processor.analyze_risk_patterns()
interactions = processor.analyze_interaction_networks()
movements = processor.analyze_movement_patterns()
clusters = processor.identify_high_risk_clusters()

# Export results
processor.export_to_parquet(output_dir)
```

### Run Analysis Script

```bash
python examples/duckdb_analysis.py
```

This will:
1. Load data from Neo4j (or generate mock data)
2. Create DuckDB tables
3. Run multiple analyses
4. Display results
5. Export to Parquet files in `data/processed/`

## Advanced Features

### Geographic Coordinates

Citizens and locations now support geographic coordinates:
- Latitude and longitude fields
- GeoDataFrame support
- Spatial operations with shapely
- Map projections and transformations

### Real-Time Updates

Dashboards support real-time data updates:
- Refresh button to reload from Neo4j
- Automatic cache invalidation
- Efficient data loading

### Fast Queries with DuckDB

DuckDB provides SQL interface for fast analytics:
- Window functions
- Aggregations
- Joins
- Time series operations
- Export to Parquet

### Large-Scale Visualization

Datashader and PyDeck for big data:
- Hexagon aggregation
- Heatmaps
- Level of detail rendering
- Efficient memory usage

## Configuration

### Environment Variables

```bash
# Neo4j connection
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="precrime2024"

# Optional: Mapbox token for enhanced maps
export MAPBOX_TOKEN="your_mapbox_token_here"
```

### Customize Dashboard

Edit dashboard files to customize:
- Color schemes (use colorcet palettes)
- Layout and sizing
- Data refresh intervals
- Filter options
- Visualization types

## Technology Stack

### HoloViz Ecosystem
- **Panel**: Dashboard framework and app templates
- **HoloViews**: Declarative data visualization
- **hvPlot**: High-level plotting API
- **GeoViews**: Geographic data visualization
- **Datashader**: Large data rendering
- **Colorcet**: Perceptually uniform colormaps
- **Param**: Parameter declarations
- **Bokeh**: Interactive visualization library

### Geospatial
- **Geopandas**: Geospatial data operations
- **PyDeck**: 3D deck.gl visualizations
- **Mapbox**: Base map tiles
- **Shapely**: Geometric operations
- **Folium**: Alternative leaflet maps

### Data Processing
- **DuckDB**: Fast in-memory SQL analytics
- **Ibis**: Dataframe abstraction layer
- **Polars**: Fast dataframe library
- **PyArrow**: Columnar data format
- **Xarray**: N-dimensional arrays

### Machine Learning
- **Optuna**: Hyperparameter optimization
- **XGBoost**: Gradient boosting
- **LightGBM**: Light gradient boosting
- **CatBoost**: Categorical boosting

## Troubleshooting

### Dashboard Not Loading

```bash
# Check if required packages are installed
pip list | grep panel
pip list | grep holoviews

# Reinstall if needed
pip install -r requirements.txt
```

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker-compose ps neo4j

# Check connection
python -c "from src.connector import Neo4jConnector; c = Neo4jConnector(); print('OK')"
```

### Port Already in Use

```bash
# Find process using port
lsof -i :5007

# Kill process
kill -9 <PID>

# Or use different port in code
```

### Mock Data Mode

If Neo4j is not available, dashboards automatically use mock data:
- 200-1000 synthetic citizens
- Realistic risk distributions
- Random geographic coordinates
- Sample interactions and movements

## Examples

### Custom Analysis with DuckDB

```python
import duckdb

# Connect to data
conn = duckdb.connect(':memory:')
conn.register('citizens', citizens_df)

# Custom query
result = conn.execute("""
    SELECT occupation, AVG(risk_score) as avg_risk
    FROM citizens
    WHERE age > 30
    GROUP BY occupation
    ORDER BY avg_risk DESC
""").df()

print(result)
```

### Export to Different Formats

```python
# Parquet (recommended for large data)
df.to_parquet('output.parquet')

# CSV
df.to_csv('output.csv')

# JSON
df.to_json('output.json')

# Excel
df.to_excel('output.xlsx')
```

### Custom Map Visualization

```python
from visualization.mapbox import MapboxVisualizer

viz = MapboxVisualizer()
citizens_gdf, locations_gdf = viz.create_geodataframe(citizens_data, locations_data)

# Create custom map
deck = viz.create_pydeck_map(citizens_gdf, locations_gdf, show_heatmap=True)
deck.to_html('custom_map.html')
```

## Performance Tips

1. **Use DuckDB for large datasets**: Much faster than pandas for aggregations
2. **Enable hexagon aggregation**: Better performance with many points
3. **Limit table page size**: Reduce initial render time
4. **Use Parquet format**: Faster I/O than CSV
5. **Filter data early**: Apply filters in SQL queries

## Future Enhancements

Potential additions:
- WebSocket for real-time streaming
- DECK.GL advanced layers (trips, paths)
- Time series animation
- Custom colormaps and themes
- Export dashboard as static HTML
- Integration with Rapids for GPU acceleration
- Kubernetes deployment templates

## Resources

- [Panel Documentation](https://panel.holoviz.org)
- [HoloViews Documentation](https://holoviews.org)
- [GeoViews Documentation](https://geoviews.org)
- [PyDeck Documentation](https://deckgl.readthedocs.io)
- [DuckDB Documentation](https://duckdb.org/docs)

## Support

For issues or questions:
1. Check documentation above
2. Review example scripts
3. Check logs for error messages
4. Open GitHub issue with details

---

**Version:** 2.0.0  
**Last Updated:** 2026-01-22  
**Status:** ✅ Production Ready
