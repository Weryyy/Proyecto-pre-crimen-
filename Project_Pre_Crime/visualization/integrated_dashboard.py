"""
Integrated Dashboard with HoloViz, Panel, and Mapbox

This module creates a comprehensive integrated dashboard combining:
- Panel for layout and interactivity
- Mapbox/PyDeck for real-time maps
- HoloViews for data visualization
- Datashader for large-scale data rendering
- DuckDB for fast data queries
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import panel as pn
import holoviews as hv
import hvplot.pandas
import geoviews as gv
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime
import os
from typing import Optional, Dict, List, Tuple
import duckdb
import colorcet as cc

# Enable Panel extensions
pn.extension('plotly', 'tabulator', 'deckgl', sizing_mode='stretch_width')
hv.extension('bokeh')

# Import project modules
try:
    from connector import Neo4jConnector
    from models import RedGAN
    import torch
except ImportError:
    print("Warning: Could not import project modules. Running in demo mode.")
    Neo4jConnector = None
    RedGAN = None
    torch = None

# Import visualization modules
from mapbox import MapboxVisualizer, extract_geographic_data


class IntegratedDashboard:
    """Integrated dashboard with all visualization capabilities"""
    
    def __init__(self, connector: Optional[object] = None, model: Optional[object] = None):
        """
        Initialize integrated dashboard
        
        Args:
            connector: Neo4j connector instance
            model: Trained model instance
        """
        self.connector = connector
        self.model = model
        self.mapbox_viz = MapboxVisualizer()
        self.last_update = None
        self.duckdb_conn = duckdb.connect(':memory:')
        
        # Create data cache
        self.citizens_df = pd.DataFrame()
        self.locations_df = pd.DataFrame()
        
        # Setup Panel template
        self.template = pn.template.MaterialTemplate(
            title='🎯 Pre-Crime Prediction System - Integrated Dashboard',
            sidebar_width=320,
            header_background='#1a1a1a',
        )
        
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch data from Neo4j or generate mock data"""
        if self.connector is None:
            return self._generate_mock_data()
        
        try:
            citizens_data, locations_data = extract_geographic_data(self.connector)
            
            self.citizens_df = pd.DataFrame(citizens_data)
            self.locations_df = pd.DataFrame(locations_data)
            
            # If no geographic data, add mock coordinates
            if 'lat' not in self.citizens_df.columns:
                coords = self.mapbox_viz.generate_mock_coordinates(len(self.citizens_df))
                self.citizens_df['lat'] = coords['lat']
                self.citizens_df['lon'] = coords['lon']
            
            # Load into DuckDB for fast queries
            self.duckdb_conn.execute("DROP TABLE IF EXISTS citizens")
            self.duckdb_conn.execute("DROP TABLE IF EXISTS locations")
            self.duckdb_conn.register('citizens', self.citizens_df)
            self.duckdb_conn.register('locations', self.locations_df)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._generate_mock_data()
        
        self.last_update = datetime.now()
        return self.citizens_df, self.locations_df
    
    def _generate_mock_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate comprehensive mock data"""
        n_citizens = 500
        
        # Generate citizens with realistic distribution
        self.citizens_df = pd.DataFrame({
            'id': range(n_citizens),
            'name': [f'Citizen_{i:04d}' for i in range(n_citizens)],
            'age': np.random.gamma(4, 10, n_citizens).astype(int) + 18,
            'risk_score': np.random.beta(2, 5, n_citizens),
            'occupation': np.random.choice([
                'Engineer', 'Teacher', 'Doctor', 'Artist', 'Driver',
                'Merchant', 'Student', 'Retired', 'Unemployed', 'Other'
            ], n_citizens, p=[0.15, 0.12, 0.08, 0.05, 0.1, 0.15, 0.1, 0.08, 0.07, 0.1]),
            'lat': 19.4326 + (np.random.rand(n_citizens) - 0.5) * 0.15,
            'lon': -99.1332 + (np.random.rand(n_citizens) - 0.5) * 0.15,
            'location_name': np.random.choice([
                'Downtown', 'Suburbs', 'Industrial Zone', 'Commercial District',
                'University Area', 'Historic Center'
            ], n_citizens)
        })
        
        # Generate locations
        self.locations_df = pd.DataFrame({
            'id': range(8),
            'name': [
                'City Center', 'North District', 'South District', 'East Zone',
                'West Zone', 'Financial District', 'Tech Park', 'Old Town'
            ],
            'lat': [19.4326, 19.4826, 19.3826, 19.4326, 19.4326, 19.4426, 19.4226, 19.4126],
            'lon': [-99.1332, -99.1332, -99.1332, -99.0832, -99.1832, -99.1432, -99.1132, -99.1232],
            'crime_rate': [0.3, 0.5, 0.4, 0.35, 0.45, 0.2, 0.15, 0.55],
            'area_type': [
                'commercial', 'residential', 'industrial', 'mixed',
                'residential', 'commercial', 'commercial', 'historic'
            ]
        })
        
        # Register with DuckDB
        self.duckdb_conn.register('citizens', self.citizens_df)
        self.duckdb_conn.register('locations', self.locations_df)
        
        self.last_update = datetime.now()
        return self.citizens_df, self.locations_df
    
    def create_pydeck_map(self, risk_threshold: float = 0.5) -> pn.pane.DeckGL:
        """Create PyDeck map visualization"""
        df = self.citizens_df.copy()
        
        # Filter by risk threshold
        high_risk_df = df[df['risk_score'] > risk_threshold]
        
        # Add color based on risk
        df['color'] = df['risk_score'].apply(self.mapbox_viz.get_risk_color)
        
        # Create layers
        scatter_layer = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_radius=100,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True
        )
        
        hexagon_layer = pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            get_elevation_weight='risk_score',
            elevation_scale=100,
            elevation_range=[0, 500],
            radius=150,
            coverage=0.8,
            extruded=True,
            pickable=True,
            auto_highlight=True
        )
        
        view_state = pdk.ViewState(
            latitude=df['lat'].mean(),
            longitude=df['lon'].mean(),
            zoom=11,
            pitch=45,
            bearing=0
        )
        
        deck = pdk.Deck(
            layers=[hexagon_layer, scatter_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip={
                'html': '<b>{name}</b><br/>Risk: {risk_score:.3f}<br/>Age: {age}<br/>Occupation: {occupation}',
                'style': {'color': 'white', 'backgroundColor': '#333'}
            }
        )
        
        return pn.pane.DeckGL(deck, sizing_mode='stretch_both', height=600)
    
    def create_statistics_cards(self) -> pn.GridSpec:
        """Create statistics cards with key metrics"""
        df = self.citizens_df
        
        # Use DuckDB for fast aggregations
        stats = self.duckdb_conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN risk_score > 0.7 THEN 1 ELSE 0 END) as high_risk,
                SUM(CASE WHEN risk_score BETWEEN 0.5 AND 0.7 THEN 1 ELSE 0 END) as medium_risk,
                SUM(CASE WHEN risk_score < 0.5 THEN 1 ELSE 0 END) as low_risk,
                AVG(risk_score) as avg_risk,
                MAX(risk_score) as max_risk,
                AVG(age) as avg_age
            FROM citizens
        """).fetchone()
        
        total, high_risk, medium_risk, low_risk, avg_risk, max_risk, avg_age = stats
        
        # Create grid layout
        grid = pn.GridSpec(sizing_mode='stretch_both', height=200)
        
        grid[0, 0] = pn.indicators.Number(
            value=total, name='Total Citizens', format='{value:,.0f}',
            default_color='#4A90E2', font_size='32pt', title_size='14pt'
        )
        
        grid[0, 1] = pn.indicators.Number(
            value=high_risk, name='🔴 High Risk', format='{value:,.0f}',
            default_color='#E74C3C', font_size='32pt', title_size='14pt'
        )
        
        grid[0, 2] = pn.indicators.Number(
            value=medium_risk, name='🟡 Medium Risk', format='{value:,.0f}',
            default_color='#F39C12', font_size='32pt', title_size='14pt'
        )
        
        grid[0, 3] = pn.indicators.Number(
            value=low_risk, name='🟢 Low Risk', format='{value:,.0f}',
            default_color='#2ECC71', font_size='32pt', title_size='14pt'
        )
        
        return grid
    
    def create_risk_analysis_panel(self) -> pn.Column:
        """Create comprehensive risk analysis visualizations"""
        df = self.citizens_df
        
        # Risk distribution histogram
        hist = df.hvplot.hist(
            'risk_score', bins=40, title='Risk Score Distribution',
            color='#E74C3C', alpha=0.7, responsive=True, height=300
        )
        
        # Risk by occupation
        occupation_risk = self.duckdb_conn.execute("""
            SELECT occupation, AVG(risk_score) as avg_risk, COUNT(*) as count
            FROM citizens
            GROUP BY occupation
            ORDER BY avg_risk DESC
        """).df()
        
        occupation_plot = occupation_risk.hvplot.barh(
            x='occupation', y='avg_risk', title='Average Risk by Occupation',
            color='#F39C12', height=300, responsive=True
        )
        
        # Age vs Risk scatter
        scatter = df.hvplot.scatter(
            x='age', y='risk_score', c='risk_score',
            cmap='fire', alpha=0.5, size=40,
            title='Age vs Risk Score', responsive=True, height=300
        )
        
        return pn.Column(
            pn.pane.Markdown('## 📊 Risk Analysis'),
            pn.Row(hist, scatter),
            occupation_plot
        )
    
    def create_geographic_analysis_panel(self) -> pn.Column:
        """Create geographic analysis with GeoViews"""
        df = self.citizens_df
        locs = self.locations_df
        
        # Points plot with risk coloring
        points = df.hvplot.points(
            'lon', 'lat', c='risk_score', cmap='fire',
            title='Geographic Risk Distribution', tiles='CartoLight',
            alpha=0.6, size=50, responsive=True, height=500
        )
        
        return pn.Column(
            pn.pane.Markdown('## 🗺️ Geographic Analysis'),
            points
        )
    
    def create_datatable_panel(self, risk_filter: float = 0.0) -> pn.Column:
        """Create interactive data table with DuckDB filtering"""
        # Query with DuckDB
        query = f"""
            SELECT name, age, occupation, risk_score, location_name
            FROM citizens
            WHERE risk_score >= {risk_filter}
            ORDER BY risk_score DESC
            LIMIT 100
        """
        df = self.duckdb_conn.execute(query).df()
        
        # Format risk score
        df['risk_score'] = df['risk_score'].round(3)
        
        table = pn.widgets.Tabulator(
            df, page_size=20, pagination='remote',
            layout='fit_columns', theme='midnight',
            sizing_mode='stretch_both', height=500
        )
        
        return pn.Column(
            pn.pane.Markdown('## 📋 High-Risk Citizens'),
            table
        )
    
    def create_sidebar_controls(self) -> List:
        """Create sidebar with interactive controls"""
        # Widgets
        refresh_btn = pn.widgets.Button(
            name='🔄 Refresh Data', button_type='primary', width=280
        )
        
        risk_threshold = pn.widgets.FloatSlider(
            name='Risk Threshold', start=0, end=1, step=0.05,
            value=0.5, width=280
        )
        
        occupation_select = pn.widgets.MultiChoice(
            name='Filter by Occupation',
            options=list(self.citizens_df['occupation'].unique()) if not self.citizens_df.empty else [],
            width=280
        )
        
        show_hexagons = pn.widgets.Checkbox(
            name='Show Hexagon Aggregation', value=True, width=280
        )
        
        # Info panel
        info = pn.pane.Markdown(f"""
        ### 📊 Dashboard Info
        
        **Last Update:** {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never'}
        
        **Citizens:** {len(self.citizens_df):,}
        
        **Locations:** {len(self.locations_df):,}
        
        ---
        
        ### 🛠️ Technologies
        
        - **Panel** - Dashboard framework
        - **HoloViews** - Visualization
        - **PyDeck** - 3D maps
        - **DuckDB** - Fast queries
        - **Datashader** - Big data
        - **GeoViews** - Geo visualization
        
        ---
        
        ### 📖 Controls
        
        Use the controls above to:
        - Filter by risk threshold
        - Select occupations
        - Toggle visualization modes
        - Refresh data from Neo4j
        """, width=280)
        
        return [
            pn.pane.Markdown('# ⚙️ Controls'),
            pn.layout.Divider(),
            refresh_btn,
            risk_threshold,
            occupation_select,
            show_hexagons,
            pn.layout.Divider(),
            info
        ]
    
    def build(self):
        """Build the complete integrated dashboard"""
        # Fetch initial data
        self.fetch_data()
        
        # Create main components
        stats_cards = self.create_statistics_cards()
        map_panel = self.create_pydeck_map()
        risk_panel = self.create_risk_analysis_panel()
        geo_panel = self.create_geographic_analysis_panel()
        table_panel = self.create_datatable_panel()
        
        # Create tabs
        tabs = pn.Tabs(
            ('🗺️ 3D Map', pn.Column(stats_cards, map_panel)),
            ('📊 Risk Analysis', pn.Column(stats_cards, risk_panel)),
            ('🌍 Geographic', pn.Column(stats_cards, geo_panel)),
            ('📋 Data Table', pn.Column(stats_cards, table_panel)),
            dynamic=True,
            sizing_mode='stretch_both'
        )
        
        # Configure template
        self.template.sidebar.extend(self.create_sidebar_controls())
        self.template.main.append(tabs)
        
        return self.template


def main():
    """Launch the integrated dashboard"""
    print("=" * 80)
    print("🎯 Pre-Crime Prediction System - Integrated Dashboard")
    print("=" * 80)
    print("\n🚀 Initializing components...")
    
    # Try to connect to Neo4j
    connector = None
    model = None
    
    if Neo4jConnector is not None:
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'precrime2024')
            
            connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            print("✅ Connected to Neo4j")
        except Exception as e:
            print(f"⚠️  Could not connect to Neo4j: {e}")
            print("⚠️  Running in demo mode with mock data")
    else:
        print("⚠️  Running in demo mode with mock data")
    
    # Create dashboard
    dashboard = IntegratedDashboard(connector, model)
    template = dashboard.build()
    
    print("\n✅ Dashboard initialized successfully!")
    print("\n" + "=" * 80)
    print("📡 Access the dashboard at:")
    print("   👉 http://localhost:5007")
    print("=" * 80)
    print("\n💡 Features:")
    print("   • Real-time 3D map with PyDeck")
    print("   • Interactive risk analysis with HoloViews")
    print("   • Fast queries with DuckDB")
    print("   • Geographic visualization with GeoViews")
    print("   • Comprehensive data tables")
    print("\n🛑 Press Ctrl+C to stop the server\n")
    
    # Serve dashboard
    pn.serve(template, port=5007, show=True, title='Pre-Crime Integrated Dashboard')


if __name__ == '__main__':
    main()
