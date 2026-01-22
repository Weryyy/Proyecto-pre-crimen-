"""
Panel Dashboard with HoloViz Integration

This module creates a comprehensive real-time dashboard using Panel, HoloViews, hvPlot,
and other HoloViz ecosystem tools for visualizing the pre-crime prediction system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import panel as pn
import holoviews as hv
import hvplot.pandas
import geoviews as gv
import geoviews.tile_sources as gts
from holoviews import opts
import pandas as pd
import numpy as np
import colorcet as cc
from datetime import datetime
import os
from typing import Optional, Dict, List, Tuple

# Enable Panel extensions
pn.extension('plotly', 'tabulator', sizing_mode='stretch_width')
hv.extension('bokeh')
gv.extension('bokeh')

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


class HoloVizDashboard:
    """Comprehensive Panel dashboard with HoloViz integration"""
    
    def __init__(self, connector: Optional[object] = None, model: Optional[object] = None):
        """
        Initialize HoloViz dashboard
        
        Args:
            connector: Neo4j connector instance
            model: Trained model instance
        """
        self.connector = connector
        self.model = model
        self.data_cache = {}
        self.last_update = None
        
        # Setup Panel template
        self.template = pn.template.FastListTemplate(
            title='Pre-Crime Prediction Dashboard - HoloViz',
            theme='dark',
            sidebar_width=300,
            header_background='#1f1f1f',
            accent_base_color='#ff0000'
        )
        
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch data from Neo4j
        
        Returns:
            Tuple of (citizens_df, locations_df)
        """
        if self.connector is None:
            # Return mock data for demo
            return self._generate_mock_data()
        
        with self.connector.driver.session() as session:
            # Fetch citizens data
            citizens_query = """
            MATCH (c:Citizen)
            OPTIONAL MATCH (c)-[m:MOVES_TO]->(l:Location)
            WITH c, l, m
            ORDER BY m.timestamp DESC
            WITH c, COLLECT(l)[0] as current_location
            RETURN c.id as id, c.name as name, c.age as age, 
                   c.risk_seed as risk_score, c.occupation as occupation,
                   COALESCE(current_location.lat, 19.4326 + (rand() - 0.5) * 0.1) as lat,
                   COALESCE(current_location.lon, -99.1332 + (rand() - 0.5) * 0.1) as lon,
                   COALESCE(current_location.name, 'Unknown') as location_name
            """
            result = session.run(citizens_query)
            citizens_data = [dict(record) for record in result]
            citizens_df = pd.DataFrame(citizens_data)
            
            # Fetch locations data
            locations_query = """
            MATCH (l:Location)
            RETURN l.id as id, l.name as name, 
                   l.lat as lat, l.lon as lon,
                   l.crime_rate as crime_rate, l.area_type as area_type
            """
            result = session.run(locations_query)
            locations_data = [dict(record) for record in result]
            locations_df = pd.DataFrame(locations_data)
        
        self.last_update = datetime.now()
        return citizens_df, locations_df
    
    def _generate_mock_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate mock data for demo purposes"""
        n_citizens = 200
        
        # Mock citizens
        citizens_df = pd.DataFrame({
            'id': range(n_citizens),
            'name': [f'Citizen_{i}' for i in range(n_citizens)],
            'age': np.random.randint(18, 80, n_citizens),
            'risk_score': np.random.beta(2, 5, n_citizens),
            'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Other'], n_citizens),
            'lat': 19.4326 + (np.random.rand(n_citizens) - 0.5) * 0.1,
            'lon': -99.1332 + (np.random.rand(n_citizens) - 0.5) * 0.1,
            'location_name': np.random.choice(['Downtown', 'Suburbs', 'Industrial', 'Commercial'], n_citizens)
        })
        
        # Mock locations
        locations_df = pd.DataFrame({
            'id': range(5),
            'name': ['City Center', 'North District', 'South District', 'East Zone', 'West Zone'],
            'lat': [19.4326, 19.4826, 19.3826, 19.4326, 19.4326],
            'lon': [-99.1332, -99.1332, -99.1332, -99.0832, -99.1832],
            'crime_rate': [0.3, 0.5, 0.4, 0.35, 0.45],
            'area_type': ['commercial', 'residential', 'industrial', 'mixed', 'residential']
        })
        
        self.last_update = datetime.now()
        return citizens_df, locations_df
    
    def create_map_panel(self, citizens_df: pd.DataFrame, locations_df: pd.DataFrame) -> pn.Column:
        """
        Create interactive map panel with GeoViews
        
        Args:
            citizens_df: Citizens DataFrame
            locations_df: Locations DataFrame
            
        Returns:
            Panel Column with map
        """
        # Create GeoViews points for citizens
        citizens_points = gv.Points(
            citizens_df,
            kdims=['lon', 'lat'],
            vdims=['name', 'risk_score', 'age', 'occupation']
        ).opts(
            opts.Points(
                color='risk_score',
                cmap='fire',
                size=8,
                alpha=0.7,
                tools=['hover'],
                colorbar=True,
                width=800,
                height=600,
                title='Citizens Risk Map'
            )
        )
        
        # Create GeoViews points for locations
        locations_points = gv.Points(
            locations_df,
            kdims=['lon', 'lat'],
            vdims=['name', 'crime_rate', 'area_type']
        ).opts(
            opts.Points(
                color='blue',
                size=15,
                marker='triangle',
                alpha=0.8,
                tools=['hover']
            )
        )
        
        # Combine with tile source
        map_plot = gts.CartoLight * locations_points * citizens_points
        
        return pn.Column(
            pn.pane.Markdown('## Geographic Risk Distribution'),
            pn.pane.HoloViews(map_plot, sizing_mode='stretch_both')
        )
    
    def create_risk_distribution_panel(self, citizens_df: pd.DataFrame) -> pn.Column:
        """
        Create risk distribution visualizations
        
        Args:
            citizens_df: Citizens DataFrame
            
        Returns:
            Panel Column with risk visualizations
        """
        # Risk histogram
        hist_plot = citizens_df.hvplot.hist(
            y='risk_score',
            bins=30,
            title='Risk Score Distribution',
            color='red',
            alpha=0.7,
            responsive=True,
            height=300
        )
        
        # Risk by occupation
        occupation_plot = citizens_df.groupby('occupation')['risk_score'].mean().hvplot.bar(
            title='Average Risk by Occupation',
            rot=45,
            color='orange',
            responsive=True,
            height=300
        )
        
        return pn.Column(
            pn.pane.Markdown('## Risk Analysis'),
            pn.Row(hist_plot, occupation_plot)
        )
    
    def create_statistics_panel(self, citizens_df: pd.DataFrame, locations_df: pd.DataFrame) -> pn.Column:
        """
        Create statistics panel with key metrics
        
        Args:
            citizens_df: Citizens DataFrame
            locations_df: Locations DataFrame
            
        Returns:
            Panel Column with statistics
        """
        total_citizens = len(citizens_df)
        high_risk = len(citizens_df[citizens_df['risk_score'] > 0.7])
        avg_risk = citizens_df['risk_score'].mean()
        max_risk = citizens_df['risk_score'].max()
        
        # Create indicator panels
        indicators = pn.Row(
            pn.indicators.Number(
                value=total_citizens,
                name='Total Citizens',
                format='{value:,.0f}',
                default_color='blue',
                font_size='24pt'
            ),
            pn.indicators.Number(
                value=high_risk,
                name='High Risk (>0.7)',
                format='{value:,.0f}',
                default_color='red',
                font_size='24pt'
            ),
            pn.indicators.Gauge(
                value=int(avg_risk * 100),
                name='Average Risk',
                bounds=(0, 100),
                colors=[(0.3, 'green'), (0.5, 'gold'), (1, 'red')],
                format='{value}%'
            ),
            pn.indicators.Gauge(
                value=int(max_risk * 100),
                name='Maximum Risk',
                bounds=(0, 100),
                colors=[(0.3, 'green'), (0.5, 'gold'), (1, 'red')],
                format='{value}%'
            )
        )
        
        return pn.Column(
            pn.pane.Markdown('## Key Metrics'),
            indicators
        )
    
    def create_datatable_panel(self, citizens_df: pd.DataFrame) -> pn.Column:
        """
        Create interactive data table
        
        Args:
            citizens_df: Citizens DataFrame
            
        Returns:
            Panel Column with data table
        """
        # Sort by risk score
        citizens_df_sorted = citizens_df.sort_values('risk_score', ascending=False)
        
        # Create Tabulator widget
        table = pn.widgets.Tabulator(
            citizens_df_sorted[['name', 'age', 'occupation', 'risk_score', 'location_name']],
            page_size=20,
            pagination='remote',
            layout='fit_columns',
            theme='midnight',
            configuration={
                'columnDefaults': {
                    'headerSort': True
                }
            }
        )
        
        return pn.Column(
            pn.pane.Markdown('## High-Risk Citizens Table'),
            table
        )
    
    def create_temporal_panel(self, citizens_df: pd.DataFrame) -> pn.Column:
        """
        Create temporal analysis panel
        
        Args:
            citizens_df: Citizens DataFrame
            
        Returns:
            Panel Column with temporal visualizations
        """
        # Age vs Risk scatter plot
        scatter = citizens_df.hvplot.scatter(
            x='age',
            y='risk_score',
            c='risk_score',
            cmap='coolwarm',
            size=50,
            alpha=0.6,
            title='Age vs Risk Score',
            responsive=True,
            height=400
        )
        
        return pn.Column(
            pn.pane.Markdown('## Demographic Analysis'),
            scatter
        )
    
    def create_sidebar(self) -> List:
        """
        Create sidebar with controls
        
        Returns:
            List of Panel components for sidebar
        """
        # Refresh button
        refresh_button = pn.widgets.Button(
            name='🔄 Refresh Data',
            button_type='primary',
            width=250
        )
        
        # Risk threshold slider
        risk_threshold = pn.widgets.FloatSlider(
            name='Risk Threshold',
            start=0,
            end=1,
            step=0.05,
            value=0.5,
            width=250
        )
        
        # Filter by occupation
        occupation_filter = pn.widgets.MultiChoice(
            name='Filter Occupations',
            options=['Engineer', 'Teacher', 'Doctor', 'Artist', 'Other'],
            value=[],
            width=250
        )
        
        # Last update info
        update_info = pn.pane.Markdown(
            f'**Last Update:** {self.last_update.strftime("%Y-%m-%d %H:%M:%S") if self.last_update else "Never"}',
            width=250
        )
        
        sidebar = [
            pn.pane.Markdown('# Controls'),
            pn.layout.Divider(),
            refresh_button,
            risk_threshold,
            occupation_filter,
            pn.layout.Divider(),
            update_info,
            pn.layout.Divider(),
            pn.pane.Markdown('''
            ### About
            Real-time dashboard for pre-crime prediction system using:
            - **Panel** for dashboard
            - **HoloViews** for visualization
            - **GeoViews** for maps
            - **hvPlot** for interactive plots
            - **Datashader** for large data
            ''')
        ]
        
        return sidebar
    
    def build_dashboard(self):
        """Build and configure the complete dashboard"""
        # Fetch data
        citizens_df, locations_df = self.fetch_data()
        
        # Create panels
        stats_panel = self.create_statistics_panel(citizens_df, locations_df)
        map_panel = self.create_map_panel(citizens_df, locations_df)
        risk_panel = self.create_risk_distribution_panel(citizens_df)
        table_panel = self.create_datatable_panel(citizens_df)
        temporal_panel = self.create_temporal_panel(citizens_df)
        
        # Create tabs for different views
        tabs = pn.Tabs(
            ('Overview', pn.Column(stats_panel, risk_panel)),
            ('Geographic Map', map_panel),
            ('Demographic Analysis', temporal_panel),
            ('Data Table', table_panel),
            sizing_mode='stretch_both'
        )
        
        # Configure template
        self.template.sidebar.extend(self.create_sidebar())
        self.template.main.append(tabs)
        
        return self.template


def create_panel_dashboard(connector: Optional[object] = None, model: Optional[object] = None):
    """
    Create Panel dashboard instance
    
    Args:
        connector: Neo4j connector instance
        model: Trained model instance
        
    Returns:
        Panel dashboard template
    """
    dashboard = HoloVizDashboard(connector, model)
    return dashboard.build_dashboard()


def main():
    """Run the Panel dashboard"""
    print("=" * 70)
    print("Starting Pre-Crime Prediction Panel Dashboard (HoloViz)")
    print("=" * 70)
    
    # Try to connect to Neo4j
    connector = None
    model = None
    
    if Neo4jConnector is not None:
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'precrime2024')
            
            connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            print("✓ Connected to Neo4j")
            
            # Try to load model if available
            checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
            if checkpoint_dir.exists() and torch is not None:
                checkpoints = list(checkpoint_dir.glob('*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    print(f"✓ Loading model from {latest_checkpoint}")
                    try:
                        data = connector.extract_subgraph()
                        model = RedGAN(in_channels=data.x.shape[1])
                        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        print("✓ Model loaded successfully")
                    except Exception as e:
                        print(f"⚠ Could not load model: {e}")
        except Exception as e:
            print(f"⚠ Could not connect to Neo4j: {e}")
            print("Running in demo mode with mock data")
    else:
        print("⚠ Running in demo mode with mock data")
    
    # Create dashboard
    dashboard = create_panel_dashboard(connector, model)
    
    print("\n✓ Dashboard initialized")
    print("\nAccess the dashboard at:")
    print("  → http://localhost:5006")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    # Serve dashboard
    pn.serve(dashboard, port=5006, show=True, title='Pre-Crime Dashboard')


if __name__ == '__main__':
    main()
