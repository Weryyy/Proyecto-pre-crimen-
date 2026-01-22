"""
Mapbox Integration for Real-Time Map Visualization

This module provides real-time map visualization using Mapbox and various mapping libraries.
It displays citizens, locations, and risk levels on an interactive map.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pydeck as pdk
from typing import Dict, List, Tuple, Optional
import os


class MapboxVisualizer:
    """Real-time map visualization with Mapbox integration"""
    
    def __init__(self, mapbox_token: Optional[str] = None):
        """
        Initialize Mapbox visualizer
        
        Args:
            mapbox_token: Mapbox API token (optional, can use env var)
        """
        self.mapbox_token = mapbox_token or os.getenv('MAPBOX_TOKEN', '')
        self.default_center = [-99.1332, 19.4326]  # Mexico City coordinates
        self.default_zoom = 11
        
    def create_geodataframe(self, citizens_data: List[Dict], locations_data: List[Dict]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Create GeoDataFrames from citizen and location data
        
        Args:
            citizens_data: List of citizen dictionaries with location info
            locations_data: List of location dictionaries
            
        Returns:
            Tuple of (citizens_gdf, locations_gdf)
        """
        # Create citizens GeoDataFrame
        if citizens_data:
            citizens_df = pd.DataFrame(citizens_data)
            geometry = [Point(xy) for xy in zip(citizens_df['lon'], citizens_df['lat'])]
            citizens_gdf = gpd.GeoDataFrame(citizens_df, geometry=geometry, crs='EPSG:4326')
        else:
            citizens_gdf = gpd.GeoDataFrame()
        
        # Create locations GeoDataFrame
        if locations_data:
            locations_df = pd.DataFrame(locations_data)
            geometry = [Point(xy) for xy in zip(locations_df['lon'], locations_df['lat'])]
            locations_gdf = gpd.GeoDataFrame(locations_df, geometry=geometry, crs='EPSG:4326')
        else:
            locations_gdf = gpd.GeoDataFrame()
        
        return citizens_gdf, locations_gdf
    
    def get_risk_color(self, risk_score: float) -> List[int]:
        """
        Get RGB color based on risk score
        
        Args:
            risk_score: Risk score between 0 and 1
            
        Returns:
            RGB color as [R, G, B, A]
        """
        if risk_score < 0.3:
            # Low risk: Green
            return [0, 255, 0, 160]
        elif risk_score < 0.5:
            # Medium risk: Yellow
            return [255, 255, 0, 160]
        elif risk_score < 0.7:
            # High risk: Orange
            return [255, 165, 0, 200]
        else:
            # Very high risk: Red
            return [255, 0, 0, 220]
    
    def create_pydeck_map(self, citizens_gdf: gpd.GeoDataFrame, 
                         locations_gdf: gpd.GeoDataFrame,
                         show_heatmap: bool = True,
                         show_connections: bool = False) -> pdk.Deck:
        """
        Create PyDeck map visualization
        
        Args:
            citizens_gdf: GeoDataFrame of citizens
            locations_gdf: GeoDataFrame of locations
            show_heatmap: Whether to show risk heatmap
            show_connections: Whether to show connections between citizens
            
        Returns:
            PyDeck Deck object
        """
        layers = []
        
        # Location markers layer
        if not locations_gdf.empty:
            locations_layer = pdk.Layer(
                'ScatterplotLayer',
                data=locations_gdf,
                get_position='[lon, lat]',
                get_radius=200,
                get_fill_color='[100, 100, 255, 180]',
                pickable=True,
                auto_highlight=True
            )
            layers.append(locations_layer)
        
        # Citizens markers layer with risk colors
        if not citizens_gdf.empty:
            # Add color column based on risk
            citizens_gdf['color'] = citizens_gdf['risk_score'].apply(self.get_risk_color)
            
            citizens_layer = pdk.Layer(
                'ScatterplotLayer',
                data=citizens_gdf,
                get_position='[lon, lat]',
                get_radius=150,
                get_fill_color='color',
                pickable=True,
                auto_highlight=True
            )
            layers.append(citizens_layer)
            
            # Optional: Add heatmap layer for risk visualization
            if show_heatmap:
                heatmap_layer = pdk.Layer(
                    'HeatmapLayer',
                    data=citizens_gdf,
                    get_position='[lon, lat]',
                    get_weight='risk_score',
                    radiusPixels=60,
                    opacity=0.5
                )
                layers.append(heatmap_layer)
        
        # Determine view state
        if not citizens_gdf.empty:
            center_lat = citizens_gdf['lat'].mean()
            center_lon = citizens_gdf['lon'].mean()
        else:
            center_lat, center_lon = self.default_center[1], self.default_center[0]
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=self.default_zoom,
            pitch=45,
            bearing=0
        )
        
        # Create deck
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip={
                'html': '<b>Name:</b> {name}<br/>'
                       '<b>Risk Score:</b> {risk_score}<br/>'
                       '<b>Age:</b> {age}<br/>'
                       '<b>Occupation:</b> {occupation}',
                'style': {
                    'backgroundColor': 'steelblue',
                    'color': 'white'
                }
            }
        )
        
        return deck
    
    def create_hexagon_map(self, citizens_gdf: gpd.GeoDataFrame) -> pdk.Deck:
        """
        Create hexagon aggregation map for large-scale visualization
        
        Args:
            citizens_gdf: GeoDataFrame of citizens
            
        Returns:
            PyDeck Deck object with hexagon layer
        """
        if citizens_gdf.empty:
            return self.create_pydeck_map(citizens_gdf, gpd.GeoDataFrame())
        
        # Create hexagon layer
        hexagon_layer = pdk.Layer(
            'HexagonLayer',
            data=citizens_gdf,
            get_position='[lon, lat]',
            get_elevation_weight='risk_score',
            elevation_scale=50,
            elevation_range=[0, 1000],
            radius=200,
            coverage=0.9,
            extruded=True,
            pickable=True,
            auto_highlight=True
        )
        
        view_state = pdk.ViewState(
            latitude=citizens_gdf['lat'].mean(),
            longitude=citizens_gdf['lon'].mean(),
            zoom=11,
            pitch=45,
            bearing=0
        )
        
        deck = pdk.Deck(
            layers=[hexagon_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10'
        )
        
        return deck
    
    def create_arc_map(self, movements_data: List[Dict]) -> pdk.Deck:
        """
        Create arc map showing movement patterns
        
        Args:
            movements_data: List of movement dictionaries with source/target coords
            
        Returns:
            PyDeck Deck object with arc layer
        """
        if not movements_data:
            return self.create_pydeck_map(gpd.GeoDataFrame(), gpd.GeoDataFrame())
        
        movements_df = pd.DataFrame(movements_data)
        
        arc_layer = pdk.Layer(
            'ArcLayer',
            data=movements_df,
            get_source_position='[source_lon, source_lat]',
            get_target_position='[target_lon, target_lat]',
            get_source_color='[255, 100, 100, 160]',
            get_target_color='[100, 100, 255, 160]',
            get_width=3,
            pickable=True,
            auto_highlight=True
        )
        
        # Calculate center from all positions
        all_lats = movements_df[['source_lat', 'target_lat']].values.flatten()
        all_lons = movements_df[['source_lon', 'target_lon']].values.flatten()
        
        view_state = pdk.ViewState(
            latitude=np.mean(all_lats),
            longitude=np.mean(all_lons),
            zoom=11,
            pitch=30,
            bearing=0
        )
        
        deck = pdk.Deck(
            layers=[arc_layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/dark-v10'
        )
        
        return deck
    
    def generate_mock_coordinates(self, n_citizens: int, center: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        Generate mock geographic coordinates for testing
        
        Args:
            n_citizens: Number of citizens to generate coordinates for
            center: Optional (lat, lon) tuple for center point
            
        Returns:
            DataFrame with lat, lon coordinates
        """
        if center is None:
            center = (self.default_center[1], self.default_center[0])  # lat, lon
        
        # Generate coordinates in a radius around center
        radius = 0.1  # degrees (~11km)
        angles = np.random.uniform(0, 2*np.pi, n_citizens)
        distances = np.random.uniform(0, radius, n_citizens)
        
        lats = center[0] + distances * np.sin(angles)
        lons = center[1] + distances * np.cos(angles)
        
        return pd.DataFrame({'lat': lats, 'lon': lons})


def extract_geographic_data(connector) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract geographic data from Neo4j
    
    Args:
        connector: Neo4j connector instance
        
    Returns:
        Tuple of (citizens_data, locations_data)
    """
    with connector.driver.session() as session:
        # Get citizens with their current location
        citizens_query = """
        MATCH (c:Citizen)
        OPTIONAL MATCH (c)-[m:MOVES_TO]->(l:Location)
        WITH c, l, m
        ORDER BY m.timestamp DESC
        WITH c, COLLECT(l)[0] as current_location
        RETURN c.id as id, c.name as name, c.age as age, 
               c.risk_seed as risk_score, c.occupation as occupation,
               COALESCE(current_location.lat, 19.4326 + (rand() - 0.5) * 0.1) as lat,
               COALESCE(current_location.lon, -99.1332 + (rand() - 0.5) * 0.1) as lon
        """
        result = session.run(citizens_query)
        citizens_data = [dict(record) for record in result]
        
        # Get locations
        locations_query = """
        MATCH (l:Location)
        RETURN l.id as id, l.name as name, 
               l.lat as lat, l.lon as lon,
               l.crime_rate as crime_rate, l.area_type as area_type
        """
        result = session.run(locations_query)
        locations_data = [dict(record) for record in result]
    
    return citizens_data, locations_data


def main():
    """Test the mapbox visualizer"""
    print("=" * 70)
    print("Testing Mapbox Visualizer")
    print("=" * 70)
    
    # Create visualizer
    visualizer = MapboxVisualizer()
    
    # Generate mock data for testing
    n_citizens = 100
    coords = visualizer.generate_mock_coordinates(n_citizens)
    
    # Create mock citizen data
    citizens_data = []
    for i in range(n_citizens):
        citizens_data.append({
            'id': i,
            'name': f'Citizen_{i}',
            'age': np.random.randint(18, 80),
            'risk_score': np.random.beta(2, 5),
            'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist']),
            'lat': coords.iloc[i]['lat'],
            'lon': coords.iloc[i]['lon']
        })
    
    # Create mock location data
    locations_data = [
        {'id': 0, 'name': 'City Center', 'lat': 19.4326, 'lon': -99.1332, 'crime_rate': 0.3, 'area_type': 'commercial'},
        {'id': 1, 'name': 'North District', 'lat': 19.4826, 'lon': -99.1332, 'crime_rate': 0.5, 'area_type': 'residential'},
        {'id': 2, 'name': 'South District', 'lat': 19.3826, 'lon': -99.1332, 'crime_rate': 0.4, 'area_type': 'industrial'},
    ]
    
    # Create GeoDataFrames
    citizens_gdf, locations_gdf = visualizer.create_geodataframe(citizens_data, locations_data)
    
    print(f"\n✓ Created {len(citizens_gdf)} citizen points")
    print(f"✓ Created {len(locations_gdf)} location points")
    
    # Create PyDeck map
    deck = visualizer.create_pydeck_map(citizens_gdf, locations_gdf)
    print("\n✓ PyDeck map created successfully")
    
    # Save to HTML for testing
    output_path = Path(__file__).parent.parent / 'examples' / 'mapbox_test.html'
    output_path.parent.mkdir(exist_ok=True)
    deck.to_html(str(output_path))
    print(f"\n✓ Map saved to: {output_path}")
    print("\nOpen the HTML file in a browser to view the map!")
    print("=" * 70)


if __name__ == '__main__':
    main()
