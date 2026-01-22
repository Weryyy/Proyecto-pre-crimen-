#!/usr/bin/env python3
"""
Installation Verification Script for HoloViz Integration

This script checks if all required dependencies are installed correctly.
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        print(f"✅ {package_name:<20} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name:<20} - MISSING")
        return False

def main():
    """Run verification checks"""
    print("=" * 70)
    print("HoloViz Integration - Installation Verification")
    print("=" * 70)
    print("\nChecking dependencies...\n")
    
    required = []
    
    # Core dependencies
    print("Core Dependencies:")
    required.append(check_import('torch', 'PyTorch'))
    required.append(check_import('torch_geometric', 'PyTorch Geometric'))
    required.append(check_import('neo4j', 'Neo4j Driver'))
    required.append(check_import('pandas', 'Pandas'))
    required.append(check_import('numpy', 'NumPy'))
    
    # Visualization basics
    print("\nVisualization Libraries:")
    required.append(check_import('plotly', 'Plotly'))
    required.append(check_import('dash', 'Dash'))
    
    # HoloViz ecosystem
    print("\nHoloViz Ecosystem:")
    required.append(check_import('panel', 'Panel'))
    required.append(check_import('holoviews', 'HoloViews'))
    required.append(check_import('hvplot', 'hvPlot'))
    required.append(check_import('geoviews', 'GeoViews'))
    required.append(check_import('datashader', 'Datashader'))
    required.append(check_import('colorcet', 'Colorcet'))
    required.append(check_import('bokeh', 'Bokeh'))
    
    # Geospatial
    print("\nGeospatial Libraries:")
    required.append(check_import('geopandas', 'Geopandas'))
    required.append(check_import('pydeck', 'PyDeck'))
    required.append(check_import('shapely', 'Shapely'))
    required.append(check_import('folium', 'Folium'))
    
    # Data processing
    print("\nData Processing:")
    required.append(check_import('duckdb', 'DuckDB'))
    required.append(check_import('ibis', 'Ibis'))
    required.append(check_import('polars', 'Polars'))
    required.append(check_import('pyarrow', 'PyArrow'))
    
    # ML optimization (optional)
    print("\nML Optimization (Optional):")
    check_import('optuna', 'Optuna')
    check_import('xgboost', 'XGBoost')
    check_import('lightgbm', 'LightGBM')
    check_import('catboost', 'CatBoost')
    
    # Network
    print("\nNetwork Analysis:")
    required.append(check_import('networkx', 'NetworkX'))
    
    # Check project files
    print("\nProject Files:")
    project_files = [
        'visualization/mapbox.py',
        'visualization/panel_dashboard.py',
        'visualization/integrated_dashboard.py',
        'examples/duckdb_analysis.py',
        'launch_dashboard.py',
    ]
    
    for file_path in project_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path:<40} - OK")
        else:
            print(f"❌ {file_path:<40} - MISSING")
            required.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    if all(required):
        print("✅ All required dependencies are installed!")
        print("\nYou can now run the dashboards:")
        print("  python launch_dashboard.py")
        print("  make dashboard-menu")
        print("=" * 70)
        return 0
    else:
        print("❌ Some required dependencies are missing!")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nOr install in groups:")
        print("  pip install panel holoviews hvplot geoviews datashader bokeh")
        print("  pip install geopandas pydeck shapely folium")
        print("  pip install duckdb ibis-framework polars pyarrow")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
