#!/usr/bin/env python3
"""
Dashboard Launcher for Pre-Crime Prediction System

This script provides an easy way to launch different dashboard variants.
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'visualization'))

def launch_plotly_dashboard():
    """Launch original Plotly Dash 3D visualization"""
    print("🚀 Launching Plotly Dash 3D Visualization...")
    print("   URL: http://localhost:8050")
    from visualization.dashboard import main
    main()

def launch_panel_dashboard():
    """Launch Panel HoloViz dashboard"""
    print("🚀 Launching Panel HoloViz Dashboard...")
    print("   URL: http://localhost:5006")
    from visualization.panel_dashboard import main
    main()

def launch_integrated_dashboard():
    """Launch integrated dashboard with all features"""
    print("🚀 Launching Integrated Dashboard...")
    print("   URL: http://localhost:5007")
    from visualization.integrated_dashboard import main
    main()

def launch_mapbox_test():
    """Test Mapbox visualization"""
    print("🚀 Testing Mapbox Visualization...")
    from visualization.mapbox import main
    main()

def launch_force_graph():
    """Launch 3D Force Graph with FastAPI backend"""
    print("🚀 Launching 3D Force Graph with FastAPI...")
    print("   API: http://localhost:8001")
    print("   Visualization: http://localhost:8001/visualization")
    from visualization.api_server import main
    main()

def show_menu():
    """Show interactive menu"""
    print("=" * 80)
    print("🎯 Pre-Crime Prediction System - Dashboard Launcher")
    print("=" * 80)
    print("\nAvailable dashboards:\n")
    print("  1. Plotly Dash 3D Visualization (Original)")
    print("     • Interactive 3D graph visualization")
    print("     • t-SNE/PCA/Spring layouts")
    print("     • Red Ball detection")
    print("     • Port: 8050\n")
    
    print("  2. Panel HoloViz Dashboard")
    print("     • Modern Panel-based interface")
    print("     • HoloViews visualizations")
    print("     • Interactive controls")
    print("     • Port: 5006\n")
    
    print("  3. Integrated Dashboard (Recommended)")
    print("     • Combines all features")
    print("     • PyDeck 3D maps")
    print("     • DuckDB fast queries")
    print("     • GeoViews geographic analysis")
    print("     • Port: 5007\n")
    
    print("  4. 3D Force Graph (NEW - GNN Visualization)")
    print("     • Three.js-based 3D force-directed graph")
    print("     • Real-time physics simulation with d3-forces")
    print("     • FastAPI backend serving Neo4j data")
    print("     • Interactive node exploration")
    print("     • Port: 8001\n")
    
    print("  5. Mapbox Test (Generate HTML)")
    print("     • Test Mapbox integration")
    print("     • Exports to HTML file\n")
    
    print("  6. Exit\n")
    print("=" * 80)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        launch_plotly_dashboard()
    elif choice == '2':
        launch_panel_dashboard()
    elif choice == '3':
        launch_integrated_dashboard()
    elif choice == '4':
        launch_force_graph()
    elif choice == '5':
        launch_mapbox_test()
    elif choice == '6':
        print("\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please try again.\n")
        show_menu()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Launch Pre-Crime Prediction System Dashboards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_dashboard.py                    # Interactive menu
  python launch_dashboard.py --plotly           # Launch Plotly dashboard
  python launch_dashboard.py --panel            # Launch Panel dashboard
  python launch_dashboard.py --integrated       # Launch integrated dashboard
  python launch_dashboard.py --force-graph      # Launch 3D Force Graph
  python launch_dashboard.py --mapbox           # Test Mapbox
  python launch_dashboard.py --all              # Show all options
        """
    )
    
    parser.add_argument('--plotly', action='store_true',
                      help='Launch Plotly Dash 3D visualization (port 8050)')
    parser.add_argument('--panel', action='store_true',
                      help='Launch Panel HoloViz dashboard (port 5006)')
    parser.add_argument('--integrated', action='store_true',
                      help='Launch integrated dashboard (port 5007)')
    parser.add_argument('--force-graph', action='store_true',
                      help='Launch 3D Force Graph with FastAPI (port 8001)')
    parser.add_argument('--mapbox', action='store_true',
                      help='Test Mapbox visualization')
    parser.add_argument('--all', action='store_true',
                      help='Show all available dashboards')
    
    args = parser.parse_args()
    
    # If no arguments provided, show interactive menu
    if not any([args.plotly, args.panel, args.integrated, args.mapbox, args.all]):
        show_menu()
        return
    
    # Show all options
    if args.all:
        print("\n" + "=" * 80)
        print("📊 Available Dashboards:")
        print("=" * 80)
        print("\n1. Plotly Dash 3D Visualization")
        print("   Command: python launch_dashboard.py --plotly")
        print("   URL: http://localhost:8050")
        print("\n2. Panel HoloViz Dashboard")
        print("   Command: python launch_dashboard.py --panel")
        print("   URL: http://localhost:5006")
        print("\n3. Integrated Dashboard")
        print("   Command: python launch_dashboard.py --integrated")
        print("   URL: http://localhost:5007")
        print("\n4. 3D Force Graph (GNN Visualization)")
        print("   Command: python launch_dashboard.py --force-graph")
        print("   URL: http://localhost:8001/visualization")
        print("\n5. Mapbox Test")
        print("   Command: python launch_dashboard.py --mapbox")
        print("=" * 80 + "\n")
        return
    
    # Launch requested dashboard
    if args.plotly:
        launch_plotly_dashboard()
    elif args.panel:
        launch_panel_dashboard()
    elif args.integrated:
        launch_integrated_dashboard()
    elif args.force_graph:
        launch_force_graph()
    elif args.mapbox:
        launch_mapbox_test()

if __name__ == '__main__':
    main()
