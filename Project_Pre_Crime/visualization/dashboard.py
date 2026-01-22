"""
3D Graph Visualization Dashboard

Interactive 3D visualization of the pre-crime prediction graph using Plotly Dash.
Visualizes:
- Citizens as nodes in 3D space
- Risk levels (color-coded)
- Interactions between citizens
- Clusters and communities
- Red Balls (high-risk individuals)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

from connector import Neo4jConnector
from models import RedGAN


class GraphVisualizer:
    """3D Graph Visualizer for Pre-Crime Prediction System"""
    
    def __init__(self, connector: Neo4jConnector, model: RedGAN = None):
        """
        Initialize visualizer
        
        Args:
            connector: Neo4j connector
            model: Trained RedGAN model (optional)
        """
        self.connector = connector
        self.model = model
        
    def extract_graph_data(self):
        """Extract graph data from Neo4j"""
        data = self.connector.extract_subgraph()
        
        # Get node information from database
        with self.connector.driver.session() as session:
            query = """
            MATCH (c:Citizen)
            RETURN c.id as id, c.name as name, c.age as age, 
                   c.risk_seed as risk, c.occupation as occupation
            ORDER BY c.id
            """
            result = session.run(query)
            node_info = [record.data() for record in result]
        
        return data, node_info
    
    def compute_3d_layout(self, data, method='tsne'):
        """
        Compute 3D positions for nodes
        
        Args:
            data: PyTorch Geometric Data object
            method: 'tsne', 'pca', or 'spring'
            
        Returns:
            Array of 3D positions [num_nodes, 3]
        """
        if method == 'tsne':
            # Use t-SNE for dimensionality reduction
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, data.x.shape[0]-1))
            
            # Add edge information to features
            features = data.x.cpu().numpy()
            
            # If we have a model, use embeddings
            if self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(data.x, data.edge_index)
                    features = output['embeddings'].cpu().numpy()
            
            positions = tsne.fit_transform(features)
            
        elif method == 'pca':
            # Use PCA for dimensionality reduction
            pca = PCA(n_components=3, random_state=42)
            features = data.x.cpu().numpy()
            
            if self.model is not None:
                self.model.eval()
                with torch.no_grad():
                    output = self.model(data.x, data.edge_index)
                    features = output['embeddings'].cpu().numpy()
            
            positions = pca.fit_transform(features)
            
        else:  # spring layout (force-directed)
            # Simple spring-based layout
            num_nodes = data.x.shape[0]
            positions = np.random.randn(num_nodes, 3) * 10
            
            # Apply spring forces
            edge_index = data.edge_index.cpu().numpy()
            for _ in range(50):
                forces = np.zeros((num_nodes, 3))
                
                # Attraction along edges
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i], edge_index[1, i]
                    diff = positions[dst] - positions[src]
                    dist = np.linalg.norm(diff) + 1e-6
                    force = diff / dist * (dist - 5.0) * 0.1
                    forces[src] += force
                    forces[dst] -= force
                
                # Repulsion between all nodes
                for i in range(num_nodes):
                    for j in range(i+1, num_nodes):
                        diff = positions[j] - positions[i]
                        dist = np.linalg.norm(diff) + 1e-6
                        force = -diff / dist * 50.0 / (dist**2)
                        forces[i] += force
                        forces[j] -= force
                
                positions += forces
        
        return positions
    
    def create_3d_plot(self, data, node_info, positions, show_edges=True, risk_threshold=0.5):
        """
        Create 3D plotly figure
        
        Args:
            data: Graph data
            node_info: Node information from database
            positions: 3D positions
            show_edges: Whether to show edges
            risk_threshold: Threshold for red balls
            
        Returns:
            Plotly figure
        """
        # Get risk predictions if model available
        if self.model is not None:
            self.model.eval()
            with torch.no_grad():
                output = self.model(data.x, data.edge_index)
                risk_scores = output['risk_scores'].cpu().numpy().flatten()
                red_balls = output['red_balls'].cpu().numpy()
        else:
            risk_scores = data.x[:, 1].cpu().numpy()
            red_balls = risk_scores > risk_threshold
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        if show_edges:
            edge_index = data.edge_index.cpu().numpy()
            edge_x, edge_y, edge_z = [], [], []
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                edge_x.extend([positions[src, 0], positions[dst, 0], None])
                edge_y.extend([positions[src, 1], positions[dst, 1], None])
                edge_z.extend([positions[src, 2], positions[dst, 2], None])
            
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='rgba(125, 125, 125, 0.2)', width=1),
                hoverinfo='none',
                name='Connections'
            ))
        
        # Separate normal and red ball nodes
        normal_mask = ~red_balls
        
        # Add normal nodes
        if normal_mask.any():
            hover_text = [
                f"<b>{info['name']}</b><br>" +
                f"ID: {info['id']}<br>" +
                f"Age: {info['age']}<br>" +
                f"Risk: {risk_scores[i]:.3f}<br>" +
                f"Occupation: {info['occupation']}"
                for i, info in enumerate(node_info) if normal_mask[i]
            ]
            
            fig.add_trace(go.Scatter3d(
                x=positions[normal_mask, 0],
                y=positions[normal_mask, 1],
                z=positions[normal_mask, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=risk_scores[normal_mask],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Risk Score"),
                    line=dict(color='white', width=0.5)
                ),
                text=hover_text,
                hoverinfo='text',
                name='Citizens'
            ))
        
        # Add red balls (high-risk nodes)
        if red_balls.any():
            hover_text = [
                f"<b>⚠️ RED BALL ⚠️</b><br>" +
                f"<b>{info['name']}</b><br>" +
                f"ID: {info['id']}<br>" +
                f"Age: {info['age']}<br>" +
                f"Risk: {risk_scores[i]:.3f}<br>" +
                f"Occupation: {info['occupation']}"
                for i, info in enumerate(node_info) if red_balls[i]
            ]
            
            fig.add_trace(go.Scatter3d(
                x=positions[red_balls, 0],
                y=positions[red_balls, 1],
                z=positions[red_balls, 2],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(color='darkred', width=2)
                ),
                text=hover_text,
                hoverinfo='text',
                name='Red Balls'
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Pre-Crime Prediction System - 3D Graph Visualization',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, title=''),
                yaxis=dict(showticklabels=False, showgrid=False, title=''),
                zaxis=dict(showticklabels=False, showgrid=False, title=''),
                bgcolor='rgb(10, 10, 10)'
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor='rgb(20, 20, 20)',
            plot_bgcolor='rgb(10, 10, 10)',
            font=dict(color='white')
        )
        
        return fig


def create_dashboard():
    """Create Dash dashboard application"""
    
    # Initialize connector
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'precrime2024')
    
    connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    
    # Try to load model if available
    model = None
    checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Loading model from {latest_checkpoint}")
            try:
                # Extract graph to get dimensions
                data = connector.extract_subgraph()
                model = RedGAN(in_channels=data.x.shape[1])
                checkpoint = torch.load(latest_checkpoint, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Could not load model: {e}")
    
    visualizer = GraphVisualizer(connector, model)
    
    # Create Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Pre-Crime Prediction System", 
                style={'textAlign': 'center', 'color': 'white', 'marginBottom': 30}),
        
        html.Div([
            html.Div([
                html.Label("Layout Method:", style={'color': 'white'}),
                dcc.Dropdown(
                    id='layout-method',
                    options=[
                        {'label': 't-SNE (Recommended)', 'value': 'tsne'},
                        {'label': 'PCA', 'value': 'pca'},
                        {'label': 'Spring Layout', 'value': 'spring'}
                    ],
                    value='tsne',
                    style={'width': '200px'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Div([
                html.Label("Risk Threshold:", style={'color': 'white'}),
                dcc.Slider(
                    id='risk-threshold',
                    min=0,
                    max=1,
                    step=0.05,
                    value=0.5,
                    marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'display': 'inline-block', 'width': '300px', 'marginRight': 20}),
            
            html.Div([
                html.Label("Show Edges:", style={'color': 'white'}),
                dcc.Checklist(
                    id='show-edges',
                    options=[{'label': ' Yes', 'value': 'show'}],
                    value=['show'],
                    style={'color': 'white'}
                )
            ], style={'display': 'inline-block', 'marginRight': 20}),
            
            html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                       style={'marginLeft': 20})
        ], style={'textAlign': 'center', 'marginBottom': 20, 'padding': 20}),
        
        dcc.Loading(
            id="loading-graph",
            type="default",
            children=dcc.Graph(
                id='3d-graph',
                style={'height': '80vh'}
            )
        ),
        
        html.Div(id='stats-display', style={
            'textAlign': 'center',
            'color': 'white',
            'marginTop': 20,
            'padding': 20,
            'backgroundColor': 'rgb(30, 30, 30)',
            'borderRadius': 5
        })
    ], style={'backgroundColor': 'rgb(20, 20, 20)', 'minHeight': '100vh', 'padding': 20})
    
    @app.callback(
        [Output('3d-graph', 'figure'),
         Output('stats-display', 'children')],
        [Input('layout-method', 'value'),
         Input('risk-threshold', 'value'),
         Input('show-edges', 'value'),
         Input('refresh-button', 'n_clicks')]
    )
    def update_graph(layout_method, risk_threshold, show_edges, n_clicks):
        # Extract data
        data, node_info = visualizer.extract_graph_data()
        
        # Compute 3D layout
        positions = visualizer.compute_3d_layout(data, method=layout_method)
        
        # Create plot
        show_edges_bool = 'show' in (show_edges or [])
        fig = visualizer.create_3d_plot(data, node_info, positions, 
                                       show_edges=show_edges_bool,
                                       risk_threshold=risk_threshold)
        
        # Compute stats
        if model is not None:
            model.eval()
            with torch.no_grad():
                output = model(data.x, data.edge_index)
                risk_scores = output['risk_scores'].cpu().numpy().flatten()
                red_balls = output['red_balls'].cpu().numpy()
        else:
            risk_scores = data.x[:, 1].cpu().numpy()
            red_balls = risk_scores > risk_threshold
        
        stats = html.Div([
            html.H3("Graph Statistics"),
            html.Div([
                html.Div([
                    html.H4(f"{data.num_nodes}"),
                    html.P("Total Citizens")
                ], style={'display': 'inline-block', 'margin': '0 30px'}),
                
                html.Div([
                    html.H4(f"{data.num_edges}"),
                    html.P("Connections")
                ], style={'display': 'inline-block', 'margin': '0 30px'}),
                
                html.Div([
                    html.H4(f"{red_balls.sum()}", style={'color': 'red'}),
                    html.P("Red Balls Detected")
                ], style={'display': 'inline-block', 'margin': '0 30px'}),
                
                html.Div([
                    html.H4(f"{risk_scores.mean():.3f}"),
                    html.P("Average Risk")
                ], style={'display': 'inline-block', 'margin': '0 30px'}),
                
                html.Div([
                    html.H4(f"{risk_scores.max():.3f}"),
                    html.P("Max Risk")
                ], style={'display': 'inline-block', 'margin': '0 30px'})
            ])
        ])
        
        return fig, stats
    
    return app, connector


def main():
    """Run the dashboard"""
    print("=" * 70)
    print("Starting Pre-Crime Prediction 3D Visualization Dashboard")
    print("=" * 70)
    
    try:
        app, connector = create_dashboard()
        
        print("\n✓ Dashboard initialized")
        print("\nAccess the visualization at:")
        print("  → http://localhost:8050")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70 + "\n")
        
        app.run_server(debug=False, host='0.0.0.0', port=8050)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'connector' in locals():
            connector.close()


if __name__ == '__main__':
    main()
