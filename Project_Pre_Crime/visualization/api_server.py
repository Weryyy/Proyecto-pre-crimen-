"""
FastAPI Backend for 3D Force Graph Visualization

This module provides a REST API to serve Neo4j graph data as JSON
for consumption by the 3D Force Graph visualization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json

try:
    from connector import Neo4jConnector
    from models import RedGAN
    import torch
except ImportError:
    print("Warning: Could not import project modules. API will use mock data.")
    Neo4jConnector = None
    RedGAN = None
    torch = None

# Initialize FastAPI app
app = FastAPI(
    title="Pre-Crime 3D Force Graph API",
    description="REST API for serving Neo4j graph data to 3D Force Graph visualization",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connector instance
connector = None
model = None


# Pydantic models for API responses
class Node(BaseModel):
    id: str
    name: str
    risk_score: float
    age: int
    occupation: str
    type: str = "citizen"
    color: Optional[str] = None
    size: Optional[float] = None


class Link(BaseModel):
    source: str
    target: str
    type: str
    strength: Optional[float] = 1.0


class GraphData(BaseModel):
    nodes: List[Node]
    links: List[Link]


class Stats(BaseModel):
    total_nodes: int
    total_links: int
    high_risk_count: int
    avg_risk: float
    max_risk: float


@app.on_event("startup")
async def startup_event():
    """Initialize Neo4j connection on startup"""
    global connector, model
    
    if Neo4jConnector is not None:
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'precrime2024')
            
            connector = Neo4jConnector(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
            print("✓ Connected to Neo4j")
            
            # Try to load model
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
            print("Using mock data mode")


@app.on_event("shutdown")
async def shutdown_event():
    """Close Neo4j connection on shutdown"""
    global connector
    if connector:
        connector.close()


def get_risk_color(risk_score: float) -> str:
    """Get color hex code based on risk score"""
    if risk_score < 0.3:
        return "#00ff00"  # Green - low risk
    elif risk_score < 0.5:
        return "#ffff00"  # Yellow - medium risk
    elif risk_score < 0.7:
        return "#ffa500"  # Orange - high risk
    else:
        return "#ff0000"  # Red - very high risk


def get_node_size(risk_score: float) -> float:
    """Get node size based on risk score"""
    return 5 + (risk_score * 15)  # Size between 5 and 20


def generate_mock_graph_data(num_nodes: int = 100) -> GraphData:
    """Generate mock graph data for testing"""
    import random
    import numpy as np
    
    nodes = []
    links = []
    
    # Generate nodes
    for i in range(num_nodes):
        risk_score = np.random.beta(2, 5)
        node = Node(
            id=f"citizen_{i}",
            name=f"Citizen {i:03d}",
            risk_score=round(risk_score, 3),
            age=random.randint(18, 80),
            occupation=random.choice(['Engineer', 'Teacher', 'Doctor', 'Artist', 'Driver']),
            type="citizen",
            color=get_risk_color(risk_score),
            size=get_node_size(risk_score)
        )
        nodes.append(node)
    
    # Generate links (interactions)
    num_links = num_nodes * 2
    for _ in range(num_links):
        source_idx = random.randint(0, num_nodes - 1)
        target_idx = random.randint(0, num_nodes - 1)
        if source_idx != target_idx:
            link = Link(
                source=f"citizen_{source_idx}",
                target=f"citizen_{target_idx}",
                type="INTERACTS_WITH",
                strength=round(random.uniform(0.3, 1.0), 2)
            )
            links.append(link)
    
    return GraphData(nodes=nodes, links=links)


def extract_graph_from_neo4j(limit: int = 200) -> GraphData:
    """Extract graph data from Neo4j"""
    if connector is None:
        return generate_mock_graph_data(limit)
    
    try:
        with connector.driver.session() as session:
            # Query for citizens (nodes)
            nodes_query = f"""
            MATCH (c:Citizen)
            RETURN c.id as id, c.name as name, c.age as age,
                   c.risk_seed as risk_score, c.occupation as occupation
            LIMIT {limit}
            """
            result = session.run(nodes_query)
            citizens_data = [dict(record) for record in result]
            
            # Create nodes
            nodes = []
            citizen_ids = set()
            for citizen in citizens_data:
                citizen_id = str(citizen['id'])
                citizen_ids.add(citizen_id)
                risk_score = float(citizen.get('risk_score', 0.5))
                
                node = Node(
                    id=citizen_id,
                    name=citizen['name'],
                    risk_score=round(risk_score, 3),
                    age=citizen['age'],
                    occupation=citizen.get('occupation', 'Unknown'),
                    type="citizen",
                    color=get_risk_color(risk_score),
                    size=get_node_size(risk_score)
                )
                nodes.append(node)
            
            # Query for interactions (links)
            links_query = f"""
            MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
            WHERE c1.id IN {list(citizen_ids)} AND c2.id IN {list(citizen_ids)}
            RETURN c1.id as source, c2.id as target, 
                   r.type as type, r.strength as strength
            LIMIT {limit * 3}
            """
            result = session.run(links_query)
            links_data = [dict(record) for record in result]
            
            # Create links
            links = []
            for link_data in links_data:
                link = Link(
                    source=str(link_data['source']),
                    target=str(link_data['target']),
                    type=link_data.get('type', 'INTERACTS_WITH'),
                    strength=float(link_data.get('strength', 1.0))
                )
                links.append(link)
            
            return GraphData(nodes=nodes, links=links)
            
    except Exception as e:
        print(f"Error extracting from Neo4j: {e}")
        return generate_mock_graph_data(limit)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <html>
        <head>
            <title>Pre-Crime 3D Force Graph API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #fff; }
                h1 { color: #ff4444; }
                a { color: #4444ff; }
                .endpoint { background: #2a2a2a; padding: 15px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>🎯 Pre-Crime 3D Force Graph API</h1>
            <p>REST API for serving Neo4j graph data to 3D Force Graph visualization</p>
            
            <h2>📡 Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>GET /api/graph</h3>
                <p>Get complete graph data (nodes and links) in 3D Force Graph format</p>
                <p>Parameters: <code>limit</code> (default: 200)</p>
                <p><a href="/api/graph?limit=50">Try it: /api/graph?limit=50</a></p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/nodes</h3>
                <p>Get only nodes data</p>
                <p>Parameters: <code>limit</code>, <code>min_risk</code></p>
                <p><a href="/api/nodes?limit=20">Try it: /api/nodes?limit=20</a></p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/stats</h3>
                <p>Get graph statistics</p>
                <p><a href="/api/stats">Try it: /api/stats</a></p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/high-risk</h3>
                <p>Get high-risk citizens only</p>
                <p>Parameters: <code>threshold</code> (default: 0.7)</p>
                <p><a href="/api/high-risk?threshold=0.6">Try it: /api/high-risk?threshold=0.6</a></p>
            </div>
            
            <div class="endpoint">
                <h3>GET /visualization</h3>
                <p>Interactive 3D Force Graph visualization</p>
                <p><a href="/visualization">Open Visualization</a></p>
            </div>
            
            <h2>📖 Documentation:</h2>
            <p><a href="/docs">Interactive API Docs (Swagger)</a></p>
            <p><a href="/redoc">Alternative API Docs (ReDoc)</a></p>
        </body>
    </html>
    """


@app.get("/api/graph", response_model=GraphData)
async def get_graph(limit: int = Query(200, ge=10, le=1000)):
    """
    Get complete graph data with nodes and links
    
    Parameters:
    - limit: Maximum number of nodes to return (10-1000)
    
    Returns:
    - GraphData with nodes and links in 3D Force Graph format
    """
    return extract_graph_from_neo4j(limit)


@app.get("/api/nodes", response_model=List[Node])
async def get_nodes(
    limit: int = Query(200, ge=1, le=1000),
    min_risk: Optional[float] = Query(None, ge=0.0, le=1.0)
):
    """
    Get nodes only, optionally filtered by minimum risk score
    
    Parameters:
    - limit: Maximum number of nodes
    - min_risk: Minimum risk score filter (optional)
    """
    graph_data = extract_graph_from_neo4j(limit)
    nodes = graph_data.nodes
    
    if min_risk is not None:
        nodes = [node for node in nodes if node.risk_score >= min_risk]
    
    return nodes


@app.get("/api/stats", response_model=Stats)
async def get_stats():
    """Get graph statistics"""
    graph_data = extract_graph_from_neo4j(500)
    
    risk_scores = [node.risk_score for node in graph_data.nodes]
    high_risk_count = sum(1 for score in risk_scores if score > 0.7)
    
    return Stats(
        total_nodes=len(graph_data.nodes),
        total_links=len(graph_data.links),
        high_risk_count=high_risk_count,
        avg_risk=round(sum(risk_scores) / len(risk_scores), 3) if risk_scores else 0.0,
        max_risk=round(max(risk_scores), 3) if risk_scores else 0.0
    )


@app.get("/api/high-risk", response_model=GraphData)
async def get_high_risk(threshold: float = Query(0.7, ge=0.0, le=1.0)):
    """
    Get subgraph containing only high-risk citizens and their connections
    
    Parameters:
    - threshold: Risk score threshold (default: 0.7)
    """
    graph_data = extract_graph_from_neo4j(500)
    
    # Filter high-risk nodes
    high_risk_nodes = [node for node in graph_data.nodes if node.risk_score >= threshold]
    high_risk_ids = {node.id for node in high_risk_nodes}
    
    # Filter links to only include connections between high-risk nodes
    filtered_links = [
        link for link in graph_data.links
        if link.source in high_risk_ids and link.target in high_risk_ids
    ]
    
    return GraphData(nodes=high_risk_nodes, links=filtered_links)


@app.get("/visualization", response_class=HTMLResponse)
async def visualization():
    """Serve 3D Force Graph visualization page"""
    html_file = Path(__file__).parent / "force_graph_3d.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>3D Force Graph</title></head>
            <body>
                <h1>3D Force Graph visualization file not found</h1>
                <p>Please ensure force_graph_3d.html is in the visualization directory</p>
            </body>
        </html>
        """, status_code=404)


def main():
    """Run the FastAPI server"""
    import uvicorn
    
    print("=" * 70)
    print("🚀 Starting Pre-Crime 3D Force Graph API")
    print("=" * 70)
    print("\n✓ FastAPI server starting...")
    print("\nAccess points:")
    print("  • API Root:          http://localhost:8001")
    print("  • 3D Visualization:  http://localhost:8001/visualization")
    print("  • API Docs:          http://localhost:8001/docs")
    print("  • Graph Data:        http://localhost:8001/api/graph")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
