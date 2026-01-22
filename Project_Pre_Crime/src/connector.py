"""
Neo4j Connector and Subgraph Extraction Module

This module handles:
- Connection to Neo4j database
- Data generation using Faker with Beta distribution for risk_seed
- Subgraph extraction for GNN training
- Graph data conversion to PyTorch Geometric format
"""

from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data


class Neo4jConnector:
    """Handles Neo4j database connections and operations"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password"):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j database URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.fake = Faker()
        Faker.seed(42)  # For reproducibility
        np.random.seed(42)
        
    def close(self):
        """Close database connection"""
        self.driver.close()
        
    def setup_schema(self, cypher_file: str):
        """
        Execute Cypher schema file to setup database
        
        Args:
            cypher_file: Path to .cypher schema file
        """
        with open(cypher_file, 'r') as f:
            queries = f.read().split(';')
            
        with self.driver.session() as session:
            for query in queries:
                query = query.strip()
                if query and not query.startswith('//'):
                    try:
                        session.run(query)
                    except Exception as e:
                        if 'CREATE CONSTRAINT' not in query and 'CREATE INDEX' not in query:
                            print(f"Warning executing query: {e}")
                            
    def generate_risk_seed(self, alpha: float = 2.0, beta: float = 5.0) -> float:
        """
        Generate risk seed using Beta distribution
        
        Beta distribution parameters:
        - alpha, beta: shape parameters controlling the distribution
        - For crime risk: alpha=2, beta=5 creates a distribution skewed towards lower risk
        
        Args:
            alpha: Alpha parameter for Beta distribution
            beta: Beta parameter for Beta distribution
            
        Returns:
            Risk seed value between 0 and 1
        """
        return np.random.beta(alpha, beta)
    
    def evolve_risk_seed(self, current_risk: float, 
                        interactions: int, 
                        location_crime_rate: float,
                        time_delta: float = 1.0) -> float:
        """
        Evolve risk seed based on interactions and environmental factors
        
        The risk evolves logically based on:
        - Current risk level
        - Number of high-risk interactions
        - Crime rate of visited locations
        - Time elapsed
        
        Args:
            current_risk: Current risk seed value
            interactions: Number of interactions with other citizens
            location_crime_rate: Average crime rate of visited locations
            time_delta: Time elapsed (in days)
            
        Returns:
            Evolved risk seed value
        """
        # Base evolution using Beta distribution with shifted parameters
        alpha_new = 2.0 + current_risk * 3.0 + interactions * 0.1
        beta_new = 5.0 - location_crime_rate * 2.0
        beta_new = max(beta_new, 1.0)  # Ensure beta stays positive
        
        # Generate new risk with some momentum from current risk
        new_risk_sample = np.random.beta(alpha_new, beta_new)
        evolved_risk = 0.7 * current_risk + 0.3 * new_risk_sample
        
        # Add environmental influence
        env_factor = location_crime_rate * 0.1 * time_delta
        evolved_risk = min(1.0, evolved_risk + env_factor)
        
        return float(evolved_risk)
    
    def generate_synthetic_data(self, 
                               num_citizens: int = 100,
                               num_locations: int = 20,
                               num_interactions: int = 200,
                               num_movements: int = 300):
        """
        Generate synthetic data using Faker and insert into Neo4j
        
        Args:
            num_citizens: Number of citizens to generate
            num_locations: Number of locations to generate
            num_interactions: Number of citizen interactions
            num_movements: Number of location movements
        """
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Generate citizens
            citizen_ids = []
            print(f"Generating {num_citizens} citizens...")
            for i in range(num_citizens):
                citizen_id = f"C{i:04d}"
                citizen_ids.append(citizen_id)
                risk_seed = self.generate_risk_seed()
                
                query = """
                CREATE (c:Citizen {
                    id: $id,
                    name: $name,
                    age: $age,
                    risk_seed: $risk_seed,
                    occupation: $occupation,
                    created_at: datetime()
                })
                """
                session.run(query, 
                          id=citizen_id,
                          name=self.fake.name(),
                          age=self.fake.random_int(min=18, max=80),
                          risk_seed=risk_seed,
                          occupation=self.fake.job())
            
            # Generate locations
            location_ids = []
            area_types = ['residential', 'commercial', 'industrial', 'recreational']
            print(f"Generating {num_locations} locations...")
            for i in range(num_locations):
                location_id = f"L{i:03d}"
                location_ids.append(location_id)
                
                query = """
                CREATE (l:Location {
                    id: $id,
                    name: $name,
                    latitude: $latitude,
                    longitude: $longitude,
                    crime_rate: $crime_rate,
                    area_type: $area_type
                })
                """
                session.run(query,
                          id=location_id,
                          name=self.fake.street_address(),
                          latitude=float(self.fake.latitude()),
                          longitude=float(self.fake.longitude()),
                          crime_rate=float(np.random.beta(2, 8)),  # Skewed towards lower crime
                          area_type=np.random.choice(area_types))
            
            # Generate interactions between citizens
            interaction_types = ['friend', 'family', 'colleague', 'conflict', 'acquaintance']
            print(f"Generating {num_interactions} interactions...")
            for _ in range(num_interactions):
                c1, c2 = np.random.choice(citizen_ids, size=2, replace=False)
                
                query = """
                MATCH (c1:Citizen {id: $id1})
                MATCH (c2:Citizen {id: $id2})
                CREATE (c1)-[:INTERACTS_WITH {
                    timestamp: datetime(),
                    interaction_type: $interaction_type,
                    frequency: $frequency,
                    strength: $strength
                }]->(c2)
                """
                session.run(query,
                          id1=c1,
                          id2=c2,
                          interaction_type=np.random.choice(interaction_types),
                          frequency=int(np.random.randint(1, 50)),
                          strength=float(np.random.beta(5, 2)))  # Skewed towards stronger relationships
            
            # Generate movements to locations
            purposes = ['work', 'shopping', 'leisure', 'visiting', 'transit']
            print(f"Generating {num_movements} movements...")
            for _ in range(num_movements):
                citizen = np.random.choice(citizen_ids)
                location = np.random.choice(location_ids)
                
                query = """
                MATCH (c:Citizen {id: $citizen_id})
                MATCH (l:Location {id: $location_id})
                CREATE (c)-[:MOVES_TO {
                    timestamp: datetime(),
                    duration: $duration,
                    purpose: $purpose
                }]->(l)
                """
                session.run(query,
                          citizen_id=citizen,
                          location_id=location,
                          duration=int(np.random.randint(5, 480)),  # 5 mins to 8 hours
                          purpose=np.random.choice(purposes))
            
            print("Synthetic data generation complete!")
    
    def extract_subgraph(self, 
                        center_node: Optional[str] = None,
                        hop_limit: int = 2) -> Data:
        """
        Extract subgraph from Neo4j and convert to PyTorch Geometric format
        
        Args:
            center_node: Central node ID (if None, extracts full graph)
            hop_limit: Number of hops for neighborhood extraction
            
        Returns:
            PyTorch Geometric Data object
        """
        with self.driver.session() as session:
            if center_node:
                # Extract k-hop neighborhood
                query = f"""
                MATCH path = (center:Citizen {{id: $center_id}})-[*1..{hop_limit}]-(neighbor)
                WITH center, neighbor, relationships(path) as rels
                RETURN center, neighbor, rels
                """
                result = session.run(query, center_id=center_node)
            else:
                # Extract full graph
                query = """
                MATCH (c:Citizen)
                OPTIONAL MATCH (c)-[r:INTERACTS_WITH]->(other:Citizen)
                OPTIONAL MATCH (c)-[m:MOVES_TO]->(l:Location)
                RETURN c, r, other, m, l
                """
                result = session.run(query)
            
            # Parse results and build graph structure
            nodes = {}
            edges = []
            node_features = []
            edge_features = []
            
            for record in result:
                # Process citizen nodes
                citizen = record.get('c') or record.get('center')
                if citizen and citizen['id'] not in nodes:
                    node_idx = len(nodes)
                    nodes[citizen['id']] = node_idx
                    # Node features: [age, risk_seed]
                    node_features.append([
                        citizen.get('age', 0) / 100.0,  # Normalize
                        citizen.get('risk_seed', 0.0)
                    ])
                
                # Process interactions
                if record.get('r') and record.get('other'):
                    other = record['other']
                    if other['id'] not in nodes:
                        node_idx = len(nodes)
                        nodes[other['id']] = node_idx
                        node_features.append([
                            other.get('age', 0) / 100.0,
                            other.get('risk_seed', 0.0)
                        ])
                    
                    edges.append([nodes[citizen['id']], nodes[other['id']]])
                    rel = record['r']
                    # Edge features: [frequency, strength]
                    edge_features.append([
                        rel.get('frequency', 0) / 50.0,  # Normalize
                        rel.get('strength', 0.0)
                    ])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
            
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_high_risk_citizens(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Query high-risk citizens based on risk_seed threshold
        
        Args:
            threshold: Risk threshold (0-1)
            
        Returns:
            DataFrame with high-risk citizen information
        """
        with self.driver.session() as session:
            query = """
            MATCH (c:Citizen)
            WHERE c.risk_seed >= $threshold
            RETURN c.id as id, c.name as name, c.age as age, 
                   c.risk_seed as risk_seed, c.occupation as occupation
            ORDER BY c.risk_seed DESC
            """
            result = session.run(query, threshold=threshold)
            records = [record.data() for record in result]
            return pd.DataFrame(records)


if __name__ == "__main__":
    # Example usage
    connector = Neo4jConnector()
    
    try:
        # Setup schema
        print("Setting up database schema...")
        connector.setup_schema("../scripts/setup_db.cypher")
        
        # Generate synthetic data
        print("\nGenerating synthetic data...")
        connector.generate_synthetic_data(
            num_citizens=100,
            num_locations=20,
            num_interactions=200,
            num_movements=300
        )
        
        # Extract subgraph
        print("\nExtracting subgraph...")
        graph_data = connector.extract_subgraph()
        print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
        
        # Query high-risk citizens
        print("\nHigh-risk citizens:")
        high_risk = connector.get_high_risk_citizens(threshold=0.3)
        print(high_risk.head())
        
    finally:
        connector.close()
