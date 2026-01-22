"""
Data Processing with DuckDB and Ibis

This module demonstrates advanced data processing capabilities using DuckDB and Ibis
for efficient querying and analytics on the pre-crime prediction data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import duckdb
import ibis
import pandas as pd
import numpy as np
from typing import Optional, Dict, List


class DataProcessor:
    """Advanced data processing with DuckDB and Ibis"""
    
    def __init__(self, connector: Optional[object] = None):
        """
        Initialize data processor
        
        Args:
            connector: Neo4j connector instance
        """
        self.connector = connector
        self.duckdb_conn = duckdb.connect(':memory:')
        self.ibis_conn = ibis.duckdb.connect(':memory:')
        
    def load_data_from_neo4j(self) -> Dict[str, pd.DataFrame]:
        """Load data from Neo4j into DataFrames"""
        if self.connector is None:
            return self._generate_sample_data()
        
        with self.connector.driver.session() as session:
            # Citizens query
            citizens_query = """
            MATCH (c:Citizen)
            OPTIONAL MATCH (c)-[m:MOVES_TO]->(l:Location)
            WITH c, l, m
            ORDER BY m.timestamp DESC
            WITH c, COLLECT(l)[0] as current_location
            RETURN c.id as citizen_id, c.name as name, c.age as age,
                   c.risk_seed as risk_score, c.occupation as occupation,
                   current_location.name as location_name,
                   current_location.crime_rate as location_crime_rate
            """
            result = session.run(citizens_query)
            citizens_df = pd.DataFrame([dict(record) for record in result])
            
            # Interactions query
            interactions_query = """
            MATCH (c1:Citizen)-[i:INTERACTS_WITH]->(c2:Citizen)
            RETURN c1.id as citizen1_id, c2.id as citizen2_id,
                   i.timestamp as timestamp, i.type as interaction_type,
                   i.frequency as frequency, i.strength as strength
            """
            result = session.run(interactions_query)
            interactions_df = pd.DataFrame([dict(record) for record in result])
            
            # Movements query
            movements_query = """
            MATCH (c:Citizen)-[m:MOVES_TO]->(l:Location)
            RETURN c.id as citizen_id, l.id as location_id,
                   m.timestamp as timestamp, m.duration as duration,
                   m.purpose as purpose
            """
            result = session.run(movements_query)
            movements_df = pd.DataFrame([dict(record) for record in result])
        
        return {
            'citizens': citizens_df,
            'interactions': interactions_df,
            'movements': movements_df
        }
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data for demonstration"""
        n_citizens = 1000
        n_interactions = 5000
        n_movements = 3000
        
        # Citizens
        citizens_df = pd.DataFrame({
            'citizen_id': range(n_citizens),
            'name': [f'Citizen_{i:04d}' for i in range(n_citizens)],
            'age': np.random.gamma(4, 10, n_citizens).astype(int) + 18,
            'risk_score': np.random.beta(2, 5, n_citizens),
            'occupation': np.random.choice([
                'Engineer', 'Teacher', 'Doctor', 'Artist', 'Driver',
                'Merchant', 'Student', 'Retired', 'Unemployed'
            ], n_citizens),
            'location_name': np.random.choice([
                'Downtown', 'Suburbs', 'Industrial', 'Commercial'
            ], n_citizens),
            'location_crime_rate': np.random.beta(2, 5, n_citizens)
        })
        
        # Interactions
        interactions_df = pd.DataFrame({
            'citizen1_id': np.random.randint(0, n_citizens, n_interactions),
            'citizen2_id': np.random.randint(0, n_citizens, n_interactions),
            'timestamp': pd.date_range('2024-01-01', periods=n_interactions, freq='1h'),
            'interaction_type': np.random.choice(['social', 'work', 'family', 'random'], n_interactions),
            'frequency': np.random.randint(1, 50, n_interactions),
            'strength': np.random.uniform(0, 1, n_interactions)
        })
        
        # Movements
        movements_df = pd.DataFrame({
            'citizen_id': np.random.randint(0, n_citizens, n_movements),
            'location_id': np.random.randint(0, 10, n_movements),
            'timestamp': pd.date_range('2024-01-01', periods=n_movements, freq='2h'),
            'duration': np.random.randint(10, 480, n_movements),
            'purpose': np.random.choice(['work', 'leisure', 'shopping', 'home'], n_movements)
        })
        
        return {
            'citizens': citizens_df,
            'interactions': interactions_df,
            'movements': movements_df
        }
    
    def setup_duckdb_tables(self, data: Dict[str, pd.DataFrame]):
        """Load DataFrames into DuckDB"""
        for table_name, df in data.items():
            self.duckdb_conn.register(table_name, df)
            print(f"✓ Loaded {len(df):,} rows into {table_name} table")
    
    def analyze_risk_patterns(self) -> pd.DataFrame:
        """Analyze risk patterns using DuckDB"""
        query = """
        SELECT 
            occupation,
            location_name,
            COUNT(*) as count,
            AVG(risk_score) as avg_risk,
            MAX(risk_score) as max_risk,
            STDDEV(risk_score) as std_risk,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY risk_score) as median_risk,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY risk_score) as p95_risk
        FROM citizens
        GROUP BY occupation, location_name
        ORDER BY avg_risk DESC
        """
        return self.duckdb_conn.execute(query).df()
    
    def analyze_interaction_networks(self) -> pd.DataFrame:
        """Analyze interaction networks"""
        query = """
        SELECT 
            c1.occupation as occ1,
            c2.occupation as occ2,
            COUNT(*) as interaction_count,
            AVG(i.strength) as avg_strength,
            AVG(i.frequency) as avg_frequency,
            AVG(c1.risk_score + c2.risk_score) / 2 as avg_combined_risk
        FROM interactions i
        JOIN citizens c1 ON i.citizen1_id = c1.citizen_id
        JOIN citizens c2 ON i.citizen2_id = c2.citizen_id
        GROUP BY c1.occupation, c2.occupation
        HAVING interaction_count > 10
        ORDER BY avg_combined_risk DESC
        """
        return self.duckdb_conn.execute(query).df()
    
    def analyze_movement_patterns(self) -> pd.DataFrame:
        """Analyze movement patterns"""
        query = """
        SELECT 
            c.occupation,
            m.purpose,
            COUNT(*) as movement_count,
            AVG(m.duration) as avg_duration,
            AVG(c.risk_score) as avg_risk
        FROM movements m
        JOIN citizens c ON m.citizen_id = c.citizen_id
        GROUP BY c.occupation, m.purpose
        ORDER BY movement_count DESC
        """
        return self.duckdb_conn.execute(query).df()
    
    def identify_high_risk_clusters(self, risk_threshold: float = 0.7) -> pd.DataFrame:
        """Identify high-risk clusters using window functions"""
        query = f"""
        WITH risk_stats AS (
            SELECT 
                citizen_id,
                name,
                risk_score,
                occupation,
                location_name,
                RANK() OVER (PARTITION BY location_name ORDER BY risk_score DESC) as location_rank,
                PERCENT_RANK() OVER (ORDER BY risk_score) as percentile
            FROM citizens
            WHERE risk_score > {risk_threshold}
        )
        SELECT 
            location_name,
            COUNT(*) as high_risk_count,
            AVG(risk_score) as avg_risk,
            MAX(risk_score) as max_risk,
            STRING_AGG(name, ', ') as top_citizens
        FROM risk_stats
        WHERE location_rank <= 5
        GROUP BY location_name
        ORDER BY high_risk_count DESC, avg_risk DESC
        """
        return self.duckdb_conn.execute(query).df()
    
    def time_series_risk_analysis(self) -> pd.DataFrame:
        """Analyze risk evolution over time"""
        query = """
        SELECT 
            DATE_TRUNC('day', m.timestamp) as date,
            COUNT(DISTINCT m.citizen_id) as active_citizens,
            AVG(c.risk_score) as avg_risk,
            SUM(CASE WHEN c.risk_score > 0.7 THEN 1 ELSE 0 END) as high_risk_count
        FROM movements m
        JOIN citizens c ON m.citizen_id = c.citizen_id
        GROUP BY DATE_TRUNC('day', m.timestamp)
        ORDER BY date
        """
        return self.duckdb_conn.execute(query).df()
    
    def export_to_parquet(self, output_dir: Path):
        """Export processed data to Parquet format"""
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export main tables
        for table_name in ['citizens', 'interactions', 'movements']:
            output_file = output_dir / f'{table_name}.parquet'
            self.duckdb_conn.execute(f"""
                COPY {table_name} TO '{output_file}' (FORMAT PARQUET)
            """)
            print(f"✓ Exported {table_name} to {output_file}")
        
        # Export analysis results
        analyses = {
            'risk_patterns': self.analyze_risk_patterns(),
            'interaction_networks': self.analyze_interaction_networks(),
            'movement_patterns': self.analyze_movement_patterns(),
            'high_risk_clusters': self.identify_high_risk_clusters()
        }
        
        for name, df in analyses.items():
            output_file = output_dir / f'{name}.parquet'
            df.to_parquet(output_file)
            print(f"✓ Exported {name} analysis to {output_file}")


def main():
    """Run data processing example"""
    print("=" * 80)
    print("Data Processing with DuckDB and Ibis")
    print("=" * 80)
    
    # Initialize processor
    processor = DataProcessor()
    
    # Load data
    print("\n📊 Loading data...")
    data = processor.load_data_from_neo4j()
    
    # Setup DuckDB tables
    print("\n🔧 Setting up DuckDB tables...")
    processor.setup_duckdb_tables(data)
    
    # Run analyses
    print("\n📈 Analyzing risk patterns...")
    risk_patterns = processor.analyze_risk_patterns()
    print(f"\nTop 10 Risk Patterns:\n{risk_patterns.head(10)}")
    
    print("\n🔗 Analyzing interaction networks...")
    interactions = processor.analyze_interaction_networks()
    print(f"\nTop 10 Interaction Patterns:\n{interactions.head(10)}")
    
    print("\n🚶 Analyzing movement patterns...")
    movements = processor.analyze_movement_patterns()
    print(f"\nTop 10 Movement Patterns:\n{movements.head(10)}")
    
    print("\n⚠️  Identifying high-risk clusters...")
    clusters = processor.identify_high_risk_clusters()
    print(f"\nHigh-Risk Clusters:\n{clusters}")
    
    # Export results
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    print(f"\n💾 Exporting results to {output_dir}...")
    processor.export_to_parquet(output_dir)
    
    print("\n✅ Data processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
