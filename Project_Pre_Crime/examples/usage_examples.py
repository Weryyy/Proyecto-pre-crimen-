"""
Ejemplo de Uso Programático del Sistema

Este script muestra cómo usar el sistema de predicción pre-crimen
programáticamente sin Docker, para integración en otros proyectos.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from connector import Neo4jConnector
from models import RedGAN
import torch
import numpy as np


def ejemplo_basico():
    """Ejemplo básico: Conectar y consultar"""
    print("=" * 60)
    print("Ejemplo 1: Conexión y Consulta Básica")
    print("=" * 60)
    
    # Conectar a Neo4j
    connector = Neo4jConnector(
        uri='bolt://localhost:7687',
        user='neo4j',
        password='precrime2024'  # o tu password
    )
    
    try:
        # Consultar ciudadanos de alto riesgo
        high_risk = connector.get_high_risk_citizens(threshold=0.5)
        print(f"\nCiudadanos de alto riesgo encontrados: {len(high_risk)}")
        print(high_risk.head())
        
        # Extraer grafo completo
        data = connector.extract_subgraph()
        print(f"\nGrafo extraído:")
        print(f"  Nodos: {data.num_nodes}")
        print(f"  Aristas: {data.num_edges}")
        print(f"  Features por nodo: {data.x.shape[1]}")
        
    finally:
        connector.close()


def ejemplo_generacion_datos():
    """Ejemplo: Generar datos sintéticos"""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Generación de Datos Sintéticos")
    print("=" * 60)
    
    connector = Neo4jConnector()
    
    try:
        # Generar pequeño conjunto de datos
        print("\nGenerando datos sintéticos...")
        connector.generate_synthetic_data(
            num_citizens=20,
            num_locations=5,
            num_interactions=30,
            num_movements=40
        )
        print("✓ Datos generados")
        
        # Extraer y mostrar
        data = connector.extract_subgraph()
        print(f"\nDatos generados:")
        print(f"  Ciudadanos: {data.num_nodes}")
        print(f"  Conexiones: {data.num_edges}")
        
    finally:
        connector.close()


def ejemplo_risk_evolution():
    """Ejemplo: Evolución de riesgo"""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Evolución de Risk Seed")
    print("=" * 60)
    
    connector = Neo4jConnector()
    
    # Generar risk seed inicial
    initial_risk = connector.generate_risk_seed(alpha=2, beta=5)
    print(f"\nRiesgo inicial: {initial_risk:.4f}")
    
    # Simular evolución a lo largo del tiempo
    current_risk = initial_risk
    print("\nEvolución del riesgo:")
    print(f"Día 0: {current_risk:.4f}")
    
    for day in range(1, 11):
        # Simular interacciones y ubicaciones
        interactions = np.random.randint(0, 10)
        crime_rate = np.random.beta(2, 8)
        
        # Evolucionar riesgo
        current_risk = connector.evolve_risk_seed(
            current_risk=current_risk,
            interactions=interactions,
            location_crime_rate=crime_rate,
            time_delta=1.0
        )
        
        print(f"Día {day}: {current_risk:.4f} "
              f"(interacciones={interactions}, crime_rate={crime_rate:.3f})")


def ejemplo_modelo():
    """Ejemplo: Usar el modelo RedGAN"""
    print("\n" + "=" * 60)
    print("Ejemplo 4: Predicción con RedGAN")
    print("=" * 60)
    
    connector = Neo4jConnector()
    
    try:
        # Extraer datos
        data = connector.extract_subgraph()
        print(f"\nDatos cargados: {data.num_nodes} nodos")
        
        # Crear modelo
        model = RedGAN(
            in_channels=data.x.shape[1],
            hidden_channels=32,
            embedding_dim=16
        )
        
        # Modo evaluación
        model.eval()
        
        # Hacer predicción
        with torch.no_grad():
            output = model(data.x, data.edge_index)
            
            risk_scores = output['risk_scores']
            red_balls = output['red_balls']
            
            print(f"\nPredicciones:")
            print(f"  Riesgo promedio: {risk_scores.mean():.4f}")
            print(f"  Riesgo máximo: {risk_scores.max():.4f}")
            print(f"  Red Balls detectados: {red_balls.sum()}/{data.num_nodes}")
            
            # Top 5 de mayor riesgo
            top_k = 5
            top_risks, top_indices = torch.topk(risk_scores.squeeze(), top_k)
            
            print(f"\nTop {top_k} individuos de mayor riesgo:")
            for i, (idx, risk) in enumerate(zip(top_indices, top_risks), 1):
                print(f"  {i}. Nodo {idx.item()}: {risk.item():.4f}")
        
    finally:
        connector.close()


def ejemplo_visualizacion_api():
    """Ejemplo: Usar API de visualización"""
    print("\n" + "=" * 60)
    print("Ejemplo 5: Preparar Datos para Visualización")
    print("=" * 60)
    
    connector = Neo4jConnector()
    
    try:
        # Extraer datos con información completa
        data = connector.extract_subgraph()
        
        # Obtener información de nodos
        with connector.driver.session() as session:
            query = """
            MATCH (c:Citizen)
            RETURN c.id as id, c.name as name, c.age as age, 
                   c.risk_seed as risk, c.occupation as occupation
            LIMIT 10
            """
            result = session.run(query)
            node_info = [record.data() for record in result]
        
        print("\nInformación de nodos para visualización:")
        for info in node_info[:5]:
            print(f"  {info['name']}: risk={info['risk']:.3f}, "
                  f"age={info['age']}, job={info['occupation']}")
        
        # Datos listos para exportar
        print(f"\nDatos preparados:")
        print(f"  Total nodos: {len(node_info)}")
        print(f"  Grafo PyG: {data}")
        
    finally:
        connector.close()


def ejemplo_consulta_personalizada():
    """Ejemplo: Consultas personalizadas a Neo4j"""
    print("\n" + "=" * 60)
    print("Ejemplo 6: Consultas Personalizadas")
    print("=" * 60)
    
    connector = Neo4jConnector()
    
    try:
        with connector.driver.session() as session:
            # Consulta 1: Ciudadanos por ocupación
            query1 = """
            MATCH (c:Citizen)
            RETURN c.occupation as occupation, 
                   avg(c.risk_seed) as avg_risk,
                   count(*) as count
            GROUP BY c.occupation
            ORDER BY avg_risk DESC
            LIMIT 5
            """
            print("\nRiesgo promedio por ocupación:")
            result = session.run(query1)
            for record in result:
                print(f"  {record['occupation']}: {record['avg_risk']:.3f} "
                      f"({record['count']} personas)")
            
            # Consulta 2: Ubicaciones peligrosas
            query2 = """
            MATCH (l:Location)
            RETURN l.name as location, l.crime_rate as crime_rate
            ORDER BY l.crime_rate DESC
            LIMIT 5
            """
            print("\nUbicaciones más peligrosas:")
            result = session.run(query2)
            for record in result:
                print(f"  {record['location']}: {record['crime_rate']:.3f}")
            
            # Consulta 3: Redes de alto riesgo
            query3 = """
            MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
            WHERE c1.risk_seed > 0.5 AND c2.risk_seed > 0.5
            RETURN count(*) as high_risk_connections
            """
            result = session.run(query3)
            record = result.single()
            print(f"\nConexiones entre individuos de alto riesgo: "
                  f"{record['high_risk_connections']}")
    
    finally:
        connector.close()


def main():
    """Ejecutar todos los ejemplos"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "EJEMPLOS DE USO PROGRAMÁTICO" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        # Nota: Requiere Neo4j corriendo
        ejemplo_basico()
        ejemplo_risk_evolution()
        ejemplo_modelo()
        ejemplo_visualizacion_api()
        ejemplo_consulta_personalizada()
        
        print("\n" + "=" * 60)
        print("✓ Todos los ejemplos completados!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nAsegúrate de que Neo4j esté corriendo:")
        print("  docker-compose up -d neo4j")
        print("  # O")
        print("  make up")


if __name__ == "__main__":
    main()
