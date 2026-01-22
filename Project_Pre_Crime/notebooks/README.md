# Jupyter Notebooks

Este directorio contiene notebooks de Jupyter para exploración interactiva del sistema.

## Acceso

Con Docker:
```bash
make jupyter
# O
docker-compose up -d jupyter
```

Luego accede a: **http://localhost:8888**

## Notebooks Sugeridos

### 1. Exploración de Datos
- Conectar a Neo4j
- Consultas Cypher
- Análisis de grafos
- Estadísticas de riesgo

### 2. Visualización Avanzada
- Gráficos personalizados
- Análisis de clusters
- Evolución temporal
- Métricas del modelo

### 3. Experimentación
- Entrenar modelos con diferentes parámetros
- Probar diferentes layouts
- Análisis de sensibilidad
- Validación cruzada

## Ejemplo de Notebook

```python
import sys
sys.path.append('/app/src')

from connector import Neo4jConnector
from models import RedGAN
import pandas as pd

# Conectar a Neo4j
connector = Neo4jConnector(
    uri='bolt://neo4j:7687',
    user='neo4j',
    password='precrime2024'
)

# Extraer datos
data = connector.extract_subgraph()
print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")

# Ver ciudadanos de alto riesgo
high_risk = connector.get_high_risk_citizens(threshold=0.5)
high_risk.head()
```

## Tips

- Los notebooks se guardan automáticamente en este directorio
- Tienen acceso completo a Neo4j y al código fuente
- Puedes instalar paquetes adicionales con `!pip install`
- Usa `%matplotlib inline` para gráficos en el notebook
