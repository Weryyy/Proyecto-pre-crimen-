# 🌐 3D Force Graph Visualization - Guía Completa

## Descripción

Visualización interactiva del grafo de redes neuronales (GNN) usando **3D Force Graph** con three.js, d3-forces, y FastAPI como backend REST.

## Arquitectura

```
┌─────────────────┐
│    Neo4j DB     │ ← Base de datos de grafos
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  FastAPI Server │ ← REST API (Python)
│  api_server.py  │   Traduce Neo4j → JSON
└────────┬────────┘
         │
         ↓ HTTP/JSON
┌─────────────────┐
│  3D Force Graph │ ← Frontend (JavaScript)
│force_graph_3d.html│   three.js + d3-forces
└─────────────────┘
```

## Componentes

### 1. FastAPI Backend (`visualization/api_server.py`)

**Servidor REST que expone la información de Neo4j como JSON.**

#### Endpoints Disponibles:

##### `GET /api/graph`
Obtiene el grafo completo con nodos y enlaces.

**Parámetros:**
- `limit` (opcional): Número máximo de nodos (default: 200, max: 1000)

**Respuesta:**
```json
{
  "nodes": [
    {
      "id": "citizen_1",
      "name": "Citizen 001",
      "risk_score": 0.732,
      "age": 45,
      "occupation": "Engineer",
      "type": "citizen",
      "color": "#ff0000",
      "size": 15.98
    }
  ],
  "links": [
    {
      "source": "citizen_1",
      "target": "citizen_2",
      "type": "INTERACTS_WITH",
      "strength": 0.85
    }
  ]
}
```

##### `GET /api/nodes`
Solo nodos, opcionalmente filtrados.

**Parámetros:**
- `limit` (opcional): Número máximo de nodos
- `min_risk` (opcional): Filtro de riesgo mínimo (0.0-1.0)

##### `GET /api/stats`
Estadísticas del grafo.

**Respuesta:**
```json
{
  "total_nodes": 200,
  "total_links": 450,
  "high_risk_count": 23,
  "avg_risk": 0.421,
  "max_risk": 0.943
}
```

##### `GET /api/high-risk`
Subgrafo con solo ciudadanos de alto riesgo.

**Parámetros:**
- `threshold` (opcional): Umbral de riesgo (default: 0.7)

##### `GET /visualization`
Página HTML con la visualización 3D interactiva.

##### `GET /docs`
Documentación interactiva de la API (Swagger UI).

### 2. Frontend 3D (`visualization/force_graph_3d.html`)

**Visualización interactiva usando three.js y 3D Force Graph.**

#### Características:

**Visualización:**
- Grafo 3D con física en tiempo real (d3-forces)
- Rotación, zoom y pan con el mouse
- Colores basados en nivel de riesgo
- Tamaño de nodos proporcional al riesgo
- Partículas animadas en los enlaces
- Etiquetas con información al pasar el mouse

**Controles:**
- Barra de límite de nodos (50-500)
- Filtro de riesgo mínimo (0.0-1.0)
- Toggle mostrar/ocultar conexiones
- Toggle mostrar/ocultar etiquetas
- Filtro "Solo alto riesgo"
- Búsqueda por nombre o ID
- Botón de recarga
- Botón centrar vista
- Botón reset filtros

**Interacción:**
- Click en nodo: Muestra información detallada
- Click en nodo: Zoom automático al nodo
- Búsqueda: Enter para buscar y hacer zoom
- Panel de información con datos del nodo seleccionado

**Leyenda de Colores:**
- 🟢 Verde: Riesgo bajo (< 0.3)
- 🟡 Amarillo: Riesgo medio (0.3-0.5)
- 🟠 Naranja: Riesgo alto (0.5-0.7)
- 🔴 Rojo: Riesgo muy alto (> 0.7)

## Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Las nuevas dependencias añadidas:
- `fastapi>=0.109.0` - Framework web moderno
- `uvicorn[standard]>=0.27.0` - Servidor ASGI
- `pydantic>=2.5.0` - Validación de datos

### 2. Verificar Archivos

```bash
# Verificar que existen los archivos
ls -la visualization/api_server.py
ls -la visualization/force_graph_3d.html
```

## Uso

### Opción 1: Launcher Interactivo

```bash
python launch_dashboard.py
```

Selecciona opción **4** para lanzar el 3D Force Graph.

### Opción 2: Comando Directo

```bash
# Lanzar servidor FastAPI
python visualization/api_server.py
```

### Opción 3: Con Makefile

```bash
make dashboard-force-graph
```

### Opción 4: Con Argumento

```bash
python launch_dashboard.py --force-graph
```

## Acceso

Una vez iniciado el servidor, accede a:

- **Visualización 3D**: http://localhost:8001/visualization
- **API Root**: http://localhost:8001
- **Documentación API**: http://localhost:8001/docs
- **Endpoint Graph**: http://localhost:8001/api/graph
- **Estadísticas**: http://localhost:8001/api/stats

## Configuración

### Variables de Entorno

```bash
# Neo4j (si está disponible)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="precrime2024"
```

### Modo Mock Data

Si Neo4j no está disponible, el sistema automáticamente genera datos sintéticos:
- 100 ciudadanos con distribución Beta(2,5) de riesgo
- ~200 interacciones aleatorias
- Datos realistas de edad y ocupación

## Guía de Uso

### 1. Exploración Básica

1. Abre http://localhost:8001/visualization
2. El grafo se carga automáticamente con 200 nodos
3. Usa el mouse para:
   - **Rotar**: Click izquierdo + arrastrar
   - **Pan**: Click derecho + arrastrar
   - **Zoom**: Rueda del mouse

### 2. Filtrado de Datos

**Por límite de nodos:**
1. Mueve la barra "Node Limit"
2. Click en "🔄 Reload Data"

**Por riesgo mínimo:**
1. Mueve la barra "Min Risk Filter"
2. Click en "🔄 Reload Data"

**Solo alto riesgo:**
1. Activa "Highlight High Risk Only"
2. Se filtran automáticamente nodos con riesgo > 0.7

### 3. Búsqueda de Nodos

1. Escribe en el cuadro de búsqueda
2. Presiona Enter
3. El grafo hace zoom al primer resultado

### 4. Información de Nodos

1. Click en cualquier nodo
2. Se muestra panel con información:
   - ID del ciudadano
   - Nombre
   - Riesgo (coloreado)
   - Edad
   - Ocupación
   - Tipo

### 5. Controles de Vista

**Mostrar/Ocultar Conexiones:**
- Desmarca "Show Connections" para ver solo nodos

**Mostrar/Ocultar Etiquetas:**
- Desmarca "Show Labels" para mejor performance

**Centrar Vista:**
- Click en "📍 Center View" para volver al centro

**Reset Completo:**
- Click en "↻ Reset Filters" para valores por defecto

## API REST - Ejemplos

### cURL

```bash
# Obtener grafo completo
curl http://localhost:8001/api/graph?limit=100

# Solo nodos con alto riesgo
curl http://localhost:8001/api/nodes?min_risk=0.7

# Estadísticas
curl http://localhost:8001/api/stats

# Subgrafo de alto riesgo
curl http://localhost:8001/api/high-risk?threshold=0.6
```

### Python

```python
import requests

# Obtener datos del grafo
response = requests.get('http://localhost:8001/api/graph?limit=150')
data = response.json()

print(f"Nodos: {len(data['nodes'])}")
print(f"Enlaces: {len(data['links'])}")

# Analizar riesgo
for node in data['nodes']:
    if node['risk_score'] > 0.8:
        print(f"Alto riesgo: {node['name']} - {node['risk_score']}")
```

### JavaScript (Frontend)

```javascript
// Fetch graph data
fetch('http://localhost:8001/api/graph?limit=200')
  .then(response => response.json())
  .then(data => {
    console.log('Nodes:', data.nodes.length);
    console.log('Links:', data.links.length);
    
    // Procesar datos
    const highRisk = data.nodes.filter(n => n.risk_score > 0.7);
    console.log('High risk citizens:', highRisk.length);
  });
```

## Integración con Neo4j

### Consultas Cypher Utilizadas

**Obtener Ciudadanos:**
```cypher
MATCH (c:Citizen)
RETURN c.id as id, c.name as name, c.age as age,
       c.risk_seed as risk_score, c.occupation as occupation
LIMIT 200
```

**Obtener Interacciones:**
```cypher
MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
WHERE c1.id IN $citizen_ids AND c2.id IN $citizen_ids
RETURN c1.id as source, c2.id as target, 
       r.type as type, r.strength as strength
LIMIT 600
```

## Performance

### Optimizaciones Implementadas

1. **Límite de Nodos**: Default 200, ajustable hasta 1000
2. **Lazy Loading**: Solo carga lo necesario
3. **Client-side Filtering**: Filtros aplicados en navegador
4. **WebGL Rendering**: Aceleración por hardware
5. **Particle Optimization**: Solo 2 partículas por enlace
6. **Damping**: Suaviza movimientos (factor 0.1)

### Recomendaciones

- **< 300 nodos**: Performance óptimo
- **300-500 nodos**: Bueno, puede lag en dispositivos lentos
- **> 500 nodos**: Considerar filtros o agregación

## Troubleshooting

### Servidor no inicia

```bash
# Verificar puerto disponible
lsof -i :8001

# Matar proceso si existe
kill -9 <PID>

# Reinstalar dependencias
pip install fastapi uvicorn pydantic
```

### Visualización en blanco

1. Abre la consola del navegador (F12)
2. Verifica errores de red
3. Confirma que API está corriendo: http://localhost:8001
4. Verifica CORS no esté bloqueando

### Neo4j no conecta

El sistema automáticamente usa datos mock si Neo4j no está disponible.

Para conectar a Neo4j:
```bash
# Iniciar Neo4j
docker-compose up -d neo4j

# Verificar conexión
curl http://localhost:7474
```

### Performance lenta

1. Reduce número de nodos (mover barra de límite)
2. Desactiva "Show Connections"
3. Desactiva "Show Labels"
4. Usa filtro de riesgo mínimo
5. Cierra otras pestañas del navegador

## Comparación con Otros Dashboards

### vs Plotly Dash (Original)
- ✅ Mejor para exploración interactiva
- ✅ Física en tiempo real más realista
- ✅ Better WebGL performance
- ❌ No tiene layouts pre-calculados (t-SNE, PCA)

### vs Panel HoloViz
- ✅ Más especializado para grafos GNN
- ✅ API REST independiente
- ✅ Visualización 3D más fluida
- ❌ Menos opciones de análisis estadístico

### vs Integrated Dashboard
- ✅ Más ligero y rápido
- ✅ Mejor para demos y presentaciones
- ❌ No tiene DuckDB ni análisis avanzado
- ❌ No tiene mapas geográficos

## Personalización

### Modificar Colores

Edita `force_graph_3d.html`:

```javascript
function get_risk_color(risk_score) {
    if (risk_score < 0.3) return "#00ff00";  // Verde
    if (risk_score < 0.5) return "#ffff00";  // Amarillo
    if (risk_score < 0.7) return "#ffa500";  // Naranja
    return "#ff0000";  // Rojo
}
```

### Ajustar Física

En `api_server.py`:

```python
graph.d3Force('charge').strength(-120)  # Repulsión entre nodos
graph.d3Force('link').distance(50)      # Longitud de enlaces
```

### Cambiar Puerto

```python
# En api_server.py, función main()
uvicorn.run(app, host="0.0.0.0", port=8001)  # Cambiar 8001
```

## Extensiones Futuras

Posibles mejoras:
- [ ] Filtros temporales (evolución del grafo)
- [ ] Clustering automático visual
- [ ] Exportar grafo a imagen/video
- [ ] VR/AR support
- [ ] Integración con análisis en tiempo real
- [ ] WebSocket para updates live
- [ ] Capas adicionales (ubicaciones, eventos)

## Referencias

- **3D Force Graph**: https://github.com/vasturiano/3d-force-graph
- **three.js**: https://threejs.org
- **d3-force**: https://github.com/d3/d3-force
- **FastAPI**: https://fastapi.tiangolo.com
- **Neo4j**: https://neo4j.com

## Soporte

Para problemas o preguntas:
1. Revisa esta guía
2. Consulta logs del servidor
3. Abre consola del navegador (F12)
4. Verifica que Neo4j está corriendo (si lo usas)
5. Reporta issue en GitHub

---

**¡Disfruta explorando el grafo en 3D!** 🌐🎯
