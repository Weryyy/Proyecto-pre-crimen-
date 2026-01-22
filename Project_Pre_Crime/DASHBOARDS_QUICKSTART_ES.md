# 🚀 Guía Rápida - Dashboards HoloViz

## Inicio Rápido

### Opción 1: Menú Interactivo (Recomendado)

```bash
python launch_dashboard.py
```

Se mostrará un menú con todas las opciones de dashboards disponibles.

### Opción 2: Comandos Directos

#### Dashboard Integrado (Recomendado) ⭐
```bash
python launch_dashboard.py --integrated
# o
python visualization/integrated_dashboard.py
```
**URL:** http://localhost:5007

**Características:**
- 🗺️ Mapas 3D con PyDeck
- 📊 Visualizaciones interactivas con HoloViews
- ⚡ Consultas SQL rápidas con DuckDB
- 🌍 Análisis geográfico con GeoViews
- 📈 Múltiples pestañas de visualización

#### Panel HoloViz Dashboard
```bash
python launch_dashboard.py --panel
# o
python visualization/panel_dashboard.py
```
**URL:** http://localhost:5006

**Características:**
- 🎨 Interfaz moderna con Panel
- 📊 Gráficos HoloViews
- 🗺️ Mapas GeoViews
- 📋 Tablas interactivas

#### Dashboard 3D Original (Plotly)
```bash
python launch_dashboard.py --plotly
# o
python visualization/dashboard.py
```
**URL:** http://localhost:8050

**Características:**
- 🌐 Visualización 3D del grafo
- 🔄 Múltiples layouts (t-SNE, PCA, Spring)
- 🔴 Detección de Red Balls
- ✨ Controles interactivos

### Opción 3: Make Commands

```bash
# Ver menú de dashboards
make dashboard-menu

# Dashboard integrado
make dashboard-integrated

# Panel HoloViz
make dashboard-panel

# Dashboard original Plotly
make dashboard-plotly

# Test de Mapbox
make dashboard-mapbox
```

## Instalación de Dependencias

### Instalación Completa
```bash
pip install -r requirements.txt
```

**Nota:** La instalación incluye 50+ paquetes y puede tardar 5-10 minutos.

### Instalación por Grupos (si hay problemas)

```bash
# 1. Dependencias básicas
pip install torch torch-geometric neo4j pandas numpy scipy scikit-learn

# 2. Visualización original
pip install plotly dash matplotlib jupyter notebook

# 3. HoloViz ecosystem
pip install panel holoviews hvplot geoviews datashader bokeh colorcet param

# 4. Geoespacial
pip install geopandas pydeck mapbox folium shapely

# 5. Procesamiento de datos
pip install duckdb ibis-framework polars pyarrow xarray

# 6. Machine Learning (opcional)
pip install optuna xgboost lightgbm catboost

# 7. Análisis de redes
pip install networkx

# 8. Utilidades
pip install python-dotenv requests
```

## Características de Cada Dashboard

### 1. Dashboard Integrado (Puerto 5007)

#### Pestañas:

**🗺️ Mapa 3D**
- Visualización PyDeck con agregación hexagonal
- Colores basados en nivel de riesgo
- Zoom, pan y rotación interactivos
- Tooltips con información detallada

**📊 Análisis de Riesgo**
- Histograma de distribución de riesgo
- Riesgo promedio por ocupación
- Gráfico de edad vs riesgo
- Estadísticas descriptivas

**🌍 Análisis Geográfico**
- Mapa GeoViews con capas de tiles
- Puntos coloreados por riesgo
- Marcadores de ubicaciones
- Información al pasar el mouse

**📋 Tabla de Datos**
- Top 100 ciudadanos de alto riesgo
- Ordenable y filtrable
- Paginación
- Exportación de datos

#### Panel Lateral:

- **Actualizar Datos**: Recarga desde Neo4j
- **Umbral de Riesgo**: Filtro deslizante (0-1)
- **Filtrar por Ocupación**: Selección múltiple
- **Mostrar Hexágonos**: Toggle de agregación 3D
- **Información**: Estadísticas y metadatos

#### Tarjetas de Estadísticas:

- Total de ciudadanos
- Alto riesgo (>0.7)
- Riesgo medio (0.5-0.7)
- Bajo riesgo (<0.5)

### 2. Panel HoloViz Dashboard (Puerto 5006)

- Diseño similar con énfasis en HoloViews
- Mapas GeoViews integrados
- Gráficos hvPlot interactivos
- Tablas Tabulator
- Indicadores y medidores Panel

### 3. Dashboard Original Plotly (Puerto 8050)

- Grafo 3D interactivo
- Layouts t-SNE/PCA/Spring
- Detección de Red Balls
- Conexiones de red
- Controles de umbral

## Análisis de Datos con DuckDB

### Ejecutar Script de Análisis

```bash
python examples/duckdb_analysis.py
```

**Qué hace:**
1. Carga datos desde Neo4j (o genera datos mock)
2. Crea tablas DuckDB
3. Ejecuta múltiples análisis
4. Muestra resultados
5. Exporta a archivos Parquet en `data/processed/`

### Análisis Incluidos:

- **Patrones de Riesgo**: Riesgo promedio por ocupación y ubicación
- **Redes de Interacción**: Análisis de conexiones sociales
- **Patrones de Movimiento**: Análisis de desplazamientos
- **Clusters de Alto Riesgo**: Identificación de zonas peligrosas
- **Series Temporales**: Evolución del riesgo en el tiempo

### Uso Programático:

```python
from examples.duckdb_analysis import DataProcessor

# Inicializar procesador
processor = DataProcessor(connector)

# Cargar datos
data = processor.load_data_from_neo4j()
processor.setup_duckdb_tables(data)

# Ejecutar análisis
risk_patterns = processor.analyze_risk_patterns()
interactions = processor.analyze_interaction_networks()
movements = processor.analyze_movement_patterns()
clusters = processor.identify_high_risk_clusters()

# Exportar resultados
processor.export_to_parquet(output_dir)
```

## Visualización de Mapas con Mapbox

### Generar Visualización HTML

```bash
python visualization/mapbox.py
```

Genera `examples/mapbox_test.html` con:
- Mapa PyDeck interactivo
- Puntos de ciudadanos coloreados por riesgo
- Marcadores de ubicaciones
- Tooltips con información

### Uso Programático:

```python
from visualization.mapbox import MapboxVisualizer

viz = MapboxVisualizer()

# Crear GeoDataFrames
citizens_gdf, locations_gdf = viz.create_geodataframe(
    citizens_data, 
    locations_data
)

# Crear mapa
deck = viz.create_pydeck_map(
    citizens_gdf, 
    locations_gdf,
    show_heatmap=True
)

# Guardar HTML
deck.to_html('mi_mapa.html')

# O crear mapa hexagonal
hex_deck = viz.create_hexagon_map(citizens_gdf)
hex_deck.to_html('mapa_hexagonal.html')
```

## Variables de Entorno

```bash
# Conexión Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="precrime2024"

# Token Mapbox (opcional)
export MAPBOX_TOKEN="tu_token_mapbox_aqui"
```

## Solución de Problemas

### Dashboard No Carga

```bash
# Verificar instalación
pip list | grep panel
pip list | grep holoviews

# Reinstalar si es necesario
pip install -r requirements.txt
```

### Neo4j No Conecta

```bash
# Verificar que Neo4j está corriendo
docker-compose ps neo4j

# Verificar conexión
python -c "from src.connector import Neo4jConnector; c = Neo4jConnector(); print('OK')"
```

### Puerto en Uso

```bash
# Ver qué proceso usa el puerto
lsof -i :5007

# Matar proceso
kill -9 <PID>
```

### Modo de Datos Mock

Si Neo4j no está disponible, los dashboards usan automáticamente datos sintéticos:
- 200-1000 ciudadanos sintéticos
- Distribución realista de riesgo
- Coordenadas geográficas aleatorias
- Interacciones y movimientos de muestra

## Comandos Útiles

```bash
# Ver ayuda de launcher
python launch_dashboard.py --help

# Ver todos los dashboards disponibles
python launch_dashboard.py --all

# Menú interactivo
python launch_dashboard.py

# Hacer análisis de datos
make duckdb-analysis

# Instalar dependencias
make install-deps
```

## Consejos de Rendimiento

1. **Usa DuckDB para datasets grandes**: Mucho más rápido que pandas
2. **Activa agregación hexagonal**: Mejor rendimiento con muchos puntos
3. **Limita tamaño de página de tabla**: Reduce tiempo de render inicial
4. **Usa formato Parquet**: I/O más rápido que CSV
5. **Aplica filtros temprano**: Usa filtros en queries SQL

## Próximos Pasos

1. **Explora la visualización**: Juega con los controles
2. **Consulta Neo4j**: Prueba queries en el browser
3. **Ejecuta análisis**: `make duckdb-analysis`
4. **Lee la documentación**: [HOLOVIZ_INTEGRATION.md](HOLOVIZ_INTEGRATION.md)
5. **Personaliza**: Modifica parámetros en los scripts

## Recursos

- [Documentación Panel](https://panel.holoviz.org)
- [Documentación HoloViews](https://holoviews.org)
- [Documentación PyDeck](https://deckgl.readthedocs.io)
- [Documentación DuckDB](https://duckdb.org/docs)
- [Documentación completa](HOLOVIZ_INTEGRATION.md)

## Soporte

Para problemas o preguntas:
1. Revisa esta guía
2. Consulta [HOLOVIZ_INTEGRATION.md](HOLOVIZ_INTEGRATION.md)
3. Verifica logs para mensajes de error
4. Abre un issue en GitHub

---

**¡Disfruta explorando los dashboards!** 🎯🗺️📊
