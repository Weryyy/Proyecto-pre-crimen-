# Docker Setup Guide - Pre-Crime Prediction System

## 🐳 Guía Rápida con Docker

Este proyecto está completamente containerizado con Docker para que puedas ejecutarlo desde cualquier lugar sin preocuparte por dependencias.

## Prerrequisitos

- **Docker** (versión 20.10+)
- **Docker Compose** (versión 2.0+)
- **Make** (opcional, para comandos simplificados)

### Instalar Docker

#### Windows/Mac
1. Descargar [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Instalar y ejecutar Docker Desktop

#### Linux
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER
```

## 🚀 Inicio Rápido

### Opción 1: Usando Make (Recomendado)

```bash
# Ver todos los comandos disponibles
make help

# Setup inicial + visualización
make demo

# O paso por paso:
make build      # Construir imágenes
make setup      # Configurar base de datos
make visualize  # Iniciar visualización 3D
```

### Opción 2: Usando Docker Compose directamente

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 📊 Servicios Disponibles

Una vez iniciado, tendrás acceso a:

| Servicio | URL | Descripción |
|----------|-----|-------------|
| **Neo4j Browser** | http://localhost:7474 | Interfaz web de Neo4j |
| **Visualización 3D** | http://localhost:8050 | Dashboard interactivo 3D |
| **Jupyter Notebook** | http://localhost:8888 | Notebooks para exploración |

### Credenciales Neo4j
- **Usuario**: `neo4j`
- **Password**: `precrime2024`

## 🎯 Uso Básico

### 1. Visualización 3D Interactiva

```bash
make visualize
# O
docker-compose up neo4j app
```

Abre tu navegador en **http://localhost:8050** y verás:

- **Grafo 3D interactivo** con todos los ciudadanos
- **Nodos coloreados** por nivel de riesgo
- **Red Balls** destacados (individuos de alto riesgo)
- **Conexiones** entre ciudadanos
- **Controles interactivos**:
  - Método de layout (t-SNE, PCA, Spring)
  - Umbral de riesgo ajustable
  - Toggle de conexiones
  - Botón de actualización

### 2. Entrenar el Modelo

```bash
make train
# O
docker-compose run --rm app python src/train.py
```

### 3. Exploración con Jupyter

```bash
make jupyter
# O
docker-compose up -d neo4j jupyter
```

Abre **http://localhost:8888** y crea un notebook nuevo.

### 4. Consultas Neo4j

Abre **http://localhost:7474** y prueba:

```cypher
// Ver todos los ciudadanos
MATCH (c:Citizen)
RETURN c.name, c.risk_seed, c.age
ORDER BY c.risk_seed DESC
LIMIT 10

// Ver red de alto riesgo
MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
WHERE c1.risk_seed > 0.5 AND c2.risk_seed > 0.5
RETURN c1, r, c2
```

## 🔧 Comandos Útiles

```bash
# Ver estado de servicios
make status
docker-compose ps

# Ver logs
make logs           # Todos los servicios
make logs-app       # Solo aplicación
make logs-neo4j     # Solo Neo4j

# Reiniciar servicios
make restart
docker-compose restart

# Detener servicios
make down
docker-compose down

# Limpiar todo (incluye volúmenes)
make clean
docker-compose down -v

# Shell en contenedor
make shell
docker-compose run --rm app /bin/bash
```

## 📁 Estructura de Volúmenes

Los siguientes directorios están montados como volúmenes:

```
Project_Pre_Crime/
├── checkpoints/    → /app/checkpoints  (modelos guardados)
├── data/          → /app/data          (datos exportados)
├── src/           → /app/src           (código fuente)
└── notebooks/     → /app/notebooks     (Jupyter notebooks)
```

Esto significa que:
- ✅ Los modelos entrenados persisten fuera del contenedor
- ✅ Puedes editar código y ver cambios sin reconstruir
- ✅ Los notebooks se guardan en tu máquina local

## 🎨 Características de la Visualización 3D

### Layouts Disponibles

1. **t-SNE** (Recomendado)
   - Preserva la estructura local
   - Agrupa nodos similares
   - Mejor para identificar clusters

2. **PCA**
   - Rápido
   - Preserva varianza global
   - Bueno para overview general

3. **Spring Layout**
   - Force-directed
   - Muestra conexiones naturales
   - Interactivo

### Interacción

- **Rotar**: Click izquierdo + arrastrar
- **Pan**: Click derecho + arrastrar
- **Zoom**: Scroll del mouse
- **Hover**: Ver información de nodos
- **Seleccionar**: Click en nodo

### Colores

- **Gradiente Viridis**: Riesgo bajo (morado) → alto (amarillo)
- **Rojo brillante**: Red Balls (alto riesgo)
- **Gris transparente**: Conexiones

## 🔄 Workflow Típico

### Desarrollo

```bash
# 1. Iniciar servicios
make up

# 2. Generar datos
make setup

# 3. Ver visualización
# Abrir http://localhost:8050

# 4. Entrenar modelo
make train

# 5. Actualizar visualización
# Refrescar navegador o click en "Refresh Data"

# 6. Explorar en Neo4j
# Abrir http://localhost:7474
```

### Producción

```bash
# Construir y ejecutar
docker-compose up -d

# Verificar salud
docker-compose ps

# Ver logs
docker-compose logs -f app
```

## 🐛 Troubleshooting

### Neo4j no se conecta

```bash
# Verificar que esté corriendo
docker-compose ps

# Esperar unos segundos
docker-compose logs neo4j

# Reintentar
docker-compose restart neo4j
```

### Puerto ya en uso

Editar `docker-compose.yml` y cambiar puertos:

```yaml
ports:
  - "8051:8050"  # Cambiar 8050 → 8051
```

### Visualización no carga

```bash
# Verificar logs
docker-compose logs app

# Regenerar datos
make setup

# Reiniciar
docker-compose restart app
```

### Memoria insuficiente

Editar `docker-compose.yml`:

```yaml
neo4j:
  environment:
    - NEO4J_dbms_memory_heap_max__size=1G  # Reducir de 2G
```

## 📝 Variables de Entorno

Puedes personalizar la configuración creando un archivo `.env`:

```bash
# .env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=precrime2024
DASH_PORT=8050
JUPYTER_PORT=8888
```

## 🔐 Seguridad

Para producción, cambia las credenciales en `docker-compose.yml`:

```yaml
environment:
  - NEO4J_AUTH=neo4j/tu_password_segura
```

## 📦 Exportar/Importar Datos

### Exportar

```bash
docker-compose exec neo4j neo4j-admin dump \
  --database=neo4j \
  --to=/data/backup.dump
```

### Importar

```bash
docker-compose exec neo4j neo4j-admin load \
  --from=/data/backup.dump \
  --database=neo4j
```

## 🚀 Deploy en Servidor

### Usando Docker

```bash
# En servidor remoto
git clone <repo>
cd Project_Pre_Crime
docker-compose up -d

# Acceder por IP del servidor
http://<server-ip>:8050
```

### Nginx Reverse Proxy (opcional)

```nginx
server {
    listen 80;
    server_name precrime.example.com;

    location / {
        proxy_pass http://localhost:8050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 📚 Recursos Adicionales

- [Docker Documentation](https://docs.docker.com/)
- [Neo4j Docker](https://neo4j.com/developer/docker/)
- [Plotly Dash](https://dash.plotly.com/)

## 💡 Tips

1. **Primera vez**: Usa `make demo` para setup completo
2. **Desarrollo**: Monta `src/` como volumen para hot-reload
3. **Producción**: Usa `docker-compose.prod.yml` con optimizaciones
4. **Backup**: Exporta datos Neo4j regularmente
5. **Monitoreo**: Usa `docker stats` para ver uso de recursos

---

¿Problemas? Abre un issue en GitHub o consulta la documentación completa.
