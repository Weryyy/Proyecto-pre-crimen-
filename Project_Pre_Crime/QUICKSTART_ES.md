# 🚀 Guía de Inicio Rápido

## Sistema de Predicción Pre-Crimen con Docker

Esta guía te ayudará a ejecutar el sistema completo en minutos usando Docker.

## ⚡ Inicio Ultra-Rápido

```bash
# 1. Construir todo
make build

# 2. Demo completo (automático)
make demo
```

¡Eso es todo! El sistema:
- ✅ Inicia Neo4j
- ✅ Crea el esquema de la base de datos
- ✅ Genera 100 ciudadanos sintéticos con datos realistas
- ✅ Crea 200 interacciones sociales
- ✅ Simula 300 movimientos entre ubicaciones
- ✅ Inicia la visualización 3D

## 🌐 Acceder a los Servicios

Una vez iniciado, abre tu navegador:

### 1. Visualización 3D (Principal)
**http://localhost:8050**

Aquí verás:
- Grafo 3D interactivo con todos los ciudadanos
- Nodos coloreados por nivel de riesgo
- "Red Balls" (individuos de alto riesgo) destacados en rojo
- Controles para ajustar el umbral de riesgo
- Diferentes algoritmos de layout (t-SNE, PCA, Spring)
- Estadísticas en tiempo real

**Controles:**
- 🖱️ **Rotar**: Click izquierdo + arrastrar
- 🖱️ **Pan**: Click derecho + arrastrar  
- 🖱️ **Zoom**: Rueda del mouse
- 👆 **Info**: Hover sobre nodos

### 2. Neo4j Browser
**http://localhost:7474**

Credenciales:
- Usuario: `neo4j`
- Password: `precrime2024`

Consultas ejemplo:
```cypher
// Ver ciudadanos de alto riesgo
MATCH (c:Citizen)
WHERE c.risk_seed > 0.5
RETURN c.name, c.risk_seed, c.occupation
ORDER BY c.risk_seed DESC
LIMIT 10

// Red de alto riesgo
MATCH (c1:Citizen)-[r:INTERACTS_WITH]->(c2:Citizen)
WHERE c1.risk_seed > 0.5 AND c2.risk_seed > 0.5
RETURN c1, r, c2
```

### 3. Jupyter Notebook (Opcional)
**http://localhost:8888**

Para análisis y experimentación avanzada.

## 🎮 Comandos Útiles

```bash
# Ver todos los comandos disponibles
make help

# Iniciar servicios
make up

# Ver logs en tiempo real
make logs

# Detener todo
make down

# Entrenar modelo
make train

# Limpiar todo (reset completo)
make clean
```

## 📊 Entender la Visualización

### Colores de los Nodos
- **Morado/Azul**: Riesgo bajo (0.0 - 0.3)
- **Verde/Amarillo**: Riesgo medio (0.3 - 0.5)
- **Amarillo/Rojo**: Riesgo alto (0.5 - 1.0)
- **Rojo Brillante (Diamante)**: Red Ball (alto riesgo detectado)

### Layouts Disponibles

1. **t-SNE** (Recomendado)
   - Agrupa nodos similares
   - Mejor para identificar clusters
   - Tarda ~10 segundos

2. **PCA**
   - Más rápido
   - Vista general
   - Instantáneo

3. **Spring Layout**
   - Basado en fuerzas
   - Muestra conexiones naturales
   - Bueno para redes pequeñas

### Controles Interactivos

- **Layout Method**: Cambia el algoritmo de posicionamiento
- **Risk Threshold**: Ajusta qué se considera "Red Ball"
- **Show Edges**: Toggle para ver/ocultar conexiones
- **Refresh Data**: Recarga datos desde Neo4j

## 🔧 Troubleshooting

### "Puerto ya en uso"
```bash
# Ver qué está usando el puerto
sudo lsof -i :8050
# O cambiar el puerto en docker-compose.yml
```

### "Neo4j no se conecta"
```bash
# Ver logs de Neo4j
make logs-neo4j

# Esperar más tiempo
docker-compose logs neo4j | grep "Started"

# Reiniciar
make restart
```

### "No veo datos"
```bash
# Regenerar datos
make setup

# Verificar en Neo4j
# Ir a http://localhost:7474 y ejecutar:
# MATCH (n) RETURN count(n)
```

### "La visualización está en blanco"
```bash
# Ver logs de la app
make logs-app

# Reiniciar visualización
docker-compose restart app
```

## 📝 Flujo de Trabajo Típico

### Día a Día
```bash
# Iniciar
make up

# Trabajar...
# Hacer cambios en src/

# Reiniciar para ver cambios
make restart

# Ver logs si hay problemas
make logs-app

# Detener al terminar
make down
```

### Experimentación
```bash
# Usar Jupyter
make jupyter
# Abrir http://localhost:8888

# O entrenar modelo
make train

# Ver resultados en visualización
# http://localhost:8050 + click "Refresh Data"
```

### Demo para Presentación
```bash
# Setup completo
make demo

# Mostrar:
# 1. http://localhost:8050 - Visualización 3D
# 2. Cambiar layouts
# 3. Ajustar threshold
# 4. Hover sobre Red Balls
# 5. http://localhost:7474 - Consultas Neo4j
```

## 🎯 Próximos Pasos

1. **Explora la visualización**: Juega con los controles
2. **Consulta Neo4j**: Prueba queries en el browser
3. **Entrena el modelo**: `make train`
4. **Lee la documentación**: `DOCKER.md` y `README.md`
5. **Experimenta**: Modifica parámetros en `src/`

## 💡 Tips Pro

1. **Datos frescos**: Ejecuta `make setup` para regenerar datos
2. **Hot reload**: Los cambios en `src/` se reflejan al reiniciar
3. **Múltiples vistas**: Abre varios navegadores con diferentes layouts
4. **Performance**: Usa PCA para grafos grandes (>200 nodos)
5. **Screenshots**: Plotly tiene botón de descarga de PNG

## 🆘 Ayuda

- Ver logs: `make logs`
- Estado: `make status`
- Shell: `make shell`
- Documentación: `DOCKER.md`
- Issues: GitHub

---

**¡Disfruta explorando la red de pre-crimen!** 🎭🔮
