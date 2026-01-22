# 📑 Índice de Documentación - Proyecto Pre-Crimen

## 🚀 Para Empezar

### Español
1. **[QUICKSTART_ES.md](QUICKSTART_ES.md)** ⭐ **¡EMPIEZA AQUÍ!**
   - Guía rápida en español
   - Instrucciones paso a paso
   - Comandos esenciales
   - Troubleshooting común

### English
2. **[README.md](README.md)** 
   - Project overview
   - Features description
   - Installation options
   - Usage examples

---

## 🐳 Docker y Deployment

3. **[DOCKER.md](DOCKER.md)** ⭐ **Guía Completa de Docker**
   - Setup con Docker
   - Comandos de Docker Compose
   - Configuración de servicios
   - Troubleshooting Docker
   - Deploy en servidor

4. **[Makefile](Makefile)**
   - Comandos automatizados
   - Scripts de conveniencia
   - Alias útiles

5. **[docker-compose.yml](docker-compose.yml)**
   - Definición de servicios
   - Configuración de puertos
   - Volúmenes y redes

6. **[Dockerfile](Dockerfile)**
   - Imagen de la aplicación
   - Dependencias del sistema

---

## 🔧 Implementación Técnica

7. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** ⭐ **Detalles Técnicos**
   - Arquitectura del sistema
   - Explicación de componentes
   - Beta distribution
   - RedGAN architecture
   - Risk seed evolution

8. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** ⭐ **Resumen Ejecutivo**
   - Estadísticas del proyecto
   - Todas las features
   - Tecnologías usadas
   - Métricas y logros

---

## 💻 Código Fuente

### Módulos Principales

9. **[src/connector.py](src/connector.py)**
   - Conexión a Neo4j
   - Generación de datos sintéticos
   - Beta distribution implementation
   - Risk seed evolution
   - Extracción de subgrafos

10. **[src/models.py](src/models.py)**
    - GraphSAGE implementation
    - GAT (Graph Attention Network)
    - RedGAN architecture
    - Loss functions

11. **[src/train.py](src/train.py)**
    - Training pipeline
    - Risk evolution during training
    - Checkpoint management
    - Evaluation metrics

12. **[visualization/dashboard.py](visualization/dashboard.py)**
    - Dash web application
    - 3D visualization
    - Interactive controls
    - Layout algorithms (t-SNE, PCA, Spring)

---

## 📊 Base de Datos

13. **[scripts/setup_db.cypher](scripts/setup_db.cypher)**
    - Schema de Neo4j
    - Constraints y índices
    - Ejemplos de queries
    - Estructura del grafo

---

## 🎓 Ejemplos y Tutoriales

14. **[examples/demo_risk_evolution.py](examples/demo_risk_evolution.py)**
    - Demostración de Beta distribution
    - Simulación de evolución de riesgo
    - Generación de gráficos
    - Análisis estadístico

15. **[examples/usage_examples.py](examples/usage_examples.py)**
    - Uso programático del sistema
    - 6 ejemplos completos
    - Integración con otros proyectos
    - Consultas personalizadas

16. **[quickstart.py](quickstart.py)**
    - Verificación de dependencias
    - Tests básicos
    - Validación de instalación

---

## 📓 Jupyter Notebooks

17. **[notebooks/README.md](notebooks/README.md)**
    - Guía de uso de Jupyter
    - Ejemplos de notebooks
    - Tips y trucos

---

## 📦 Configuración

18. **[requirements.txt](requirements.txt)**
    - Todas las dependencias Python
    - Versiones específicas
    - Comentarios por categoría

19. **[.dockerignore](.dockerignore)**
    - Archivos excluidos del build
    - Optimización de imagen

---

## 🗺️ Mapa de Navegación

### Por Objetivo:

#### 🎯 Quiero empezar rápido
1. [QUICKSTART_ES.md](QUICKSTART_ES.md) → `make demo`
2. Abrir http://localhost:8050
3. ¡Listo!

#### 🔍 Quiero entender el sistema
1. [README.md](README.md) - Overview
2. [IMPLEMENTATION.md](IMPLEMENTATION.md) - Detalles técnicos
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Resumen completo

#### 🐳 Quiero usar Docker
1. [DOCKER.md](DOCKER.md) - Guía completa
2. [Makefile](Makefile) - Comandos disponibles
3. [docker-compose.yml](docker-compose.yml) - Configuración

#### 💻 Quiero programar
1. [examples/usage_examples.py](examples/usage_examples.py) - Ejemplos
2. [src/connector.py](src/connector.py) - API de datos
3. [src/models.py](src/models.py) - Modelos
4. [src/train.py](src/train.py) - Training

#### 📊 Quiero visualizar
1. [visualization/dashboard.py](visualization/dashboard.py) - Dashboard
2. `make visualize` - Iniciar
3. http://localhost:8050 - Acceder

#### 🗄️ Quiero consultar la base de datos
1. [scripts/setup_db.cypher](scripts/setup_db.cypher) - Schema
2. http://localhost:7474 - Neo4j Browser
3. Credenciales: neo4j/precrime2024

---

## 📚 Orden de Lectura Recomendado

### Para Usuarios:
1. **[QUICKSTART_ES.md](QUICKSTART_ES.md)** - Empezar
2. **[DOCKER.md](DOCKER.md)** - Profundizar en Docker
3. **[examples/usage_examples.py](examples/usage_examples.py)** - Ver ejemplos

### Para Desarrolladores:
1. **[README.md](README.md)** - Context
2. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Arquitectura
3. **[src/connector.py](src/connector.py)** - Datos
4. **[src/models.py](src/models.py)** - Modelos
5. **[src/train.py](src/train.py)** - Training
6. **[visualization/dashboard.py](visualization/dashboard.py)** - Viz

### Para Investigadores:
1. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Resumen
2. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detalles
3. **[examples/demo_risk_evolution.py](examples/demo_risk_evolution.py)** - Beta dist
4. **[src/models.py](src/models.py)** - Arquitectura GNN

---

## 🔗 Enlaces Rápidos

### Servicios Web (cuando está corriendo):
- **Visualización 3D**: http://localhost:8050
- **Neo4j Browser**: http://localhost:7474
- **Jupyter Notebook**: http://localhost:8888

### Comandos Útiles:
```bash
make help       # Ver todos los comandos
make demo       # Demo completo
make visualize  # Solo visualización
make train      # Entrenar modelo
make logs       # Ver logs
make down       # Detener todo
```

---

## 📞 Ayuda y Soporte

### Problemas Comunes:
- Ver **[DOCKER.md](DOCKER.md)** - Sección Troubleshooting
- Ver **[QUICKSTART_ES.md](QUICKSTART_ES.md)** - Sección Troubleshooting

### Para Más Ayuda:
- GitHub Issues
- Documentación de dependencias
- Comunidad de usuarios

---

## ✨ Documentos Destacados

### ⭐ Top 3 Más Importantes:
1. **[QUICKSTART_ES.md](QUICKSTART_ES.md)** - Para empezar
2. **[DOCKER.md](DOCKER.md)** - Para deployment
3. **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Para entender

### 📊 Para Visualización:
- [visualization/dashboard.py](visualization/dashboard.py)
- [QUICKSTART_ES.md](QUICKSTART_ES.md) - Sección Visualización

### 🧠 Para Machine Learning:
- [src/models.py](src/models.py)
- [src/train.py](src/train.py)
- [IMPLEMENTATION.md](IMPLEMENTATION.md)

### 🎲 Para Datos:
- [src/connector.py](src/connector.py)
- [scripts/setup_db.cypher](scripts/setup_db.cypher)
- [examples/usage_examples.py](examples/usage_examples.py)

---

## 📝 Notas

- Todos los documentos `.md` están en formato Markdown
- Los archivos Python (`.py`) contienen docstrings detallados
- El código está comentado en inglés
- La documentación de usuario está en español e inglés

---

**Última actualización**: 2026-01-22  
**Versión del proyecto**: 1.0.0  
**Total de archivos**: 19

---

¡Feliz coding! 🚀
