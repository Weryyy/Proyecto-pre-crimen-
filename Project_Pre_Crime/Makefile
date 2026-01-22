# Makefile para Pre-Crime Prediction System

.PHONY: help build up down restart logs clean train visualize jupyter test

help: ## Mostrar ayuda
	@echo "Pre-Crime Prediction System - Comandos Docker"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Construir imágenes Docker
	docker-compose build

up: ## Iniciar todos los servicios
	docker-compose up -d
	@echo ""
	@echo "✓ Servicios iniciados!"
	@echo ""
	@echo "Acceso a los servicios:"
	@echo "  • Neo4j Browser:         http://localhost:7474"
	@echo "  • Visualización 3D:      http://localhost:8050"
	@echo "  • 🆕 Panel Dashboard:    http://localhost:5006"
	@echo "  • 🆕 Integrated Dashboard: http://localhost:5007"
	@echo "  • Jupyter Notebook:      http://localhost:8888"
	@echo ""
	@echo "Credenciales Neo4j:"
	@echo "  Usuario: neo4j"
	@echo "  Password: precrime2024"
	@echo ""

down: ## Detener todos los servicios
	docker-compose down

restart: ## Reiniciar todos los servicios
	docker-compose restart

logs: ## Ver logs de todos los servicios
	docker-compose logs -f

logs-app: ## Ver logs de la aplicación
	docker-compose logs -f app

logs-neo4j: ## Ver logs de Neo4j
	docker-compose logs -f neo4j

clean: ## Limpiar volúmenes y contenedores
	docker-compose down -v
	rm -rf checkpoints/*.pth checkpoints/*.ckpt

train: ## Ejecutar entrenamiento
	docker-compose run --rm app python src/train.py

visualize: ## Iniciar solo visualización
	docker-compose up -d neo4j
	@echo "Esperando a que Neo4j esté listo..."
	@sleep 10
	docker-compose up app

jupyter: ## Iniciar solo Jupyter
	docker-compose up -d neo4j jupyter
	@echo ""
	@echo "✓ Jupyter Notebook iniciado!"
	@echo "  Acceso: http://localhost:8888"
	@echo ""

test: ## Ejecutar tests básicos
	docker-compose run --rm app python quickstart.py

setup: ## Setup inicial completo
	@echo "Configurando Pre-Crime Prediction System..."
	docker-compose up -d neo4j
	@echo "Esperando a que Neo4j esté listo..."
	@sleep 15
	@echo "Generando datos sintéticos..."
	docker-compose run --rm app python -c "from src.connector import Neo4jConnector; import os; c = Neo4jConnector(uri=os.getenv('NEO4J_URI'), user=os.getenv('NEO4J_USER'), password=os.getenv('NEO4J_PASSWORD')); c.setup_schema('../scripts/setup_db.cypher'); c.generate_synthetic_data(num_citizens=100, num_locations=20, num_interactions=200, num_movements=300); c.close()"
	@echo "✓ Setup completado!"
	@echo ""
	@echo "Siguiente paso: make visualize"

demo: ## Demo completo (setup + visualización)
	$(MAKE) setup
	$(MAKE) visualize

status: ## Ver estado de los servicios
	docker-compose ps

shell: ## Shell en el contenedor de la aplicación
	docker-compose run --rm app /bin/bash

# 🆕 New HoloViz Dashboard Commands

dashboard-menu: ## Menú interactivo de dashboards
	python launch_dashboard.py

dashboard-plotly: ## Dashboard 3D original (Plotly)
	python visualization/dashboard.py

dashboard-panel: ## Panel HoloViz dashboard
	python visualization/panel_dashboard.py

dashboard-integrated: ## Dashboard integrado (recomendado)
	python visualization/integrated_dashboard.py

dashboard-mapbox: ## Test de Mapbox (genera HTML)
	python visualization/mapbox.py

install-deps: ## Instalar dependencias de Python
	pip install -r requirements.txt

duckdb-analysis: ## Análisis con DuckDB
	python examples/duckdb_analysis.py
