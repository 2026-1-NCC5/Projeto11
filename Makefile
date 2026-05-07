COMPOSE ?= docker compose

.PHONY: up dev down logs ps shell-backend migrate test help

help: ## Lista os targets disponíveis
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up: ## Sobe o backend (Docker) em background
	$(COMPOSE) up -d

dev: ## Sobe o backend (Docker) em foreground com logs
	$(COMPOSE) up

down: ## Para e remove os containers
	$(COMPOSE) down

logs: ## Acompanha os logs do backend
	$(COMPOSE) logs -f backend

ps: ## Lista os containers do projeto
	$(COMPOSE) ps

shell-backend: ## Abre um shell bash dentro do container backend
	$(COMPOSE) exec backend bash

migrate: ## Aplica migrations (alembic upgrade head) dentro do container
	$(COMPOSE) exec backend alembic upgrade head

test: ## Roda a suíte de testes do backend
	$(COMPOSE) exec backend pytest
