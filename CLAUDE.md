# CLAUDE.md

> Este arquivo é lido automaticamente pelo Claude Code em toda sessão.
> Contém o contexto essencial do projeto. Mantenha enxuto e atualizado.

## Sobre o Projeto

**Nome:** Lideranças Empáticas (LE)
**Instituição:** FECAP
**Descrição:** Sistema de gestão de campanha de arrecadação de alimentos com detecção automática via Visão Computacional (YOLO).

**Domínio:**

- Professores e alunos cadastram-se na plataforma
- Alunos formam grupos (mín. 4 membros)
- Pacotes de alimentos são detectados automaticamente por câmera + YOLO
- Dashboard mostra ranking, totais, distribuição por categoria
- Categorias: Arroz, Feijão, Açúcar, Macarrão, Óleo, Fubá

## Documentos de Referência

Antes de qualquer mudança significativa, consulte:

- `Escopo_LE_Frontend_v2.1.docx` — escopo funcional canônico
- `PLANO_EXECUCAO_LE_v2.md` — fases de desenvolvimento
- `DECISIONS.md` — decisões arquiteturais (ADRs)

## Arquitetura

```
src/
├── backend/        FastAPI + SQLAlchemy async + Alembic
├── frontend/       Next.js 14 App Router + TypeScript
├── cv_detector/    Python + Ultralytics YOLO + cliente HTTP
└── db/             PostgreSQL 16 + init.sql + seeds
```

Comunicação:

- Frontend → Backend: REST + cookies httpOnly (JWT)
- CV Detector → Backend: REST + header `X-CV-Token`
- Backend → Postgres: SQLAlchemy async (asyncpg)

Orquestração: Docker Compose (raiz do repo, fora de `src/`).

## Stack — Não Negociar

| Camada         | Tech                                       | Por quê                           |
| -------------- | ------------------------------------------ | --------------------------------- |
| Backend        | Python 3.12, FastAPI, SQLAlchemy 2.0 async | Performance + tipagem             |
| Frontend       | Next.js 14 App Router, TypeScript strict   | App Router para layouts aninhados |
| UI             | Tailwind + shadcn/ui                       | Customização total via tokens     |
| Forms          | React Hook Form + Zod                      | Validação ponta-a-ponta           |
| Data fetching  | TanStack Query 5                           | Cache + invalidação               |
| State          | Zustand                                    | Leve, sem boilerplate             |
| DB             | PostgreSQL 16                              | UUIDs, ENUMs, CHECK constraints   |
| ORM            | SQLAlchemy 2.0 async + Alembic             | Padrão Python sério               |
| Package Python | uv                                         | Mais rápido que poetry            |
| Package Node   | pnpm                                       | Mais rápido + economiza disco     |

## Convenções de Código

### Geral

- **Idioma:** comentários, commits e nomes de variáveis em **português** quando refletem domínio (ex: `arroz`, `grupo`, `professor`); inglês para código técnico (`async def`, `useEffect`, etc.).
- **Encoding:** UTF-8 em tudo. Português com acentos preservados.
- **EOL:** LF (Unix). `.gitattributes` força isso.

### Python (backend, cv_detector)

- Formatador: **ruff format** (substitui black + isort)
- Linter: **ruff check** com regras: E, F, I, UP, B, C4
- Type hints: **obrigatórias** em assinaturas de função pública
- Docstrings: Google style apenas em serviços e endpoints
- Async: **sempre** que houver I/O (DB, HTTP, arquivos grandes)
- Imports relativos: NÃO usar. Sempre absolutos a partir de `app.`
- Constantes em SCREAMING_SNAKE_CASE no topo do módulo

### TypeScript (frontend)

- Strict mode: **on**
- `any`: proibido (usar `unknown` + narrowing)
- Imports: usar alias `@/*` para `src/*`
- Componentes: PascalCase, um por arquivo
- Hooks: `useNomeAlgo` em `src/hooks/`
- Server Components por padrão; `"use client"` apenas onde necessário

### Banco de Dados

- PKs: **sempre UUID** (`gen_random_uuid()`)
- Timestamps: **TIMESTAMPTZ** sempre, UTC
- Soft delete: **não usar** salvo justificativa em ADR
- Migrations: **uma por PR**, descritivas (`add_evidence_dedup_index`, não `update`)
- ENUMs PostgreSQL para valores fixos (categorias, roles)

### Cores e UI

**Proibido cor hardcoded em JSX/CSS.** Sempre via tokens do `tailwind.config.ts` ou `lib/categoryColors.ts`.

```tsx
// ❌ Errado
<div className="bg-[#16A34A]">

// ✅ Certo
<div className="bg-categoria-feijao">
```

## Regras de Negócio Críticas

Estas regras estão validadas em **frontend (Zod) E backend (Pydantic + DB CHECK)**. Nunca confie em só uma camada.

1. **Domínio de e-mail por papel:**
   - Professor: `@fecap.br`
   - Aluno: `@edu.fecap.br`

2. **Formato do RA:**
   - Professor: 6 dígitos
   - Aluno: 8 dígitos no formato AAMMXXXX
     - AA = 2 últimos dígitos do ano de entrada
     - MM = mês de entrada
     - XXXX = sequencial
     - Exemplo: `24026298` = entrada em fev/2024

3. **Cursos válidos** (lista fechada):
   - Administração
   - Ciências Contábeis
   - Ciências Econômicas

4. **Senha:** mín. 8 chars, pelo menos 1 maiúscula, 1 minúscula, 1 dígito, 1 símbolo.

5. **Grupo:** mínimo 4 integrantes para ser criado.

6. **Privacidade do Feed:**
   - Aluno só vê evidências do **próprio grupo**
   - Filtragem é **no backend** (query por `group_id`), nunca no client

7. **Dedup de evidências:** `dedup_hash` UNIQUE no banco. Detector não envia mesma evidência duas vezes (hash de categoria + bbox quantizado + bucket de 5s).

## Comandos Frequentes

```bash
# Subir stack principal
make up

# Subir stack com cv_detector (Linux only)
make up-cv

# Logs de um serviço
make logs s=backend

# Shell em containers
make shell-backend     # bash
make shell-db          # psql

# Banco
make migrate           # alembic upgrade head
make seed              # carrega dev_seed.sql

# Testes
make test              # pytest no backend

# Limpar tudo (CUIDADO: apaga volumes)
make nuke
```

## O Que NÃO Fazer

- ❌ **Não armazenar tokens em `localStorage`** — sempre cookies httpOnly
- ❌ **Não criar endpoints sem autenticação** salvo `/health` e `/api/v1/auth/*`
- ❌ **Não filtrar privacidade no frontend** — sempre no backend (query level)
- ❌ **Não duplicar regras de negócio sem schema compartilhado** (Zod ↔ Pydantic precisam estar sincronizados)
- ❌ **Não commitar `.env`** — só `.env.example`
- ❌ **Não rodar `cv_detector` em produção cloud** — webcam é local
- ❌ **Não esquecer de revogar refresh tokens** ao trocar senha
- ❌ **Não usar `print()`** — sempre logger estruturado (loguru no Python, console.log apenas em dev)

## Quando Estiver em Dúvida

1. **Sobre escopo funcional:** consulte `Escopo_LE_Frontend_v2.1.docx`
2. **Sobre arquitetura:** consulte `DECISIONS.md`
3. **Sobre fase atual:** consulte `PLANO_EXECUCAO_LE_v2.md`
4. **Sobre comando:** rode `make help`
5. **Sobre API:** rode `docker compose exec backend curl localhost:8000/docs` ou abra `/docs` no navegador (Swagger)

## Plano de Apresentação

Este projeto será apresentado na FECAP usando notebook emprestado.
Estratégia em `APRESENTACAO_FACULDADE.md`. Ao alterar qualquer coisa
crítica para a demo (URL de prod, dados de seed, fluxo de login),
**atualize esse documento também**.

---

_Última atualização: [data]_
