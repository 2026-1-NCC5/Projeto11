# Backend — Lideranças Empáticas

API REST em **FastAPI + SQLAlchemy 2.0 async + Alembic**, rodando em **Docker** durante o desenvolvimento. O banco fica no **Supabase Postgres** (não há Postgres local). A autenticação é feita pelo próprio FastAPI (JWT em cookie httpOnly + Bearer fallback), sem usar o Supabase Auth.

---

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) com Compose v2
- [GNU Make](https://www.gnu.org/software/make/) (já vem no Git Bash do Windows)
- Acesso ao projeto Supabase do time (credenciais no `.env`)

> Para desenvolvimento local **sem Docker**, é possível usar `uv` + `uvicorn`, mas o caminho oficial é o Docker.

---

## Variáveis de ambiente

Copie o arquivo `.env.example` (na raiz) para `.env` e preencha:

```env
DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>:5432/postgres
SUPABASE_URL=https://<projeto>.supabase.co
SUPABASE_KEY=<service_role_key>
SUPABASE_FRAMES_BUCKET=frames
JWT_SECRET=<gere com `openssl rand -hex 32`>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
ENV=development
BACKEND_PORT=8000
```

---

## Subir o backend

A partir da **raiz do repositório** (onde está o `Makefile`):

```bash
make up           # sobe o container em background
make logs         # acompanha logs
make ps           # lista containers do projeto
make down         # para e remove
```

Equivalente sem Make:

```bash
docker compose up -d
docker compose logs -f backend
docker compose down
```

O servidor fica em `http://localhost:8000`. Swagger UI em `http://localhost:8000/docs`.

---

## Migrations

Migrations Alembic ficam em `alembic/versions/`.

```bash
make migrate                                          # alembic upgrade head dentro do container
make shell-backend                                    # bash dentro do container
# dentro do container:
alembic revision --autogenerate -m "descricao"        # gerar nova migration
alembic upgrade head                                  # aplicar
alembic downgrade -1                                  # reverter última
alembic stamp head                                    # marcar como aplicado (não roda DDL)
```

> O Supabase pode estar com o schema já aplicado via MCP. Nesse caso, marque o head com `alembic stamp <revision>` antes do primeiro `upgrade`.

---

## Testes

```bash
make test                                             # roda pytest dentro do container
make shell-backend
pytest -k auth                                        # filtros por nome
pytest -x --pdb                                       # parar no 1º erro e abrir debugger
```

---

## Lint e formatação

Usamos **ruff** (formatter + linter).

```bash
make shell-backend
ruff format .
ruff check . --fix
```

---

## Estrutura

```
src/backend/
├── alembic/                # migrations versionadas
├── app/
│   ├── api/v1/             # rotas REST (auth, groups, evidences, users)
│   ├── core/               # config, security, deps
│   ├── db/                 # base SQLAlchemy + session
│   ├── models/             # ORM
│   ├── schemas/            # Pydantic
│   ├── services/           # regras de negócio
│   └── main.py             # entrypoint FastAPI
├── tests/
├── Dockerfile
└── pyproject.toml
```

---

## Apresentação (expor a API)

Para que o frontend hospedado no Vercel consiga falar com o backend local durante a demo:

```bash
ngrok http 8000
```

Atualize `NEXT_PUBLIC_API_URL` no projeto Vercel com a URL pública gerada (pode ser feito via MCP Vercel).
