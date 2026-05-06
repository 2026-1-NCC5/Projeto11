# Plano de Execução — Lideranças Empáticas (v2)

**Stack:** Next.js 14 + FastAPI + PostgreSQL + Python CV Detector (YOLO) + Docker Compose
**Modo de execução:** Claude Code, em fases sequenciais
**Estrutura:** monorepo com pasta `src/` agregadora (template FECAP)

---

## 1. Visão Geral da Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                      DOCKER COMPOSE NETWORK                      │
│                                                                  │
│  ┌───────────────┐    ┌────────────────┐    ┌────────────────┐  │
│  │   frontend    │───▶│    backend     │───▶│   postgres     │  │
│  │  Next.js 14   │    │   FastAPI      │    │  PostgreSQL    │  │
│  │  :3000        │    │   :8000        │    │  :5432         │  │
│  └───────────────┘    └────────▲───────┘    └────────────────┘  │
│                                │                                 │
│                                │ POST /ingest/evidence           │
│                                │                                 │
│                       ┌────────┴────────┐                        │
│                       │   cv_detector   │                        │
│                       │  Python + YOLO  │                        │
│                       │  (best.pt)      │                        │
│                       └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Estrutura do Monorepo

A raiz do repositório segue o template FECAP com `src/` agregando todo o código. **A primeira ação será renomear a pasta `frontend detector` para `cv_detector`** (preservando todo o conteúdo, incluindo `treino_alimentos_final/weights/best.pt`).

```
liderancas-empaticas/                  ← raiz do repositório
├── docker-compose.yml                 ← raiz, fora de src/
├── docker-compose.dev.yml
├── .env.example
├── .env                               ← gitignored
├── .gitignore
├── README.md
├── Makefile
├── docs/                              ← (se já existir, manter)
│
└── src/                               ← código de aplicação
    │
    ├── backend/                       ← FastAPI (a ser criado)
    │   ├── Dockerfile
    │   ├── pyproject.toml
    │   ├── alembic.ini
    │   ├── alembic/
    │   └── app/
    │       ├── main.py
    │       ├── core/
    │       ├── api/v1/
    │       ├── models/
    │       ├── schemas/
    │       ├── services/
    │       └── db/
    │
    ├── frontend/                      ← Next.js 14 (a ser criado)
    │   ├── Dockerfile
    │   ├── package.json
    │   ├── next.config.mjs
    │   └── src/
    │       ├── app/
    │       ├── components/
    │       ├── hooks/
    │       ├── lib/
    │       └── types/
    │
    ├── cv_detector/                   ← renomeado de "frontend detector"
    │   ├── Dockerfile                 ← a criar
    │   ├── requirements.txt
    │   ├── detect.py                  ← refatorar do existente
    │   ├── api_client.py              ← novo: cliente HTTP
    │   └── treino_alimentos_final/
    │       ├── weights/
    │       │   └── best.pt            ← já existente
    │       └── ...                    ← preservar todo o conteúdo
    │
    └── db/                            ← novo: schema + seeds
        ├── Dockerfile
        ├── init.sql
        └── seeds/
            └── dev_seed.sql
```

**Pontos de atenção sobre os paths:**

| Contexto                           | Path                                               |
| ---------------------------------- | -------------------------------------------------- |
| Docker build do backend            | `./src/backend`                                    |
| Docker build do frontend           | `./src/frontend`                                   |
| Docker build do cv_detector        | `./src/cv_detector`                                |
| Docker build do db                 | `./src/db`                                         |
| Volume do modelo YOLO no host      | `./src/cv_detector/treino_alimentos_final/weights` |
| Volume do modelo YOLO no container | `/app/treino_alimentos_final/weights:ro`           |

---

## 3. Pré-requisitos

- Docker Desktop instalado (Compose v2)
- Claude Code autenticado
- Repositório Git inicializado
- Logos `liderancas_empaticas.png` e `fecap.png` para `src/frontend/public/`
- `best.pt` já presente em `src/cv_detector/treino_alimentos_final/weights/`
- `Escopo_LE_Frontend_v2.1.docx` na raiz (referência)

---

## 4. Fases de Execução

> **Regra de ouro:** uma fase = uma sessão do Claude Code. Valide os critérios de aceite antes de avançar.

---

### FASE 0 — Setup do Monorepo

**Objetivo:** Renomear `frontend detector` para `cv_detector`, criar `src/db/`, criar arquivos raiz (compose, Makefile, env, gitignore, README).

**Prompt para Claude Code:**

```
Estamos preparando o monorepo do projeto "Lideranças Empáticas" no
diretório atual. A estrutura atual segue o template FECAP com pasta src/.

Estado atual de src/:
- src/backend/   (vazio ou parcial — vamos popular nas próximas fases)
- src/frontend/  (vazio ou parcial)
- src/frontend detector/   ← renomear PARA src/cv_detector/
  Conteúdo a preservar:
    - todos os scripts Python existentes
    - pasta treino_alimentos_final/ inteira (incluindo weights/best.pt)

Tarefas:

1. Renomeie a pasta "src/frontend detector" para "src/cv_detector"
   preservando todo o conteúdo. Use git mv se for um repo Git.

2. Crie src/db/ vazia.

3. Crie na RAIZ do repositório (não dentro de src/):

   .gitignore  cobrindo:
     - Python: __pycache__, *.pyc, .venv, .pytest_cache, .mypy_cache, .ruff_cache
     - Node: node_modules, .next, .turbo, npm-debug.log
     - Docker: pgdata/, data/, .docker/
     - IDE: .idea/, .vscode/, *.swp
     - Env: .env, .env.local, .env.*.local
     - YOLO: src/cv_detector/treino_alimentos_final/weights/*.pt
       (use Git LFS ou DVC para versionar pesos — adicionar nota no README)
     - Frames: data/frames/

   .env.example  com TODAS as variáveis:
     POSTGRES_USER=le_user
     POSTGRES_PASSWORD=changeme
     POSTGRES_DB=liderancas_empaticas
     POSTGRES_PORT=5432

     BACKEND_PORT=8000
     JWT_SECRET=changeme_use_openssl_rand
     JWT_ALGORITHM=HS256
     ACCESS_TOKEN_EXPIRE_MINUTES=30
     REFRESH_TOKEN_EXPIRE_DAYS=7

     FRONTEND_PORT=3000
     NEXT_PUBLIC_API_URL=http://localhost:8000

     CV_API_TOKEN=changeme_shared_secret
     CV_BACKEND_URL=http://backend:8000
     CV_GROUP_ID=
     CV_CONFIDENCE_THRESHOLD=0.6
     CV_CAMERA_INDEX=0

   Makefile  com targets:
     up, up-cv, dev, down, nuke, logs, ps,
     shell-backend, shell-db, migrate, seed, test
     help (target padrão listando todos)

   README.md  raiz explicando:
     - Visão geral da arquitetura
     - Pré-requisitos (Docker, .env)
     - Setup: clonar, copiar .env.example para .env, make up, make migrate, make seed
     - Estrutura de pastas (com src/)
     - Como rodar o cv_detector em Linux vs Mac/Windows
     - Como atualizar o modelo YOLO

NÃO crie ainda Dockerfiles nem código de aplicação. Só base + estrutura.
Confirme ao final que a renomeação foi bem sucedida e o best.pt continua em
src/cv_detector/treino_alimentos_final/weights/best.pt.
```

**Critério de aceite:**

- `ls src/` mostra `backend/`, `frontend/`, `cv_detector/`, `db/`
- `ls src/cv_detector/treino_alimentos_final/weights/` mostra `best.pt`
- `cp .env.example .env` funciona
- `make help` lista os comandos

---

### FASE 1 — Banco de Dados (Schema)

**Objetivo:** Schema SQL completo, seeds, Dockerfile do Postgres em `src/db/`.

**Prompt para Claude Code:**

```
Crie o schema SQL completo em src/db/init.sql para PostgreSQL 16.
Use o documento Escopo_LE_Frontend_v2.1.docx como referência canônica
das regras de negócio.

Requisitos:

1. UUIDs como PK em todas as tabelas (extensão pgcrypto)

2. ENUMs PostgreSQL:
   - food_category: arroz, feijao, acucar, macarrao, oleo, fuba
   - user_role: professor, aluno
   - period_type: matutino, noturno

3. Tabela users:
   - id UUID PK
   - email TEXT UNIQUE NOT NULL
   - role user_role NOT NULL
   - ra TEXT NOT NULL UNIQUE
   - full_name TEXT NOT NULL
   - course TEXT NOT NULL CHECK (course IN ('Administração','Ciências Contábeis','Ciências Econômicas'))
   - semester SMALLINT CHECK (semester BETWEEN 1 AND 8)
   - period period_type NOT NULL
   - password_hash TEXT NOT NULL
   - created_at, updated_at TIMESTAMPTZ
   - CHECK constraint condicional (use CASE WHEN):
     * Se role='professor': email LIKE '%@fecap.br' E ra ~ '^\d{6}$'
     * Se role='aluno': email LIKE '%@edu.fecap.br' E ra ~ '^\d{8}$'

4. Tabela groups:
   - id UUID PK, name TEXT UNIQUE NOT NULL
   - created_by UUID REFERENCES users(id)
   - created_at TIMESTAMPTZ

5. Tabela group_members (PK composta):
   - group_id UUID REFERENCES groups(id) ON DELETE CASCADE
   - user_id UUID REFERENCES users(id) ON DELETE CASCADE
   - joined_at TIMESTAMPTZ
   - PRIMARY KEY (group_id, user_id)

6. Tabela detection_sessions:
   - id UUID PK
   - group_id UUID REFERENCES groups(id)
   - started_by UUID REFERENCES users(id)
   - started_at, ended_at TIMESTAMPTZ

7. Tabela evidences:
   - id UUID PK
   - session_id UUID REFERENCES detection_sessions(id)
   - group_id UUID REFERENCES groups(id) NOT NULL  (denormalizado para query)
   - category food_category NOT NULL
   - frame_url TEXT
   - confidence NUMERIC(4,3)
   - detected_at TIMESTAMPTZ NOT NULL
   - dedup_hash TEXT UNIQUE NOT NULL
   - created_at TIMESTAMPTZ DEFAULT NOW()

8. Tabela refresh_tokens:
   - id UUID PK
   - user_id UUID REFERENCES users(id) ON DELETE CASCADE
   - token_hash TEXT NOT NULL
   - expires_at TIMESTAMPTZ NOT NULL
   - revoked BOOLEAN DEFAULT false

9. Indexes:
   - evidences (group_id, detected_at DESC)
   - evidences (category)
   - evidences (session_id)
   - users (email)
   - groups (name)
   - refresh_tokens (user_id, revoked)

10. Trigger updated_at em users (função set_updated_at()).

Crie src/db/seeds/dev_seed.sql com:
- 2 professores (com ra de 6 dígitos, email @fecap.br)
- 12 alunos (com ra AAMMXXXX, email @edu.fecap.br) — varie cursos e períodos
- 3 grupos com 4 membros cada
- 2 detection_sessions (uma encerrada, uma ativa)
- 30 evidences distribuídas nas 6 categorias e nos 3 grupos
  (com dedup_hash únicos)
- Use senhas hasheadas com bcrypt — gere offline e cole o hash de
  "Senha@123" para todos (simplifica testes)

Crie src/db/Dockerfile baseado em postgres:16-alpine:
- Copia init.sql para /docker-entrypoint-initdb.d/01_init.sql
- Copia seeds/ para /seeds/ (não autoexecutar — make seed dispara)
- HEALTHCHECK pg_isready

Valide o SQL rodando:
docker build -t le-db ./src/db
docker run --name le-db-test -e POSTGRES_PASSWORD=test -d le-db
docker exec le-db-test psql -U postgres -d postgres -c "\dt"
```

**Critério de aceite:**

- Build da imagem passa sem warnings
- Container sobe e expõe todas as tabelas
- Inserção de aluno com email `@fecap.br` falha no CHECK
- Inserção de professor com RA de 8 dígitos falha no CHECK

---

### FASE 2 — Backend Core (Auth)

**Objetivo:** FastAPI + SQLAlchemy + Alembic + auth JWT em cookies httpOnly.

**Prompt para Claude Code:**

```
Crie o backend FastAPI em src/backend/ conforme o Escopo_LE_Frontend_v2.1.docx.

Stack:
- Python 3.12, FastAPI, SQLAlchemy 2.0 async, asyncpg, Alembic
- Pydantic v2 + pydantic-settings
- python-jose[cryptography] para JWT
- passlib[bcrypt] para senhas
- uv como gerenciador de dependências (mais rápido que poetry)

Estrutura:
src/backend/
  pyproject.toml
  alembic.ini
  alembic/
    env.py (configurar para async)
    versions/
  app/
    main.py
    core/
      config.py        (Settings via pydantic-settings)
      security.py      (hash, verify, JWT helpers)
      deps.py          (get_db, get_current_user, require_role)
    db/
      base.py          (Base = DeclarativeBase)
      session.py       (async_engine, async_session_maker)
    models/
      __init__.py
      user.py
      group.py
      group_member.py
      session.py
      evidence.py
      refresh_token.py
    schemas/
      __init__.py
      auth.py
      user.py
      group.py
      evidence.py
    api/
      v1/
        __init__.py
        router.py
        auth.py
    services/
      auth_service.py
  Dockerfile
  .dockerignore

Endpoints da Fase 2 (auth.py):
- POST /api/v1/auth/register
- POST /api/v1/auth/login
- POST /api/v1/auth/refresh
- POST /api/v1/auth/logout
- GET  /api/v1/auth/me
- GET  /health

Validação no schema auth.RegisterIn:
Use pydantic discriminated union por 'role':
- RegisterProfessorIn: email regex r'^[\w.+-]+@fecap\.br$'  + ra regex r'^\d{6}$'
- RegisterAlunoIn:    email regex r'^[\w.+-]+@edu\.fecap\.br$' + ra regex r'^\d{8}$'
Senha: validator que checa min 8 chars, upper, lower, digit, symbol.

JWT/Cookies:
- access_token em cookie httpOnly + Secure (em prod) + SameSite=Lax
- refresh_token em cookie httpOnly separado, path=/api/v1/auth/refresh
- /refresh rotaciona o refresh (revoga o antigo, emite novo)
- /logout revoga o refresh atual e limpa cookies

Dockerfile multi-stage:
FROM python:3.12-slim AS builder
... instala uv, gera venv com deps
FROM python:3.12-slim AS runtime
... copia venv, código, usuário não-root, expõe 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

Alembic:
- alembic init com async template
- gerar primeira revision com autogenerate (deve casar com init.sql)
- documentar no README do backend: alembic upgrade head

CORS:
allow_origins=[settings.FRONTEND_URL]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

**Critério de aceite:**

- `docker build -t le-backend ./src/backend` passa
- Subir backend + db, registrar professor e aluno com regras corretas
- Tentar registrar aluno com `@fecap.br` retorna 422 com mensagem clara
- Login → cookies setados → /me retorna o usuário
- /refresh rotaciona corretamente

---

### FASE 3 — Backend Domínio

**Objetivo:** Dashboard, evidências, grupos, exportação, ingestão da CV.

**Prompt para Claude Code:**

```
Adicione os endpoints de domínio ao backend, em src/backend/app/api/v1/.
Use Escopo_LE_Frontend_v2.1.docx como referência.

Routers a criar:

1. dashboard.py — todas exigem autenticação:
   - GET /dashboard/summary?group_id=&start=&end=
     Retorna { total_kg, leading_category: {name, kg}, active_groups }
   - GET /dashboard/by-category?group_id=&start=&end=
     Retorna { arroz: kg, feijao: kg, ... }
   - GET /dashboard/timeline?group_id=&start=&end=&granularity=day|week
     Retorna [{ date, arroz, feijao, ... }]
   - GET /dashboard/ranking?limit=10
     Retorna [{ position, group_id, name, total_kg }]
   - GET /dashboard/by-group-stacked?start=&end=
     Retorna [{ group_id, name, arroz, feijao, ... }]

   IMPORTANTE: como evidences armazena unidades discretas (1 pacote),
   use uma tabela ou config para "peso médio por categoria" e converta
   para kg na agregação. Sugira no código onde plugar essa configuração.

2. evidences.py:
   - GET /evidences?categories=&page=&size=&start=&end=
     Aluno: filtro AUTOMÁTICO por seu group_id (no service, não no client)
     Professor: pode ver tudo, opcional ?group_id=
     Paginação cursor-based ou offset (offset é mais simples)
   - GET /evidences/export?format=csv|pdf&...mesmos filtros
     CSV: usar Python csv padrão (não pandas para evitar peso)
     PDF: reportlab com layout simples (logo no topo, tabela, totais)
     StreamingResponse com Content-Disposition: attachment

3. groups.py:
   - GET /groups
     Aluno: retorna apenas o próprio (ou 404 se não tem grupo)
     Professor: retorna todos com contagem de membros
   - GET /groups/{id}
     Inclui lista de members (id, name, email, ra)
     Aluno só pode acessar o próprio grupo
   - POST /groups (require_role=professor)
     Body: { name, member_ids: [UUID] (mín. 4) }
     Validação 4+ no service, 422 se menor
   - PUT /groups/{id}/members (require_role=professor)
     Body: { member_ids } — substitui completamente
     Mantém validação 4+
   - DELETE /groups/{id} (require_role=professor)

4. ingest.py — endpoint para o cv_detector:
   - POST /ingest/evidence
     Auth: header X-CV-Token === settings.CV_API_TOKEN (constant time compare)
     Body: {
       session_id: UUID (opcional — cria sessão automaticamente se ausente),
       group_id: UUID,
       category: food_category,
       frame_base64: str,
       confidence: float,
       detected_at: datetime,
       dedup_hash: str
     }
     Salvar frame:
     - decodificar base64
     - escrever em /app/data/frames/{yyyy}/{mm}/{dd}/{evidence_id}.jpg
     - frame_url = relativo, servido por endpoint estático /frames/{path}
     ON CONFLICT (dedup_hash) DO NOTHING — retorna 200 com {created: false}

   - POST /ingest/session/start
     Body: { group_id }
     Retorna { session_id }

   - POST /ingest/session/end
     Body: { session_id }

   - GET /frames/{path:path}  (servir frames com auth do usuário)

Adicionar em main.py:
app.mount("/frames", StaticFiles(directory="data/frames"), name="frames")
(ou criar endpoint custom com auth)

Testes pytest mínimos em src/backend/tests/:
- conftest.py com fixtures (db de teste, client httpx, users)
- test_auth.py — register/login/me/refresh
- test_evidences.py — aluno só vê seu grupo
- test_groups.py — regra dos 4 membros
- test_ingest.py — dedup funciona

Atualizar pyproject.toml com pytest, pytest-asyncio, httpx, reportlab.
```

**Critério de aceite:**

- Aluno autenticado em /evidences só recebe evidências do próprio grupo
- Professor cria grupo com 3 membros → 422; com 4 → 201
- POST /ingest/evidence com X-CV-Token correto cria; mesma hash 2x não duplica
- Export CSV abre no Excel; PDF é legível
- `pytest src/backend/tests` passa

---

### FASE 4 — CV Detector (refator)

**Objetivo:** Adaptar os scripts existentes em `src/cv_detector/` para enviar dados ao backend, sem alterar a lógica YOLO.

**Prompt para Claude Code:**

```
Em src/cv_detector/ existem scripts Python de detecção YOLO já funcionais
e o modelo treinado em treino_alimentos_final/weights/best.pt.

Tarefa: refatorar APENAS o que for necessário para integrar com o backend,
preservando a lógica de detecção existente.

Etapas:

1. Listar e analisar os scripts atuais. Identifique:
   - O loop principal de detecção
   - Como as classes do modelo mapeiam para alimentos
   - Onde a inferência YOLO é executada

2. Criar src/cv_detector/api_client.py:
   class APIClient:
     - __init__(base_url, token)
     - start_session(group_id) -> session_id
     - end_session(session_id)
     - send_evidence(session_id, group_id, category, frame_jpeg_bytes, confidence, detected_at, dedup_hash) -> bool
     Use httpx + tenacity para retry exponencial (3 tentativas)
     Headers: X-CV-Token = self.token
     Frame: encodar para base64 antes de enviar

3. Criar src/cv_detector/dedup.py:
   def compute_dedup_hash(category: str, bbox: tuple, timestamp: datetime) -> str:
     - Quantizar bbox para grid de 10px (round(x/10)*10)
     - Quantizar timestamp para bucket de 5 segundos
     - sha256(f"{category}|{qx}|{qy}|{qw}|{qh}|{bucket}").hexdigest()
   Isso evita o mesmo objeto parado gerar múltiplas evidências.

4. Refatorar o loop principal (criar detect.py se ainda não existir como entrypoint):
   - Ler config de env: BACKEND_URL, CV_API_TOKEN, GROUP_ID,
     CONFIDENCE_THRESHOLD, MODEL_PATH, CAMERA_INDEX
   - MODEL_PATH default: /app/treino_alimentos_final/weights/best.pt
   - Inicializar YOLO(MODEL_PATH)
   - Inicializar APIClient e fazer start_session(GROUP_ID)
   - Loop:
     a) Capturar frame da webcam (cv2.VideoCapture)
     b) Inferência: model.predict(frame, conf=CONFIDENCE_THRESHOLD)
     c) Para cada detecção:
        - Mapear class_id → category enum
        - Calcular dedup_hash
        - Enviar via api_client.send_evidence
     d) Mostrar preview (cv2.imshow) — opcional, controlado por env DISPLAY
     e) Sair com 'q'
   - Tratamento de SIGTERM: end_session antes de encerrar
   - Logger estruturado (loguru)

5. Mapeamento class_id → category:
   - Examinar o data.yaml ou names do modelo treinado
   - Criar dict explícito em detect.py:
     CATEGORY_MAP = {
       0: "arroz", 1: "feijao", 2: "acucar",
       3: "macarrao", 4: "oleo", 5: "fuba"
     }
     (ajustar conforme as classes reais do best.pt)

6. requirements.txt:
   ultralytics==8.x
   opencv-python-headless==4.x  (headless para Docker; troque para opencv-python se rodar local com display)
   httpx, tenacity, loguru, python-dotenv

7. src/cv_detector/Dockerfile:
   FROM ultralytics/ultralytics:latest-cpu
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ENV PYTHONUNBUFFERED=1
   CMD ["python", "detect.py"]

   IMPORTANTE: o Dockerfile NÃO copia o best.pt para a imagem.
   O modelo será montado via volume no docker-compose.

8. README.md em src/cv_detector/ explicando:
   - Como rodar fora do Docker (recomendado em Mac/Windows):
     cd src/cv_detector
     pip install -r requirements.txt
     export BACKEND_URL=http://localhost:8000 CV_API_TOKEN=... GROUP_ID=...
     python detect.py
   - Como rodar dentro do Docker (Linux com /dev/video0):
     docker compose --profile cv up cv_detector

NÃO altere a lógica de inferência YOLO — apenas envolva-a com o api_client
e o loop estruturado. Se houver scripts de treino, deixar como estão.
```

**Critério de aceite:**

- `python src/cv_detector/detect.py` rodando local conecta ao backend e envia evidências
- Dedup funciona: objeto parado por 30s gera ~6 evidências (não 600)
- Frame chega no backend e é salvo em `data/frames/`
- Backend retorna a evidência via GET /evidences

---

### FASE 5 — Frontend Setup

**Objetivo:** Bootstrap Next.js 14 com toda infra, sem telas funcionais ainda.

**Prompt para Claude Code:**

```
Crie o frontend Next.js 14 em src/frontend/ conforme o
Escopo_LE_Frontend_v2.1.docx.

Stack:
- Next.js 14 App Router, TypeScript strict, src/ structure
- pnpm como package manager
- Tailwind CSS 3
- shadcn/ui
- TanStack Query 5
- Zustand 4
- React Hook Form 7 + Zod 3
- Axios (com interceptor para refresh)
- Recharts 2
- date-fns 3 (locale pt-BR)
- lucide-react
- framer-motion

Tarefas:

1. Bootstrap:
   cd src/frontend
   pnpm create next-app . --typescript --tailwind --app --src-dir --import-alias "@/*"

2. Inicializar shadcn/ui:
   pnpm dlx shadcn@latest init
   Escolher cores customizadas (vamos sobrescrever).
   Adicionar componentes base: button, input, label, card, tabs, select,
   dropdown-menu, dialog, sheet, table, badge, toast, skeleton, separator

3. Sobrescrever tailwind.config.ts e globals.css com TODOS os tokens:
   theme.extend.colors:
     primary: { DEFAULT: "#2A7A4B", foreground: "#FFFFFF" }
     accent: "#3DA066"
     fecap: "#1A6B2A"
     surface: "#F4F6F4"
     muted: "#6B7B6E"
     border: "#D6E4DA"
     categoria: {
       arroz: "#D97706", feijao: "#16A34A", acucar: "#0284C7",
       macarrao: "#EA580C", oleo: "#CA8A04", fuba: "#92400E"
     }
   theme.extend.fontFamily:
     display: ["var(--font-blacker)", "Inter", "system-ui"]
     sans: ["var(--font-inter)", "system-ui"]

4. Criar src/lib/categoryColors.ts com tipo Category e mapas:
   - CATEGORY_COLORS: Record<Category, string> (hex)
   - CATEGORY_LABELS: Record<Category, string> (label pt-BR)
   - CATEGORIES: Category[] (ordem canônica)

5. Criar src/lib/api.ts (Axios):
   - baseURL = process.env.NEXT_PUBLIC_API_URL
   - withCredentials: true
   - Interceptor de response: se 401 → tenta /auth/refresh → reexecuta
     se ainda 401 → redireciona /login

6. Criar src/store/authStore.ts (Zustand):
   - state: user, role, isLoading, isAuthenticated
   - actions: setUser, clearUser, hydrate (chama /auth/me)
   - persist? não — cookie httpOnly é a fonte de verdade

7. Criar src/lib/validators.ts com Zod:
   - passwordSchema (8 chars, regex de classes)
   - loginSchema
   - registerProfessorSchema (email @fecap.br + ra 6 dígitos)
   - registerAlunoSchema (email @edu.fecap.br + ra 8 dígitos AAMMXXXX)
   - registerSchema = z.discriminatedUnion("role", [...])
   - decodeStudentRA(ra): { year, month, monthName }

8. Criar src/components/providers/:
   - QueryProvider (TanStack)
   - ThemeProvider (light only por enquanto)

9. src/app/layout.tsx integrando providers + fonts (next/font/local):
   - Inter via next/font/google
   - Blacker Sans Pro via next/font/local apontando para public/fonts/
     (criar pasta public/fonts/ com README explicando que arquivos
     .woff2 devem ser colocados ali — gitignored)

10. src/app/page.tsx placeholder:
    Apenas redireciona para /login (ou /dashboard se autenticado).

11. src/frontend/Dockerfile multi-stage:
    Stage 1 (deps): node:20-alpine, pnpm install --frozen-lockfile
    Stage 2 (builder): build com next build (output: standalone)
    Stage 3 (runner): node:20-alpine, copia .next/standalone, .next/static, public
    USER node, EXPOSE 3000, CMD ["node", "server.js"]
    HEALTHCHECK curl /

12. .dockerignore com node_modules, .next, .git

13. next.config.mjs:
    output: "standalone"
    images.remotePatterns para o backend (frame thumbnails)

NÃO criar páginas de login/dashboard ainda. Apenas a infraestrutura.
Validação: pnpm dev sobe e mostra placeholder. docker build passa.
```

**Critério de aceite:**

- `cd src/frontend && pnpm dev` sobe e mostra placeholder
- `docker build -t le-frontend ./src/frontend` passa
- Adicionar `<div className="bg-categoria-feijao">test</div>` mostra cor verde correta

---

### FASE 6 — Frontend Auth

**Objetivo:** Login, cadastro com validação condicional, AuthGuard, layout protegido.

**Prompt para Claude Code:**

```
Implemente o módulo de autenticação do frontend conforme seções 4
do Escopo_LE_Frontend_v2.1.docx.

Estrutura:
src/frontend/src/app/
  (auth)/
    layout.tsx          # layout sem sidebar
    login/
      page.tsx
  (app)/
    layout.tsx          # layout COM sidebar + AuthGuard
    dashboard/page.tsx  # placeholder por enquanto

Tela /login:
- Layout dois painéis (40/60), responsivo (stack em mobile)
- Painel esquerdo: bg-primary, logo Lideranças Empáticas centralizada
  (Image do next/image, src="/logo_le.png"), tagline,
  logo FECAP no rodapé
- Painel direito: tabs "Entrar" | "Cadastrar"

Tab Entrar:
- React Hook Form + zodResolver(loginSchema)
- Input email com placeholder
- Input password com toggle show/hide
- Submit chama mutation login(POST /auth/login)
- Em sucesso: hidrata authStore via /auth/me, redireciona /dashboard
- Em erro 401: toast "Credenciais inválidas"

Tab Cadastrar:
- Form com TODOS os campos do escopo
- Resolver dinâmico baseado no role selecionado:
  const role = watch("role")
  const schema = role === "professor" ? registerProfessorSchema : registerAlunoSchema
- RA helper text DINÂMICO:
  Se role=aluno e ra.length >= 4:
    const { year, monthName } = decodeStudentRA(ra)
    "Entrada em {monthName} de 20{year}"
  Se role=aluno: "Formato AAMMXXXX (ano + mês + sequencial)"
  Se role=professor: "RA institucional de 6 dígitos"
- PasswordStrengthMeter (4 segmentos coloridos com framer-motion)
- Confirmar senha com check verde animado quando coincide
- Submit chama mutation register
- Em sucesso: faz login automático e redireciona

Componentes a criar em src/frontend/src/components/auth/:
- LoginForm.tsx
- RegisterForm.tsx
- PasswordStrengthMeter.tsx
- StudentRAInput.tsx (com decodificador inline)
- RoleToggle.tsx (radio entre Professor/Aluno com icons)

AuthGuard em src/app/(app)/layout.tsx:
- useEffect: chama authStore.hydrate() no mount
- Se !isAuthenticated && !isLoading: redirect("/login")
- Renderiza Sidebar + main com children

Sidebar (src/components/layout/Sidebar.tsx):
- Width 220px, fixed, full height
- Header: logo Lideranças Empáticas + texto "Lideranças Empáticas"
- Nav items (4): Dashboard, Evidências, Grupos, Perfil
  - Cada item: lucide icon + label
  - Active state: bg-primary/10 + barra esquerda 4px primary
- Footer: avatar com iniciais + nome + role + logo FECAP
- Mobile (<768px): drawer com hamburguer no Topbar

Topbar (mobile only):
- Hamburguer + título da rota atual + avatar à direita

Mutations React Query:
- useLoginMutation
- useRegisterMutation
- useLogoutMutation (POST /auth/logout, limpa store, redireciona)

Hook useAuth:
- Encapsula authStore + queries de /me

Acessibilidade:
- aria-labels em todos inputs
- Focus visible com ring primary
- Mensagens de erro com role="alert"
- Tab order lógica
```

**Critério de aceite:**

- Cadastrar professor funciona; aluno também
- Aluno com email @fecap.br: erro inline imediato (não só no submit)
- Digitar `2402` no RA do aluno: helper exibe "Entrada em Fevereiro de 2024"
- Digitar `9912` no RA do aluno: helper exibe "Entrada em Dezembro de 1999" (ou ajustar lógica de cutoff)
- Login redireciona /dashboard
- F5 mantém autenticado
- Logout funciona e redireciona
- Tentar acessar /dashboard sem auth redireciona /login

---

### FASE 7 — Frontend Dashboard

**Objetivo:** Dashboard completo conforme seção 5 do escopo.

**Prompt para Claude Code:**

```
Implemente /dashboard conforme seção 5 do Escopo_LE_Frontend_v2.1.docx.

Componentes em src/frontend/src/components/dashboard/:

1. FilterBar.tsx:
   - DateRangePicker (shadcn date-picker em modo range)
   - GroupSelect (dropdown buscável)
   - Botão Aplicar
   - Estado em URL searchParams (sincronizado com TanStack Query keys)

2. PrimaryStatCards.tsx (linha de 3 cards equal-width):
   a) StatCard "Total Geral":
      - Borda esquerda 6px primary
      - Número grande font-display, animação de contagem framer-motion
      - Sufixo "kg"
      - Subtitle "Arrecadação total da campanha"
      - Ícone Scale (lucide)
   b) StatCard "Categoria Líder":
      - Borda esquerda 6px na cor da categoria líder
      - Texto "Arroz — 612 kg" formato (label + valor)
      - Ícone Trophy
   c) StatCard "Grupos Ativos":
      - Borda neutra
      - Número de grupos
      - Ícone Users

3. CategorySection.tsx:
   - Header: "Distribuição por Categoria"
   - CategoryFilterPills (single-select):
     [Todas] [Arroz] [Feijão] [Açúcar] [Macarrão] [Óleo] [Fubá]
     Pill ativa: bg na cor da categoria + texto branco
     Pill inativa: outlined com border-categoria-X + texto categoria-X
   - Body condicional:
     * Se "Todas":
       <StackedBarHorizontal data={byCategory} />
       Legenda inline com 6 itens (• cor + label + valor kg)
     * Se categoria selecionada:
       Layout 2 colunas:
       - Esquerda: número grande na cor da categoria + "% do total" + ranking
       - Direita: mini sparkline de área (Recharts AreaChart simples)

4. TimelineChart.tsx (60% width):
   - Recharts AreaChart com 6 áreas, uma por categoria
   - useState hoveredCategory: ao hover em uma linha, todas exceto ela
     vão para opacity 0.1 (use Recharts onMouseEnter por linha)
   - Legenda clicável para toggle visibility
   - Eixo X: datas formatadas pt-BR (date-fns format)
   - Eixo Y: kg

5. RankingCard.tsx (40% width, lado direito do timeline):
   - Lista numerada 1-10
   - Top 3 com emoji medalha (🥇🥈🥉)
   - Linha: posição + nome do grupo + (ml-auto) total kg
   - useQuery com refetchInterval: 30000

6. ByGroupStackedChart.tsx (full-width, abaixo):
   - Recharts BarChart com 6 series stacked
   - Barras na vertical, eixo X = nomes dos grupos
   - Click numa barra: setSelectedGroupId → abre Sheet (drawer) com
     drill-down detalhado do grupo (lista de evidências por categoria)

Hooks em src/frontend/src/hooks/:
- useDashboardSummary(filters)
- useDashboardByCategory(filters, category?)
- useDashboardTimeline(filters)
- useDashboardRanking()
- useDashboardByGroupStacked(filters)

Página /app/dashboard/page.tsx:
- Title "Dashboard" + breadcrumb
- FilterBar
- PrimaryStatCards
- CategorySection
- Grid 2 colunas: TimelineChart (col-span-3) + RankingCard (col-span-2) (em md+)
- ByGroupStackedChart (full width)

Skeleton loaders durante fetch (componentes Skeleton da shadcn).
Empty states com ilustração simples + texto "Nenhuma evidência ainda".
Erros via toast.
Cores SEMPRE de lib/categoryColors.ts — proibido hardcoded em JSX.
```

**Critério de aceite:**

- Cards primários animam contagem ao montar
- Selecionar pill "Feijão" troca para o estado expandido
- Hover em uma linha do timeline isola visualmente
- Click em barra abre drawer
- Polling do ranking visível em DevTools Network

---

### FASE 8 — Frontend Evidências, Grupos, Perfil

**Prompt para Claude Code:**

```
Implemente as telas restantes conforme seções 6, 7 e 8 do
Escopo_LE_Frontend_v2.1.docx.

A. /app/evidencias/page.tsx:
   - PageHeader com título + 2 botões export
     ExportButton dispara GET /evidences/export?format=csv com filtros atuais
     (use fetch + window.URL.createObjectURL para download)
   - PrivacyBanner: bg-primary/10, border-l-primary, ícone Lock,
     texto sobre privacidade
   - FilterBar: date range + CategoryFilterPills (multi-select)
   - DataTable (shadcn):
     Colunas: Frame (Image 64x64 rounded), Categoria (Badge colorido),
              Data, Hora, Validado por
     Linhas alternadas (bg-surface em odds)
     Click no thumbnail abre Dialog com imagem em tamanho real
   - Paginação no rodapé (shadcn pagination)

B. /app/grupos/page.tsx:
   const role = useAuth().role
   role === "aluno" ? <MyGroupView /> : <AdminGroupView />

   MyGroupView:
   - GroupHeroCard: nome bold lg, data criação muted, 2 badges
     (total kg + posição ranking)
   - Grid 2-col de MembroCard:
     Avatar circular com iniciais (bg-primary/20),
     Nome bold, Email muted, RA muted small
     Card do user logado: border-2 border-primary + badge "Você"

   AdminGroupView:
   - Header com botão "+ Criar Novo Grupo" (abre modal)
   - Stats row (3 cards): Total grupos, Total alunos, Grupos válidos (4+ membros)
   - Grid de GrupoCard:
     Nome, AvatarGroup (max 4 + "+N"), Total kg,
     StatusBadge (Ativo verde / Incompleto amber),
     Actions: Ver detalhes, Gerenciar membros

   CriarGrupoModal (Dialog shadcn):
   - Input nome
   - Combobox MultiselectAlunos (busca por nome/email/RA)
   - Lista de selecionados com X para remover
   - Progress bar: (selecionados.length / 4) * 100, max 100
     Texto: "X de 4 integrantes mínimos"
     Cor: amber se <4, verde se >=4
   - Botão Criar disabled se name vazio ou selecionados < 4

C. /app/perfil/page.tsx:
   Layout: max-w-2xl mx-auto

   ProfileCard:
     Avatar grande (h-20 w-20) com iniciais
     Nome (text-2xl bold)
     Badge cargo (Aluno verde ou Professor primary)
     Grid 2-col de pares label/valor: Email, RA, Curso, Semestre, Período
     "Membro desde [mês/ano]"

   ChangePasswordCard (Collapsible):
     Trigger: "Alterar Senha" + chevron
     Content:
       Input senha atual
       Input nova senha com PasswordStrengthMeter
       Input confirmar
       Botão "Salvar nova senha"
       Mutation PUT /users/me/password (criar endpoint no backend Fase 3
       se ainda não existe — adicionar agora)

   DangerZoneCard:
     border-destructive
     "Sair da conta"
     "Você será desconectado desta sessão"
     Botão Logout (variant="destructive" outline)

Acessibilidade:
- Modal com focus trap (shadcn Dialog já cobre)
- Multiselect navegável por teclado
- aria-live em toasts e mudanças de progress bar
```

**Critério de aceite:**

- Aluno em /grupos vê só MyGroupView
- Professor consegue criar grupo (4+ membros) — botão habilita só ao chegar
- Trocar senha end-to-end
- Logout limpa sessão e redireciona

---

### FASE 9 — Docker Compose

**Prompt para Claude Code:**

```
Crie docker-compose.yml e docker-compose.dev.yml na RAIZ do repo
(fora de src/).

docker-compose.yml:

services:
  postgres:
    build: ./src/db
    container_name: le_postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./src/db/seeds:/seeds:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks: [le_net]
    restart: unless-stopped

  backend:
    build: ./src/backend
    container_name: le_backend
    depends_on:
      postgres: { condition: service_healthy }
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      JWT_SECRET: ${JWT_SECRET}
      JWT_ALGORITHM: ${JWT_ALGORITHM}
      ACCESS_TOKEN_EXPIRE_MINUTES: ${ACCESS_TOKEN_EXPIRE_MINUTES}
      REFRESH_TOKEN_EXPIRE_DAYS: ${REFRESH_TOKEN_EXPIRE_DAYS}
      CV_API_TOKEN: ${CV_API_TOKEN}
      FRONTEND_URL: http://localhost:${FRONTEND_PORT:-3000}
    ports:
      - "${BACKEND_PORT:-8000}:8000"
    volumes:
      - frames:/app/data/frames
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks: [le_net]
    restart: unless-stopped

  frontend:
    build:
      context: ./src/frontend
      args:
        NEXT_PUBLIC_API_URL: http://localhost:${BACKEND_PORT:-8000}
    container_name: le_frontend
    depends_on: [backend]
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:${BACKEND_PORT:-8000}
    ports:
      - "${FRONTEND_PORT:-3000}:3000"
    networks: [le_net]
    restart: unless-stopped

  cv_detector:
    build: ./src/cv_detector
    container_name: le_cv_detector
    profiles: ["cv"]
    depends_on:
      backend: { condition: service_healthy }
    environment:
      BACKEND_URL: http://backend:8000
      CV_API_TOKEN: ${CV_API_TOKEN}
      CV_GROUP_ID: ${CV_GROUP_ID}
      CV_CONFIDENCE_THRESHOLD: ${CV_CONFIDENCE_THRESHOLD:-0.6}
      CV_CAMERA_INDEX: ${CV_CAMERA_INDEX:-0}
      MODEL_PATH: /app/treino_alimentos_final/weights/best.pt
    volumes:
      - ./src/cv_detector/treino_alimentos_final/weights:/app/treino_alimentos_final/weights:ro
    devices:
      - "/dev/video0:/dev/video0"  # Linux only
    networks: [le_net]
    restart: on-failure

networks:
  le_net:
    driver: bridge

volumes:
  pgdata:
  frames:

---

docker-compose.dev.yml (override para desenvolvimento):

services:
  backend:
    volumes:
      - ./src/backend:/app
      - /app/.venv  # excluir .venv do mount
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    environment:
      ENV: development

  frontend:
    volumes:
      - ./src/frontend:/app
      - /app/node_modules
      - /app/.next
    command: pnpm dev
    environment:
      NODE_ENV: development

---

Atualizar Makefile na raiz:

.PHONY: help up up-cv dev down nuke logs ps shell-backend shell-db migrate seed test

help:  ## Lista comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

up:  ## Sobe stack principal (sem cv_detector)
	docker compose up -d postgres backend frontend

up-cv:  ## Sobe stack completa (com cv_detector — Linux only)
	docker compose --profile cv up -d

dev:  ## Sobe em modo desenvolvimento com hot-reload
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

down:  ## Para todos os serviços
	docker compose --profile cv down

nuke:  ## Para tudo e remove volumes (CUIDADO: apaga dados!)
	docker compose --profile cv down -v

logs:  ## Logs do serviço (ex: make logs s=backend)
	docker compose logs -f $(s)

ps:  ## Lista containers
	docker compose ps

shell-backend:  ## Shell no container backend
	docker compose exec backend bash

shell-db:  ## psql no container postgres
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

migrate:  ## Aplica migrations
	docker compose exec backend alembic upgrade head

seed:  ## Carrega dados de desenvolvimento
	docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -f /seeds/dev_seed.sql

test:  ## Roda testes do backend
	docker compose exec backend pytest

Validar com: docker compose config
```

**Critério de aceite:**

- `make up` sobe 3 serviços, todos healthy
- `make up-cv` sobe os 4 (em Linux)
- `curl localhost:8000/health` → 200
- `curl localhost:3000` → HTML
- `make migrate && make seed` funciona
- `make logs s=backend` funciona
- `make nuke` limpa tudo

---

### FASE 10 — Polish e Documentação

**Prompt para Claude Code:**

```
1. Atualizar README.md raiz com:
   - Visão geral + diagrama de arquitetura ASCII
   - Pré-requisitos (Docker, .env)
   - Quick start (5 comandos)
   - Estrutura de pastas
   - Variáveis de ambiente documentadas
   - Como rodar cv_detector em Linux vs Mac/Windows
   - Como atualizar o modelo (substituir best.pt + restart)
   - Troubleshooting comum

2. docs/ na raiz:
   - architecture.md (decisões: monorepo, cookies httpOnly, async SQLA)
   - api.md (referência de endpoints com exemplos curl)
   - deployment.md (como deployar em VPS/cloud)

3. .github/workflows/ci.yml:
   - Jobs:
     * lint-backend (ruff check + ruff format --check)
     * type-check-backend (mypy)
     * test-backend (pytest com postgres service)
     * lint-frontend (eslint)
     * type-check-frontend (tsc --noEmit)
     * build-images (docker build de todos sem push)

4. SMOKE_TEST.md com checklist manual:
   [ ] make up sobe sem erros
   [ ] make migrate aplica
   [ ] make seed carrega
   [ ] Cadastrar professor com email @fecap.br + RA 6 dígitos
   [ ] Cadastrar aluno com email @edu.fecap.br + RA 8 dígitos
   [ ] Login com cada um
   [ ] Professor cria grupo (4+ alunos)
   [ ] cv_detector envia evidência (rodar local: python src/cv_detector/detect.py)
   [ ] Evidência aparece no Feed do aluno do grupo
   [ ] Aluno NÃO vê evidências de outros grupos
   [ ] Filtros do dashboard alteram dados
   [ ] Polling ranking atualiza
   [ ] Export CSV abre no Excel
   [ ] Export PDF é legível
   [ ] Trocar senha funciona
   [ ] Logout limpa sessão
   [ ] make nuke limpa tudo
```

**Critério de aceite:**

- Outro dev clona o repo e roda em < 10 minutos seguindo o README
- Smoke test 100% verde
- CI passa no GitHub

---

## 5. Fluxo de Execução Recomendado

```
Fase 0  ─▶  Fase 1  ─▶  Fase 2  ─▶  Fase 3
                                        │
                                        ▼
                          ┌─── Fase 9 (parcial: db+backend) ───┐
                          │   [Validar API com curl/Postman]   │
                          └────────────────────┬───────────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                       Fase 5  ─▶ 6 ─▶ 7 ─▶ 8                    Fase 4
                          │                                          │
                          └────────────────┬─────────────────────────┘
                                           ▼
                                       Fase 9 (completa)
                                           │
                                           ▼
                                       Fase 10
```

**Estimativa por fase** (com Claude Code, dev experiente):

- F0: 30min · F1: 1h · F2: 3h · F3: 3h · F4: 2h
- F5: 1h · F6: 3h · F7: 4h · F8: 3h · F9: 1h · F10: 1h

**Total:** 22-25 horas em 6-8 sessões.

---

## 6. Práticas com Claude Code

1. **Anexe `Escopo_LE_Frontend_v2.1.docx` no início de cada sessão** e cite a seção relevante.
2. **Uma fase por sessão.** Contexto poluído faz o Claude Code alucinar dependências.
3. **Peça plano antes de código** em fases complexas: _"Antes de codar, descreva o plano em bullets para eu validar."_
4. **Commits semânticos por fase**: `feat(backend): add auth module` etc.
5. **Mantenha um `DECISIONS.md`** com trade-offs (uv vs poetry, cookies vs Authorization header).
6. **Quando algo der errado**: copie erro completo + comando + contexto. Não peça "conserte" sem detalhes.

---

## 7. Riscos e Mitigações

| Risco                                | Mitigação                                                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Webcam no Docker em Mac/Windows      | Rodar `cv_detector` fora do Docker em dev. Documentado no README.                                       |
| `best.pt` versionado no Git          | Adicionar ao `.gitignore`. Documentar onde baixar/regenerar. Considerar Git LFS.                        |
| Cookies httpOnly + CORS              | `Access-Control-Allow-Origin` específico + `Allow-Credentials: true`. Frontend `withCredentials: true`. |
| Migrations conflitantes em time      | Cada PR cria sua migration. Nunca editar migration mergeada.                                            |
| Performance polling ranking          | Index em `evidences(group_id, detected_at)` + cache backend de 30s (Redis se escalar).                  |
| Validação duplicada (Zod + Pydantic) | Considerar `openapi-typescript` para gerar tipos TS a partir do FastAPI.                                |
| Path com espaço                      | Resolvido — pasta renomeada para `cv_detector`.                                                         |
| Pesos YOLO com tamanho >100MB        | Git LFS desde o início (configurar antes do primeiro commit grande).                                    |

---

## 8. Antes de Começar a Fase 0

Checklist:

- [ ] `git status` limpo (commitar/stashar pendências)
- [ ] Backup do `best.pt` em local externo (segurança)
- [ ] Pré-requisitos instalados (Docker + Claude Code)
- [ ] `.env` ainda não existe (vai ser criado após `.env.example`)
- [ ] Acesso de escrita na pasta do projeto

---

_Plano de Execução v2 — Lideranças Empáticas · FECAP_
_Estrutura: monorepo com src/ · cv_detector renomeado · docker-compose orquestrado_
