# Decisões Arquiteturais — Lideranças Empáticas

> Architecture Decision Records (ADRs). Cada decisão é registrada uma vez
> e nunca alterada. Se a decisão mudar, crie um novo ADR com status
> "Substitui ADR-XXX" e marque o antigo como "Substituído".

**Formato:** Status · Contexto · Decisão · Consequências · Alternativas

---

## ADR-001 — Monorepo com pasta `src/` agregadora

**Status:** Aceito · 2025-XX-XX

**Contexto:**
O projeto tem 4 componentes (backend, frontend, cv_detector, db) que evoluem juntos e precisam estar versionados como uma unidade coerente. Além disso, o template institucional da FECAP exige a pasta `src/` como raiz do código.

**Decisão:**
Adotar monorepo com a estrutura:

```
liderancas-empaticas/
├── docker-compose.yml          (raiz)
├── Makefile                    (raiz)
├── .env.example                (raiz)
└── src/
    ├── backend/
    ├── frontend/
    ├── cv_detector/
    └── db/
```

**Consequências:**

- ✅ Atomic commits cruzando camadas (ex: muda schema + backend + frontend num único PR)
- ✅ Setup de novo dev: 1 clone, 1 `make up`
- ✅ Conformidade com template FECAP
- ⚠️ Build contexts do Docker são `./src/X` em vez de `./X` — cuidado em `docker-compose.yml`
- ⚠️ CI precisa de paths-filter para evitar buildar tudo a cada PR

**Alternativas consideradas:**

- **Polirepo (4 repos separados):** rejeitado — sincronização de versões vira problema, contributor experience pior.
- **Monorepo sem `src/`:** rejeitado — quebra padrão da FECAP.
- **Nx ou Turborepo:** rejeitado — overkill para 4 serviços com stacks distintas (não há código compartilhado entre eles).

---

## ADR-002 — Renomeação de `frontend detector` para `cv_detector`

**Status:** Aceito · 2025-XX-XX

**Contexto:**
A pasta original estava nomeada como `frontend detector` (com espaço). Espaço em path causa problemas em Docker build context, Makefile, scripts shell e imports Python.

**Decisão:**
Renomear para `cv_detector` (snake_case, sem espaço, semanticamente preciso — é um detector de visão computacional, não de frontend).

**Consequências:**

- ✅ Compatível com toolchain padrão sem escapes
- ✅ Nome reflete melhor a função do componente
- ⚠️ Histórico Git pode quebrar git blame se renomeado sem `git mv`

**Alternativas consideradas:**

- Manter o nome com escape em todo lugar: rejeitado pelo custo de manutenção.
- Renomear para `frontend_detector`: aceitável mas mantém o nome enganoso (não é frontend).

---

## ADR-003 — Cookies httpOnly em vez de Authorization header

**Status:** Aceito · 2025-XX-XX

**Contexto:**
Tokens JWT precisam ser armazenados de algum jeito no client. As opções são `localStorage`, `sessionStorage` ou cookies (com flag httpOnly).

**Decisão:**
Armazenar **access token** e **refresh token** em **cookies httpOnly + Secure (em prod) + SameSite=Lax**. Frontend usa Axios com `withCredentials: true`. Backend lê tokens do cookie via FastAPI.

**Consequências:**

- ✅ Imune a XSS (JavaScript não consegue ler cookies httpOnly)
- ✅ Refresh automático funciona sem expor tokens
- ⚠️ CORS exige configuração específica (`Access-Control-Allow-Credentials: true` + origin específico, não `*`)
- ⚠️ Em desenvolvimento HTTP, cookie `Secure` não funciona — precisa de flag condicional por ambiente
- ⚠️ Mobile apps nativos (futuro) não conseguem ler cookies — vão precisar de Authorization header. Ressalva documentada.

**Alternativas consideradas:**

- **localStorage:** rejeitado — vulnerável a XSS (qualquer biblioteca npm pode roubar).
- **sessionStorage:** rejeitado — perde sessão ao fechar aba (UX ruim).
- **Authorization header com tokens em memória:** rejeitado — refresh complexo, F5 perde sessão.

---

## ADR-004 — PostgreSQL local via Docker, não Supabase

**Status:** Aceito · 2025-XX-XX

**Contexto:**
O projeto inicialmente cogitou Supabase pela facilidade. Mas a apresentação na FECAP exige um setup que rode 100% offline e o professor possa inspecionar tudo.

**Decisão:**
PostgreSQL 16 rodando em container próprio, com `init.sql` versionado e migrations Alembic. Conexão local via `DATABASE_URL`. Em produção (cloud), pode ser Railway Postgres, Supabase ou Neon — a app é agnóstica.

**Consequências:**

- ✅ Roda 100% offline (importante para apresentação)
- ✅ Schema versionado e auditável
- ✅ Sem vendor lock-in
- ⚠️ Sem Realtime/Auth/Storage prontos (teríamos com Supabase) — implementamos manualmente

**Alternativas consideradas:**

- **Supabase (managed):** rejeitado para dev — depende de internet, vendor lock-in. **Aceitável para produção** se preço escalar mal no Railway.
- **SQLite:** rejeitado — não suporta UUIDs nativos, ENUMs, CHECK condicionais.

---

## ADR-005 — uv como gerenciador de pacotes Python

**Status:** Aceito · 2025-XX-XX

**Contexto:**
Backend e cv_detector são Python. Precisamos de um gerenciador que seja rápido, reprodutível e compatível com Docker.

**Decisão:**
Usar **uv** (https://github.com/astral-sh/uv) como gerenciador. Lockfile `uv.lock` versionado. `pyproject.toml` como manifesto.

**Consequências:**

- ✅ 10-100x mais rápido que pip/poetry
- ✅ Resolução determinística
- ✅ Compatível com PEP 621 (`pyproject.toml`)
- ⚠️ Ferramenta nova (2024) — equipe pode não conhecer. Documentar comandos no README.

**Alternativas consideradas:**

- **Poetry:** maduro mas lento, especialmente em CI.
- **pip + requirements.txt:** sem lockfile robusto, problemas de reprodutibilidade.
- **pip-tools:** funciona mas exige mais boilerplate.

---

## ADR-006 — pnpm como gerenciador de pacotes Node

**Status:** Aceito · 2025-XX-XX

**Contexto:**
Frontend Next.js precisa de gerenciador de pacotes Node.

**Decisão:**
Usar **pnpm**. `pnpm-lock.yaml` versionado.

**Consequências:**

- ✅ Mais rápido que npm/yarn
- ✅ Economiza disco (content-addressable store)
- ✅ Strict por padrão (`hoisting` controlado)
- ⚠️ Alguns pacotes legados podem quebrar com strict mode — usar `node-linker=hoisted` em casos específicos.

**Alternativas consideradas:**

- **npm:** lento e ruim para monorepo.
- **yarn classic (1.x):** legado.
- **yarn berry (2+):** PnP causa problemas com várias libs.
- **bun:** muito novo para projeto acadêmico — risco de bugs.

---

## ADR-007 — shadcn/ui em vez de MUI/Chakra/Mantine

**Status:** Aceito · 2025-XX-XX

**Contexto:**
O escopo exige design minimalista e personalizado, com tokens de cor específicos da identidade Lideranças Empáticas. Bibliotecas opinionadas dificultam customização profunda.

**Decisão:**
Usar **shadcn/ui** — componentes copy-paste construídos sobre Radix UI primitives + Tailwind. Cada componente fica no nosso código, podemos editar livremente.

**Consequências:**

- ✅ Customização total (componentes são nossos)
- ✅ Acessibilidade (Radix garante)
- ✅ Bundle size pequeno (só o que importamos)
- ⚠️ Updates manuais (re-rodar `pnpm dlx shadcn add` quando houver fix upstream)

**Alternativas consideradas:**

- **Material UI:** opinionado demais, brigas com Tailwind.
- **Chakra UI:** boa mas estilo padrão muito diferente do briefing.
- **Mantine:** excelente mas a estética padrão tem cara de "todo dashboard moderno" — queremos diferenciação.

---

## ADR-008 — Validação dupla com Zod (frontend) + Pydantic (backend)

**Status:** Aceito · 2025-XX-XX

**Contexto:**
Regras de negócio (formato de RA, domínio de e-mail, força de senha) precisam ser validadas no frontend para UX (feedback inline) e no backend para segurança.

**Decisão:**
Manter **schemas em duplicata** (Zod no frontend, Pydantic no backend) com testes que verificam que ambos rejeitam/aceitam os mesmos inputs.

**Consequências:**

- ✅ Frontend dá feedback imediato (sem round-trip)
- ✅ Backend nunca confia no client (defesa em profundidade)
- ⚠️ Duplicação de código — risco de drift se uma camada for alterada sem a outra
- ⚠️ Mitigação: testes de contrato + considerar futuro `openapi-typescript` para gerar tipos TS a partir do FastAPI

**Alternativas consideradas:**

- **Validação só no backend:** UX pobre (feedback só no submit).
- **Validação só no frontend:** **inseguro** — qualquer cliente HTTP pula a validação.
- **Schema único compartilhado (ex: zod-to-pydantic):** ferramenta imatura, exige Node no backend.

---

## ADR-009 — Dedup de evidências por hash composto

**Status:** Aceito · 2025-XX-XX

**Contexto:**
O CV detector roda a 30 FPS. Um pacote parado em frente à câmera por 10 segundos geraria 300 evidências idênticas, o que poluiria os dados e o ranking.

**Decisão:**
Cada evidência tem um `dedup_hash` UNIQUE no banco, calculado como:

```
sha256(category | bbox_quantizado_em_grid_10px | bucket_de_5_segundos)
```

Backend faz `INSERT ... ON CONFLICT (dedup_hash) DO NOTHING`.

**Consequências:**

- ✅ Mesmo objeto parado: ~12 evidências/min em vez de ~1800
- ✅ Backend simples (constraint do DB faz o trabalho)
- ⚠️ Granularidade do bucket é trade-off: 5s pode perder pacotes movidos rapidamente.
  - Decisão: começar com 5s, ajustar baseado em testes reais
- ⚠️ Quantização do bbox é sensível à câmera. Documentar e parametrizar.

**Alternativas consideradas:**

- **Tracking de objeto (DeepSORT, ByteTrack):** mais robusto mas pesado em CPU/GPU.
- **Dedup só por timestamp:** falha se 2 pacotes diferentes passam no mesmo segundo.
- **Dedup no detector:** rejeitado — single source of truth deve ser o backend.

---

## ADR-010 — Mantém pesos YOLO versionados no Git

**Status:** Aceito · 2025-XX-XX (substitui orientação inicial do PLANO_EXECUCAO_LE_v2.md)

**Contexto:**
A orientação inicial era ignorar `*.pt` no `.gitignore` para evitar repos pesados. Porém, o professor avaliador precisa conseguir clonar e rodar o sistema completo, e o modelo treinado é parte essencial do entregável.

**Decisão:**
Versionar `best.pt` no Git. Se o arquivo passar de 100MB (limite do GitHub), usar Git LFS.

**Consequências:**

- ✅ Avaliador clona e roda direto
- ✅ Reprodutibilidade total
- ⚠️ Tamanho do clone aumenta — aceitável para projeto acadêmico
- ⚠️ Treino do modelo continua fora do versionamento (datasets, runs/\*) — só os pesos finais

**Alternativas consideradas:**

- **DVC (Data Version Control):** profissional mas exige bucket S3/GCS, complica avaliação.
- **Hospedagem externa (Drive, Dropbox):** rejeitado — link pode quebrar, professor não tem garantia de acesso.

---

## ADR-XXX — Template para Novos ADRs

**Status:** [Aceito | Substituído por ADR-YYY | Rejeitado | Proposto] · YYYY-MM-DD

**Contexto:**
Por que estamos tomando essa decisão agora? Qual problema queremos resolver?

**Decisão:**
O que vamos fazer? Seja específico e direto.

**Consequências:**

- ✅ O que ganhamos
- ⚠️ O que perdemos ou trade-offs aceitos

**Alternativas consideradas:**

- **Opção X:** por que foi rejeitada
- **Opção Y:** por que foi rejeitada

---

_Mantenha este documento conforme decisões arquiteturais surgem._
_Não edite ADRs aceitos — crie novos que os substituam._
