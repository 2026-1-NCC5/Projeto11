# `src/backend/sql/`

Scripts SQL canônicos do projeto **Lideranças Empáticas**. São o ponto de partida — o backend (FastAPI + Alembic) consome dados a partir do schema definido aqui.

## Arquivos

| Arquivo              | Tipo | Idempotente | O que faz                                                                                |
| -------------------- | ---- | ----------- | ---------------------------------------------------------------------------------------- |
| `init.sql`           | DDL  | sim         | Extensões, ENUMs, tabelas, índices, trigger `updated_at`, RLS desabilitado.              |
| `storage_setup.sql`  | mix  | sim         | Cria bucket `frames` (privado) + policy de leitura para autenticados.                    |
| `seed.sql`           | DML  | sim (TRUNCATE no topo) | 2 professores, 12 alunos, 3 grupos, 2 sessions (1 ativa + 1 encerrada), 30 evidences. |

> **Auth** é feito pelo FastAPI (JWT) — RLS é desabilitado de propósito. **Não** rodar nenhum script com Auth/Realtime do Supabase ativos.

## Aplicação — caminho recomendado (MCP Supabase via Claude Code)

Já está configurado no `.mcp.json` da raiz. No Claude Code basta pedir:

```
"Aplique src/backend/sql/init.sql no Supabase via MCP."
"Aplique src/backend/sql/storage_setup.sql via MCP."
"Aplique src/backend/sql/seed.sql via MCP."
```

A aplicação inicial deste schema **já foi feita** via MCP em 2026-05-07 (migrations registradas: `drop_legacy_prototype`, `init_schema`, `storage_setup_frames`).

## Aplicação manual (Supabase Studio → SQL Editor)

Caso esteja sem MCP:

1. Abrir o projeto no [Supabase Dashboard](https://supabase.com/dashboard).
2. Menu lateral → **SQL Editor** → **New query**.
3. Colar o conteúdo de `init.sql`, executar (`Ctrl+Enter`).
4. Repetir para `storage_setup.sql`, depois `seed.sql`.

## Aplicação via Alembic (futuro)

Quando o backend for codado, este schema vira a primeira revisão Alembic. O comando padrão será:

```bash
make migrate    # equivalente a alembic upgrade head dentro do container
```

Até lá, `init.sql` é a fonte da verdade do schema.

## Limpeza do bucket legado `evidencias`

O bucket `evidencias` (público, do protótipo antigo) **não pode ser apagado via SQL** — o Supabase bloqueia `DELETE` direto em `storage.buckets`/`storage.objects`. Limpe pelo Studio:

1. Storage → `evidencias` → selecionar tudo → **Delete**.
2. Storage → `evidencias` → menu (`⋯`) → **Delete bucket**.

Após isso só `frames` (privado) deve existir.

## Credenciais do seed

Todos os usuários (professores e alunos) têm a senha `Senha@123`. O hash é gerado pelo Postgres via `crypt('Senha@123', gen_salt('bf'))` (bcrypt) — cada linha terá um hash diferente, conforme prática segura.

E-mails de teste úteis:

- Professor: `marcos.nakatsugawa@fecap.br` / `rafael.rossetti@fecap.br`
- Alunos (Equipe Alfa): `flavia.costa@edu.fecap.br`, `guilherme.muniz@edu.fecap.br`, `lucas.moreira@edu.fecap.br`, `maria.foloni@edu.fecap.br`
- Demais alunos seguem o padrão `nome.sobrenome@edu.fecap.br`.

## Como verificar o estado do banco via MCP

No Claude Code:

```
"Liste tabelas do public e conte linhas em users, groups, group_members, detection_sessions, evidences, refresh_tokens."
```

Ou diretamente:

```sql
SELECT 'users' AS table, count(*) FROM public.users
UNION ALL SELECT 'groups',             count(*) FROM public.groups
UNION ALL SELECT 'group_members',      count(*) FROM public.group_members
UNION ALL SELECT 'detection_sessions', count(*) FROM public.detection_sessions
UNION ALL SELECT 'evidences',          count(*) FROM public.evidences
UNION ALL SELECT 'refresh_tokens',     count(*) FROM public.refresh_tokens;
```

## Quando reescrever?

- **`init.sql`** — só altere se o escopo (`Escopo_LE_Frontend_v2.1.docx`) mudar.
- **`seed.sql`** — pode crescer com mais dados de demo. Mantenha senhas/UUIDs estáveis.
- **`storage_setup.sql`** — só altere ao introduzir novos buckets ou políticas.
