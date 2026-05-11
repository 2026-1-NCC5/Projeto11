# Frontend — Lideranças Empáticas

SPA em **TanStack Start + React 19 + TypeScript strict**, com **TailwindCSS + shadcn/ui**, **TanStack Query** para data fetching e **Zustand** para estado global. Build estático servido pela **Vercel**.

---

## Pré-requisitos

- [Node.js 20+](https://nodejs.org/) (LTS)
- [pnpm](https://pnpm.io/installation) (gerenciador oficial do projeto)
- Backend rodando localmente em `http://localhost:8000` (veja `src/backend/README.md`)

---

## Variáveis de ambiente

Crie `.env.local` em `src/frontend/`:

```env
VITE_API_URL=http://localhost:8000/api/v1
```

Para apontar para um backend exposto via ngrok (demo), troque pela URL pública. A Vercel usa `NEXT_PUBLIC_API_URL`/`VITE_API_URL` configurada no painel — preferir gerenciá-las via MCP Vercel.

---

## Rodar em desenvolvimento

A partir de `src/frontend/`:

```bash
pnpm install           # instala dependências (gera node_modules + bun.lock)
pnpm dev               # sobe Vite dev server em http://localhost:5173
```

---

## Build e preview

```bash
pnpm build             # build de produção (gera dist/)
pnpm build:dev         # build em modo development (para debug)
pnpm preview           # servidor estático local sobre dist/
```

---

## Lint e formatação

```bash
pnpm lint              # ESLint
pnpm format            # Prettier (write)
```

---

## Estrutura

```
src/frontend/
├── src/
│   ├── routes/         # rotas (dashboard, grupos, perfil, login, etc.)
│   ├── components/     # componentes compartilhados
│   ├── lib/            # api clients, queries, utils
│   ├── assets/         # imagens e SVGs
│   ├── styles.css      # tailwind + tokens
│   ├── router.tsx
│   ├── routeTree.gen.ts
│   └── start.ts        # entrypoint TanStack Start
├── vite.config.ts
├── tsconfig.json
├── package.json
└── vercel.json         # config de deploy
```

---

## Deploy (Vercel)

O projeto está conectado ao GitHub: **push em `main` dispara deploy automático** de produção. PRs criam preview deployments.

Comandos úteis (via Vercel CLI, opcional):

```bash
vercel link                              # conecta este diretório ao projeto Vercel
vercel env pull .env.local               # baixa env vars do ambiente para local
vercel --prod                            # deploy manual de produção
```

Para inspecionar deploys, ler logs ou trocar `NEXT_PUBLIC_API_URL` durante a apresentação, prefira o **MCP Vercel** no Claude Code.

---

## Convenções

- **Server Components por padrão**; `"use client"` só onde indispensável.
- `any` **proibido** — use `unknown` + narrowing.
- Cores **sempre via tokens** (Tailwind) — nada de hexadecimal hardcoded.
- Alias `@/*` mapeia para `src/*`.
- Validação de formulário com **React Hook Form + Zod**; o schema Zod deve espelhar o Pydantic do backend.
