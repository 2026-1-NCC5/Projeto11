import { defineConfig } from "@lovable.dev/vite-tanstack-config";

// Build em modo SPA (sem SSR). O TanStack Start pre-renderiza um shell
// estático que boota a aplicação no client. Cloudflare adapter desabilitado.
// Saída em `dist/client/` é servida estática por Vercel.
export default defineConfig({
  cloudflare: false,
  tanstackStart: {
    spa: {
      enabled: true,
      prerender: { outputPath: "/index" },
    },
  },
  vite: {
    server: { port: 3000, strictPort: true },
    preview: { port: 3000, strictPort: true },
  },
});
