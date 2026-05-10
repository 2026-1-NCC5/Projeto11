import { defineConfig, loadEnv } from "vite";
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import tsconfigPaths from "vite-tsconfig-paths";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import react from "@vitejs/plugin-react";

// SPA mode (sem SSR). TanStack Start pre-renderiza um shell estático
// que boota a aplicação no client. Saída em `dist/client/` é servida
// estática pela Vercel.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const define: Record<string, string> = {};
  for (const [k, v] of Object.entries(env)) {
    define[`import.meta.env.${k}`] = JSON.stringify(v);
  }

  return {
    define,
    resolve: {
      alias: {
        "@": path.resolve(process.cwd(), "src"),
      },
      dedupe: [
        "react",
        "react-dom",
        "react/jsx-runtime",
        "react/jsx-dev-runtime",
        "@tanstack/react-query",
        "@tanstack/query-core",
      ],
    },
    server: { port: 3000, strictPort: true, host: "::" },
    preview: { port: 3000, strictPort: true },
    plugins: [
      tailwindcss(),
      tsconfigPaths({ projects: ["./tsconfig.json"] }),
      tanstackStart({
        spa: {
          enabled: true,
          prerender: { outputPath: "/index" },
        },
        importProtection: {
          behavior: "error",
          client: {
            files: ["**/server/**"],
            specifiers: ["server-only"],
          },
        },
      }),
      react(),
    ],
  };
});
