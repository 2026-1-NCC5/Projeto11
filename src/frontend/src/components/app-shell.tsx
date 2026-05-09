import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { BrandLogo, FecapMark } from "./brand";
import { useAuth } from "@/lib/auth-context";

const NAV = [
  { id: "dashboard", to: "/dashboard", label: "Dashboard",
    icon: "M3 12l9-9 9 9v9a2 2 0 0 1-2 2h-4v-7h-6v7H5a2 2 0 0 1-2-2z" },
  { id: "evidencias", to: "/evidencias", label: "Evidências",
    icon: "M4 5h12l4 4v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zm6 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6z" },
  { id: "grupos", to: "/grupos", label: "Grupos",
    icon: "M8 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm8 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM2 19a6 6 0 0 1 12 0v2H2zm14-1a5 5 0 0 1 6 5h-6z" },
  { id: "perfil", to: "/perfil", label: "Perfil",
    icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm0 2c-4 0-8 2-8 6v2h16v-2c0-4-4-6-8-6z" },
] as const;

function Sidebar() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const { user } = useAuth();
  const displayName = user?.full_name ?? "Visitante";
  const role = user?.role === "professor" ? "Professor" : "Aluno";
  const initials = displayName.split(" ").map((s) => s[0]).slice(0, 2).join("").toUpperCase();

  return (
    <aside
      className="flex flex-col text-white border-r"
      style={{ width: 220, minWidth: 220, background: "var(--forest)", borderColor: "var(--forest)" }}
    >
      <div className="px-5 pt-6 pb-5 flex flex-col items-center gap-3 border-b border-white/10">
        <BrandLogo size={84} />
        <div className="text-[11px] tracking-[1.4px] font-semibold text-white/65">SISTEMA DE ARRECADAÇÃO</div>
      </div>
      <nav className="px-3 py-4 flex flex-col gap-[2px] flex-1">
        <div className="text-[10px] tracking-[1.5px] font-bold text-white/45 px-3 pt-2 pb-[6px]">NAVEGAÇÃO</div>
        {NAV.map((n) => {
          const active = path.startsWith(n.to);
          return (
            <Link
              key={n.id} to={n.to}
              className="flex items-center gap-3 px-[14px] py-[11px] rounded-[10px] text-[14px] transition-colors"
              style={{
                background: active ? "var(--brand-accent)" : "transparent",
                color: active ? "#fff" : "rgba(255,255,255,0.78)",
                fontWeight: active ? 600 : 500,
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d={n.icon} /></svg>
              <span>{n.label}</span>
              {active && <span className="ml-auto w-[6px] h-[6px] rounded-full bg-white" />}
            </Link>
          );
        })}
      </nav>
      <div className="p-4 border-t border-white/10">
        <div className="flex items-center gap-[10px] p-2 rounded-[10px] bg-white/[.06]">
          <div
            className="w-9 h-9 rounded-full flex items-center justify-center font-bold text-[13px]"
            style={{ background: "var(--brand-accent)" }}
          >{initials}</div>
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold whitespace-nowrap overflow-hidden text-ellipsis">{displayName}</div>
            <div className="text-[11px] text-white/55">{role}</div>
          </div>
        </div>
        <div className="mt-[14px] flex items-center justify-between px-1">
          <span className="text-[10px] tracking-[1.2px] text-white/40">POWERED BY</span>
          <FecapMark color="#fff" height={14} />
        </div>
      </div>
    </aside>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex w-full min-h-screen bg-cream text-body">
      <Sidebar />
      <main className="flex-1 relative overflow-hidden">{children}</main>
    </div>
  );
}
