import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { RequireAuth } from "@/components/require-auth";
import { Btn, Card, Chip, PageHeader } from "@/components/le-ui";
import { useEvidencesFeed, useMyGroup, type FoodCategory } from "@/lib/queries";
import { useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/evidencias")({
  component: () => <RequireAuth><Evidencias /></RequireAuth>,
});

function PgBtn({ active, children }: { active?: boolean; children: React.ReactNode }) {
  return (
    <span className="w-8 h-8 inline-flex items-center justify-center rounded-lg text-[13px] font-semibold font-mono"
      style={{
        background: active ? "var(--forest)" : "#fff",
        color: active ? "#fff" : "var(--body-color)",
        border: active ? "none" : "1px solid var(--hairline)",
      }}>{children}</span>
  );
}

function FrameThumb({ cat, url }: { cat: FoodCategory; url?: string }) {
  const colors: Record<FoodCategory, string> = {
    arroz: "var(--arroz-light)",
    feijao: "var(--feijao-light)",
    macarrao: "var(--macarrao-light)",
    acucar: "#F4ECDD",
    oleo: "#FFF5D9",
    fuba: "#FBE6B5",
  };
  const dot: Record<FoodCategory, string> = {
    arroz: "var(--arroz)",
    feijao: "var(--feijao)",
    macarrao: "var(--macarrao)",
    acucar: "#6B4F1A",
    oleo: "#8C6B14",
    fuba: "#7A5414",
  };
  return (
    <div className="relative w-16 h-16 rounded-[10px] border border-hairline overflow-hidden flex items-center justify-center"
      style={{ background: `linear-gradient(135deg, ${colors[cat]}, var(--cream))` }}>
      {url ? (
        <img src={url} alt={cat} className="absolute inset-0 w-full h-full object-cover" />
      ) : (
        <svg width="100%" height="100%" className="absolute opacity-50">
          <defs>
            <pattern id={`gr-${cat}`} width="6" height="6" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1.2" fill={dot[cat]} opacity="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill={`url(#gr-${cat})`} />
        </svg>
      )}
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--forest)" strokeWidth="1.8"
        className="absolute top-1 right-1 bg-white/85 rounded p-[2px]">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
        <circle cx="12" cy="13" r="4" />
      </svg>
    </div>
  );
}

function Evidencias() {
  const { user } = useAuth();
  const myGroup = useMyGroup();
  const groupId = user?.role === "professor" ? undefined : myGroup.data?.group?.id;
  const [filter, setFilter] = useState<FoodCategory | "all">("all");
  const [page, setPage] = useState(1);
  const PER = 8;
  const feed = useEvidencesFeed({ groupId, limit: 200, category: filter });
  const all = feed.data ?? [];
  const totalPages = Math.max(1, Math.ceil(all.length / PER));
  const rows = all.slice((page - 1) * PER, page * PER);

  const catLabel = (c: FoodCategory) => ({
    arroz: "Arroz",
    feijao: "Feijão",
    macarrao: "Macarrão",
    acucar: "Açúcar",
    oleo: "Óleo",
    fuba: "Fubá",
  }[c]);
  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")}/${d.getFullYear()}`;
  };
  const fmtTime = (iso: string) => new Date(iso).toLocaleTimeString("pt-BR", { hour12: false });

  return (
    <div className="h-full overflow-auto pb-24 lg:pb-0">
      <PageHeader
        title="Feed de Evidências"
        subtitle={user?.role === "professor" ? "Registros automáticos de contagens — todos os grupos" : `Registros automáticos de contagens — ${myGroup.data?.group?.name ?? "seu grupo"}`}
        right={<><Btn kind="ghost" sm>↓ CSV</Btn><Btn kind="primary" sm>↓ PDF</Btn></>}
      />

      <div className="mx-4 sm:mx-6 lg:mx-8 mt-4 sm:mt-5 p-3 sm:p-4 rounded-lg flex gap-2 sm:gap-3 items-start"
        style={{ background: "var(--sage)", borderLeft: "4px solid var(--brand-accent)" }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--forest)" strokeWidth="2" className="shrink-0 mt-[2px]">
          <rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
        <div className="text-[12px] sm:text-[13px] leading-snug" style={{ color: "var(--forest)" }}>
          <strong>{user?.role === "professor" ? "Visão administrativa." : "Você está visualizando apenas os registros do seu grupo."}</strong>
          <span className="ml-1 sm:ml-2 opacity-85 block sm:inline" style={{ color: "var(--green)" }}>
            {user?.role === "professor" ? "Todos os grupos da campanha estão listados." : "Outros grupos não têm acesso a essas evidências."}
          </span>
        </div>
      </div>

      <div className="px-4 sm:px-6 lg:px-8 py-4 sm:py-5 flex flex-col sm:flex-row gap-2 sm:gap-[10px] sm:items-center">
        <div className="flex gap-1 p-1 bg-white rounded-full border border-hairline overflow-x-auto -mx-1 px-1">
          {(["all", "arroz", "feijao", "acucar", "macarrao", "oleo", "fuba"] as const).map((k) => (
            <button key={k} onClick={() => { setFilter(k); setPage(1); }}
              className="px-[14px] py-[6px] rounded-full text-[12px] font-semibold transition-colors whitespace-nowrap shrink-0"
              style={{
                background: filter === k ? "var(--forest)" : "transparent",
                color: filter === k ? "#fff" : "var(--body-color)",
              }}>
              {k === "all" ? "Todos" : catLabel(k)}
            </button>
          ))}
        </div>
        <div className="sm:ml-auto text-[12px] text-soft">
          {feed.isLoading ? "Carregando…" : `${all.length} registros encontrados`}
        </div>
      </div>

      <div className="px-4 sm:px-6 lg:px-8 pb-6 lg:pb-8">
        <Card style={{ padding: 0 }}>
          {/* Desktop table header */}
          <div className="hidden lg:grid px-6 py-[14px] text-[11px] font-bold tracking-[1.2px] text-soft border-b border-hairline bg-cream"
            style={{ gridTemplateColumns: "110px 1fr 130px 120px 1fr 80px" }}>
            <span>FRAME</span><span>CATEGORIA</span><span>DATA</span><span>HORA</span><span>VALIDADO POR</span><span>AÇÕES</span>
          </div>
          {/* Desktop table rows */}
          <div className="hidden lg:block">
            {rows.map((r, i) => (
              <div key={r.id} className="grid items-center px-6 py-[14px] border-b border-hairline text-[13px]"
                style={{
                  gridTemplateColumns: "110px 1fr 130px 120px 1fr 80px",
                  background: i % 2 === 0 ? "#fff" : "var(--cream)",
                }}>
                <FrameThumb cat={r.category as FoodCategory} url={r.frame_url} />
                <span><Chip tone={r.category as FoodCategory}>{catLabel(r.category as FoodCategory)}</Chip></span>
                <span className="font-mono text-[13px] text-body">{fmtDate(r.detected_at)}</span>
                <span className="font-mono text-[13px] text-body">{fmtTime(r.detected_at)}</span>
                <span className="flex items-center gap-2">
                  <span className="w-[22px] h-[22px] rounded-full inline-flex items-center justify-center" style={{ background: "var(--sage)" }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
                  </span>
                  <span className="text-[13px] font-medium">Sistema CV</span>
                  <span className="text-[11px] text-soft font-mono">· {(Number(r.confidence) * 100).toFixed(1)}%</span>
                </span>
                <span className="flex gap-2 text-soft">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" /><circle cx="12" cy="12" r="3" /></svg>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" /></svg>
                </span>
              </div>
            ))}
          </div>
          {/* Mobile cards */}
          <div className="lg:hidden divide-y divide-hairline">
            {rows.map((r) => (
              <div key={r.id} className="flex gap-3 p-4 items-start">
                <FrameThumb cat={r.category as FoodCategory} url={r.frame_url} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <Chip tone={r.category as FoodCategory}>{catLabel(r.category as FoodCategory)}</Chip>
                    <span className="text-[11px] text-soft font-mono">{(Number(r.confidence) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-[13px] text-body font-mono">
                    {fmtDate(r.detected_at)} · {fmtTime(r.detected_at)}
                  </div>
                  <div className="flex items-center gap-1 mt-1 text-[11px] text-soft">
                    <span className="w-[16px] h-[16px] rounded-full inline-flex items-center justify-center" style={{ background: "var(--sage)" }}>
                      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
                    </span>
                    Sistema CV
                  </div>
                </div>
              </div>
            ))}
          </div>
          {rows.length === 0 && (
            <div className="px-6 py-12 text-center text-soft text-sm">
              {feed.isLoading ? "Carregando evidências…" : "Nenhuma evidência registrada para este filtro."}
            </div>
          )}
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 px-4 sm:px-6 py-4 bg-cream">
            <span className="text-[12px] text-soft">
              Mostrando <strong className="text-ink">{rows.length === 0 ? 0 : (page - 1) * PER + 1}–{(page - 1) * PER + rows.length}</strong> de <strong className="text-ink">{all.length}</strong> registros
            </span>
            <div className="flex gap-[6px] flex-wrap">
              <button onClick={() => setPage((p) => Math.max(1, p - 1))}><PgBtn>←</PgBtn></button>
              {Array.from({ length: Math.min(totalPages, 4) }).map((_, i) => {
                const n = i + 1;
                return <button key={n} onClick={() => setPage(n)}><PgBtn active={page === n}>{n}</PgBtn></button>;
              })}
              {totalPages > 4 && <PgBtn>···</PgBtn>}
              {totalPages > 4 && <button onClick={() => setPage(totalPages)}><PgBtn active={page === totalPages}>{totalPages}</PgBtn></button>}
              <button onClick={() => setPage((p) => Math.min(totalPages, p + 1))}><PgBtn>→</PgBtn></button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
