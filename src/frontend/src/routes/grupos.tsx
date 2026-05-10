import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { RequireAuth } from "@/components/require-auth";
import { Btn, Card, Chip, Digits, Field, Input, PageHeader } from "@/components/le-ui";
import { useAllGroupsAdmin, useGroupsRanking, useMyGroup } from "@/lib/queries";
import { useAuth } from "@/lib/auth-context";
import { groupsApi, type StudentSearch } from "@/lib/groups-api";
import { ApiError } from "@/lib/auth-api";

export const Route = createFileRoute("/grupos")({
  component: () => <RequireAuth><GruposRouter /></RequireAuth>,
});

function GruposRouter() {
  const { user } = useAuth();
  return user?.role === "professor" ? <ProfessorView /> : <AlunoView />;
}

/* ============ Aluno ============ */
function AlunoView() {
  const { data, isLoading } = useMyGroup();
  const ranking = useGroupsRanking();
  const rank = data ? (ranking.data ?? []).findIndex((g) => g.id === data.group?.id) + 1 : 0;
  const medal = rank === 1 ? "🥇" : rank === 2 ? "🥈" : rank === 3 ? "🥉" : "";

  return (
    <div className="h-full overflow-auto pb-24 lg:pb-0">
      <PageHeader title="Meu Grupo" subtitle="Você é membro do grupo abaixo" />
      <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-4 lg:gap-6">
        {isLoading && <Card style={{ padding: 32 }}>Carregando…</Card>}
        {!isLoading && !data && (
          <Card style={{ padding: 32 }}>
            <div className="text-soft">Você ainda não foi adicionado a um grupo. Procure seu professor.</div>
          </Card>
        )}
        {data && (
          <>
            <div className="relative overflow-hidden rounded-[20px] p-5 sm:p-7 lg:p-9 text-white"
              style={{ background: "var(--forest)" }}>
              <svg className="absolute opacity-[.05] hidden sm:block" style={{ right: -20, top: -20 }} width="280" height="280" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="96" fill="none" stroke="#fff" strokeWidth="2" />
                <circle cx="100" cy="100" r="76" fill="none" stroke="#fff" strokeWidth="2" />
              </svg>
              <div className="flex flex-col lg:flex-row lg:justify-between lg:items-end gap-4 relative">
                <div className="min-w-0">
                  <div className="text-[12px] tracking-[1.5px] font-bold text-white/60">SEU GRUPO</div>
                  <h2 className="my-2 text-[32px] sm:text-[40px] lg:text-[56px] font-semibold leading-none tracking-[-0.03em] break-words">{data.group?.name}</h2>
                  <div className="text-[13px] sm:text-sm text-white/70">
                    Criado em {data.group ? new Date(data.group.created_at).toLocaleDateString("pt-BR") : "—"}
                  </div>
                </div>
                <div className="flex gap-2 sm:gap-3">
                  <div className="flex-1 lg:flex-initial rounded-[14px] px-4 sm:px-[22px] py-3 sm:py-[14px]" style={{ background: "rgba(255,255,255,0.08)" }}>
                    <div className="text-[10px] tracking-[1.5px] font-bold text-white/55">ARRECADADO</div>
                    <div className="flex items-baseline gap-1 mt-1">
                      <Digits size={28} color="#fff">{data.kg}</Digits>
                      <span className="text-sm text-white/70">kg</span>
                    </div>
                  </div>
                  <div className="flex-1 lg:flex-initial rounded-[14px] px-4 sm:px-[22px] py-3 sm:py-[14px]" style={{ background: "var(--brand-accent)" }}>
                    <div className="text-[10px] tracking-[1.5px] font-bold text-white/70">POSIÇÃO</div>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className="text-sm text-white/70">#</span>
                      <Digits size={28} color="#fff">{rank || "—"}</Digits>
                      {medal && <span className="text-base ml-[6px]">{medal}</span>}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-between items-baseline gap-2 flex-wrap">
              <h3 className="m-0 text-[18px] sm:text-[20px] lg:text-[22px] font-semibold tracking-[-0.015em]">
                Integrantes <span className="text-soft font-medium">{data.members.length}/5</span>
              </h3>
              {data.members.length >= 4 && <Chip tone="success">✓ Grupo completo</Chip>}
              {data.members.length < 4 && <Chip tone="warning">⚠ Faltam integrantes</Chip>}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 lg:gap-4">
              {data.members.map((m) => (
                <Card key={m.user_id} style={{
                  padding: 18,
                  boxShadow: m.isMe ? "0 0 0 2px var(--brand-accent), 0 1px 1px rgba(0,0,0,0.04)" : undefined,
                }}>
                  <div className="flex gap-3 sm:gap-4 items-center">
                    <div className="w-[48px] h-[48px] sm:w-[52px] sm:h-[52px] rounded-full flex items-center justify-center text-[15px] sm:text-[17px] font-bold text-white shrink-0"
                      style={{ background: m.isMe ? "var(--brand-accent)" : "var(--forest)" }}>
                      {(m.full_name || "?").split(" ").map((s) => s[0]).slice(0, 2).join("").toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-[15px] sm:text-[16px] font-semibold text-ink truncate">{m.full_name}</span>
                        {m.isMe && <Chip tone="sage">VOCÊ</Chip>}
                        {m.isLeader && <Chip tone="gold">★ líder</Chip>}
                      </div>
                      <div className="text-[12px] sm:text-[13px] text-soft mt-[2px] truncate">{m.email}</div>
                      <div className="text-[12px] text-soft mt-1 font-mono">RA · {m.ra}</div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ============ Professor ============ */
function MiniStat({ label, value, sub, tone }: { label: string; value: string | number; sub: string; tone?: "forest" | "success" }) {
  const dark = tone === "forest";
  return (
    <Card className="p-4 sm:p-5" style={{ background: dark ? "var(--forest)" : "#fff", color: dark ? "#fff" : "var(--body-color)" }}>
      <div className="text-[11px] tracking-[1.5px] font-bold" style={{ color: dark ? "rgba(255,255,255,0.6)" : "var(--soft)" }}>{label}</div>
      <div className="flex items-baseline gap-2 mt-2">
        <span
          className="font-mono tabular tracking-tight text-[28px] sm:text-[36px] lg:text-[40px] font-bold leading-none"
          style={{ color: dark ? "#fff" : "var(--ink)", letterSpacing: "-0.03em" }}
        >{value}</span>
      </div>
      <div className="text-[12px] mt-1" style={{ color: dark ? "rgba(255,255,255,0.5)" : "var(--soft)" }}>{sub}</div>
    </Card>
  );
}

type AdminGroup = { id: string; name: string; count: number; kg: number; status: string; created_at: string };

function GroupAdminCard({
  g,
  onOpen,
}: {
  g: AdminGroup;
  onOpen: (id: string) => void;
}) {
  return (
    <Card className="p-5 sm:p-6">
      <div className="flex justify-between items-start gap-2">
        <div className="min-w-0">
          <h4 className="m-0 text-[17px] sm:text-[20px] font-semibold tracking-[-0.01em] truncate">{g.name}</h4>
          <div className="text-[12px] text-soft mt-1">
            {g.count} integrantes · {new Date(g.created_at).toLocaleDateString("pt-BR")}
          </div>
        </div>
        {g.status === "ativo" ? <Chip tone="success">● Ativo</Chip> : <Chip tone="warning">⚠ Incompleto</Chip>}
      </div>
      <div className="flex items-center gap-3 mt-[18px]">
        <div className="flex">
          {Array.from({ length: Math.min(g.count, 4) }).map((_, i) => (
            <div key={i} className="w-8 h-8 rounded-full text-white flex items-center justify-center text-[11px] font-bold border-2 border-white"
              style={{ background: "var(--forest)", marginLeft: i === 0 ? 0 : -8 }}>·</div>
          ))}
          {g.count > 4 && (
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold border-2 border-white"
              style={{ background: "var(--cream)", color: "var(--body-color)", marginLeft: -8 }}>+{g.count - 4}</div>
          )}
        </div>
        <div className="ml-auto flex items-baseline gap-1">
          <Digits size={26}>{g.kg}</Digits>
          <span className="text-[13px] text-soft">kg</span>
        </div>
      </div>
      <div className="flex flex-col sm:flex-row gap-2 mt-4 pt-4 border-t border-hairline">
        <Btn kind="ghost" sm className="flex-1" onClick={() => onOpen(g.id)}>Ver detalhes</Btn>
        <Btn kind="outline" sm className="flex-1" onClick={() => onOpen(g.id)}>Gerenciar</Btn>
      </div>
    </Card>
  );
}

type Panel = { mode: "create" } | { mode: "detail"; groupId: string } | null;

function ProfessorView() {
  const { data, isLoading } = useAllGroupsAdmin();
  const [panel, setPanel] = useState<Panel>(null);

  return (
    <div className="h-full overflow-auto relative pb-24 lg:pb-0">
      <PageHeader
        title="Gestão de Grupos"
        subtitle="Visão administrativa · Campanha 2026.1"
        right={<Btn kind="primary" sm onClick={() => setPanel({ mode: "create" })}>+ Novo Grupo</Btn>}
      />

      <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-4 lg:gap-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 lg:gap-4">
          <MiniStat label="TOTAL DE GRUPOS" value={data?.totals.totalGroups ?? 0} sub="ativos na campanha" tone="forest" />
          <MiniStat label="TOTAL DE ALUNOS" value={data?.totals.totalStudents ?? 0} sub="cadastrados em grupos" />
          <MiniStat label="GRUPOS COMPLETOS" value={data?.totals.completeGroups ?? 0} sub="com mín. 4 integrantes" tone="success" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 lg:gap-4">
          {(data?.groups ?? []).map((g) => (
            <GroupAdminCard key={g.id} g={g} onOpen={(id) => setPanel({ mode: "detail", groupId: id })} />
          ))}
          {!isLoading && (data?.groups ?? []).length === 0 && (
            <Card style={{ padding: 32 }}><div className="text-soft">Nenhum grupo cadastrado ainda.</div></Card>
          )}
        </div>
      </div>

      {panel?.mode === "create" && <CreateGroupPanel onClose={() => setPanel(null)} />}
      {panel?.mode === "detail" && (
        <GroupDetailPanel groupId={panel.groupId} onClose={() => setPanel(null)} />
      )}
    </div>
  );
}

/* ============ Painel: criar grupo ============ */

function PanelShell({
  title,
  eyebrow,
  onClose,
  footer,
  children,
}: {
  title: string;
  eyebrow: string;
  onClose: () => void;
  footer: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <>
      <div
        className="lg:hidden fixed inset-0 z-40 bg-black/50"
        onClick={onClose}
        aria-hidden
      />
      <div
        className="fixed lg:absolute z-50 inset-0 lg:inset-auto lg:right-0 lg:top-0 lg:bottom-0 flex flex-col bg-white border-l border-hairline w-full lg:w-[480px]"
        style={{ boxShadow: "-12px 0 32px rgba(20,56,36,0.08)" }}
      >
        <div className="p-5 sm:p-6 lg:p-7 border-b border-hairline flex justify-between items-start gap-3">
          <div className="min-w-0">
            <div className="text-[11px] tracking-[1.5px] font-bold text-soft mb-[6px]">{eyebrow}</div>
            <h3 className="m-0 text-[20px] sm:text-[22px] lg:text-[24px] font-semibold tracking-[-0.015em] truncate">{title}</h3>
          </div>
          <button onClick={onClose} className="w-9 h-9 rounded-full border border-hairline bg-white cursor-pointer shrink-0 text-lg">×</button>
        </div>
        <div className="p-5 sm:p-6 lg:p-7 flex-1 overflow-y-auto">{children}</div>
        <div className="p-4 sm:p-5 border-t border-hairline flex gap-[10px] justify-end" style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}>{footer}</div>
      </div>
    </>
  );
}

function StudentSearchRow({
  s,
  selected,
  onToggle,
}: {
  s: StudentSearch;
  selected: boolean;
  onToggle: () => void;
}) {
  const disabled = s.has_group && !selected;
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className="flex items-center gap-3 p-3 rounded-[10px] text-left w-full transition-colors disabled:cursor-not-allowed disabled:opacity-50"
      style={{
        background: selected ? "var(--sage)" : "#fff",
        boxShadow: `inset 0 0 0 ${selected ? 2 : 1}px ${selected ? "var(--brand-accent)" : "var(--hairline)"}`,
      }}
    >
      <div className="w-9 h-9 rounded-full flex items-center justify-center text-[12px] font-bold text-white shrink-0"
        style={{ background: "var(--forest)" }}>
        {(s.full_name || "?").split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase()}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[13px] font-semibold text-ink truncate">{s.full_name}</div>
        <div className="text-[11px] text-soft truncate">{s.email} · RA {s.ra}</div>
        {s.has_group && !selected && (
          <div className="text-[10px] text-soft mt-[2px]">já está em outro grupo</div>
        )}
      </div>
      <div className="text-[18px]" style={{ color: selected ? "var(--brand-accent)" : "var(--soft)" }}>
        {selected ? "✓" : "+"}
      </div>
    </button>
  );
}

function useStudentSearch(q: string) {
  return useQuery({
    queryKey: ["students-search", q],
    queryFn: () => groupsApi.searchStudents(q),
    staleTime: 30_000,
  });
}

function CreateGroupPanel({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<StudentSearch[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [debounced, setDebounced] = useState("");

  useEffect(() => {
    const t = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(t);
  }, [search]);

  const { data: results = [], isFetching } = useStudentSearch(debounced);

  const selectedIds = useMemo(() => new Set(selected.map((s) => s.id)), [selected]);
  const minOk = selected.length >= 4;
  const canSubmit = name.trim().length >= 2 && minOk && !busy;

  function toggle(s: StudentSearch) {
    setErr(null);
    setSelected((cur) =>
      cur.some((x) => x.id === s.id) ? cur.filter((x) => x.id !== s.id) : [...cur, s]
    );
  }

  async function submit() {
    setErr(null);
    setBusy(true);
    try {
      await groupsApi.create({
        name: name.trim(),
        member_ids: selected.map((s) => s.id),
      });
      await qc.invalidateQueries({ queryKey: ["groups-admin"] });
      onClose();
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <PanelShell
      eyebrow="NOVO GRUPO"
      title="Criar grupo"
      onClose={onClose}
      footer={
        <>
          <Btn kind="ghost" onClick={onClose}>Cancelar</Btn>
          <Btn kind="primary" disabled={!canSubmit} onClick={submit}>
            {busy ? "Criando…" : "Criar Grupo"}
          </Btn>
        </>
      }
    >
      <Field label="Nome do grupo">
        <Input value={name} onChange={setName} placeholder="Coletivo Verde" focused={!name} />
      </Field>

      <Field label="Adicionar alunos cadastrados" hint={isFetching ? "buscando…" : undefined}>
        <Input
          value={search}
          onChange={setSearch}
          placeholder="Buscar por nome, e-mail ou RA…"
          right={
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--soft)" strokeWidth="2">
              <circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" />
            </svg>
          }
        />
      </Field>

      {results.length > 0 && (
        <div className="flex flex-col gap-2 mb-5 max-h-[260px] overflow-y-auto pr-1">
          {results.map((s) => (
            <StudentSearchRow
              key={s.id}
              s={s}
              selected={selectedIds.has(s.id)}
              onToggle={() => toggle(s)}
            />
          ))}
        </div>
      )}

      <div className="mt-5">
        <div className="flex justify-between mb-[10px]">
          <span className="text-[12px] font-semibold">Integrantes selecionados</span>
          <span className="text-[12px] text-soft font-mono">{selected.length} / 4 mínimo</span>
        </div>
        <div className="flex flex-col gap-2">
          {selected.length === 0 && (
            <div className="flex items-center gap-3 p-3 rounded-[10px] border border-dashed border-hairline text-soft text-[12px]">
              Nenhum aluno adicionado ainda.
            </div>
          )}
          {selected.map((s) => (
            <div key={s.id} className="flex items-center gap-3 p-3 rounded-[10px] bg-cream">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white"
                style={{ background: "var(--forest)" }}>
                {(s.full_name || "?").split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold text-ink truncate">{s.full_name}</div>
                <div className="text-[11px] text-soft truncate">RA {s.ra}</div>
              </div>
              <button
                type="button"
                onClick={() => toggle(s)}
                className="text-[14px] text-soft hover:text-danger cursor-pointer"
                aria-label="Remover"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-6">
        <div className="flex justify-between mb-2 text-[12px]">
          <span className="font-semibold">{selected.length} de 4 integrantes mínimos</span>
          <span className="text-soft font-mono">{Math.min(100, Math.round((selected.length / 4) * 100))}%</span>
        </div>
        <div className="h-2 rounded-full bg-cream">
          <div
            className="h-full rounded-full transition-[width]"
            style={{
              width: `${Math.min(100, (selected.length / 4) * 100)}%`,
              background: "linear-gradient(90deg, var(--brand-accent), var(--green))",
            }}
          />
        </div>
      </div>

      {err && (
        <div className="text-[12px] mt-4 px-3 py-2 rounded-lg" style={{ background: "var(--danger-bg)", color: "#7A2929" }}>
          {err}
        </div>
      )}
    </PanelShell>
  );
}

/* ============ Painel: detalhes / gerenciar membros ============ */

function GroupDetailPanel({ groupId, onClose }: { groupId: string; onClose: () => void }) {
  const qc = useQueryClient();
  const detail = useQuery({
    queryKey: ["group-detail", groupId],
    queryFn: () => groupsApi.detail(groupId),
  });
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(search), 250);
    return () => clearTimeout(t);
  }, [search]);

  const { data: results = [] } = useStudentSearch(debounced);
  const memberIds = useMemo(
    () => new Set((detail.data?.members ?? []).map((m) => m.user_id)),
    [detail.data]
  );

  async function refreshAll() {
    await Promise.all([
      qc.invalidateQueries({ queryKey: ["group-detail", groupId] }),
      qc.invalidateQueries({ queryKey: ["groups-admin"] }),
      qc.invalidateQueries({ queryKey: ["students-search"] }),
    ]);
  }

  async function add(userId: string) {
    setErr(null);
    setBusy(userId);
    try {
      await groupsApi.addMember(groupId, userId);
      await refreshAll();
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function remove(userId: string) {
    setErr(null);
    setBusy(userId);
    try {
      await groupsApi.removeMember(groupId, userId);
      await refreshAll();
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : (e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  const g = detail.data;

  return (
    <PanelShell
      eyebrow="GRUPO"
      title={g?.name ?? "Carregando…"}
      onClose={onClose}
      footer={<Btn kind="ghost" onClick={onClose}>Fechar</Btn>}
    >
      {detail.isLoading && <div className="text-soft text-[13px]">Carregando…</div>}
      {detail.error && (
        <div className="text-[12px] px-3 py-2 rounded-lg" style={{ background: "var(--danger-bg)", color: "#7A2929" }}>
          {(detail.error as Error).message}
        </div>
      )}
      {g && (
        <>
          <div className="grid grid-cols-2 gap-3 mb-6">
            <Card style={{ padding: 16 }}>
              <div className="text-[10px] tracking-[1.5px] font-bold text-soft">CRIADO EM</div>
              <div className="text-[15px] font-semibold mt-1">
                {new Date(g.created_at).toLocaleDateString("pt-BR")}
              </div>
            </Card>
            <Card style={{ padding: 16 }}>
              <div className="text-[10px] tracking-[1.5px] font-bold text-soft">ARRECADADO</div>
              <div className="flex items-baseline gap-1 mt-1">
                <Digits size={20}>{g.kg}</Digits>
                <span className="text-[12px] text-soft">kg</span>
              </div>
            </Card>
          </div>

          <div className="flex justify-between items-baseline mb-2">
            <div className="text-[12px] font-semibold">
              Integrantes <span className="text-soft font-mono">{g.members.length}</span>
            </div>
            {g.members.length >= 4
              ? <Chip tone="success">grupo completo</Chip>
              : <Chip tone="warning">faltam {4 - g.members.length}</Chip>}
          </div>
          <div className="flex flex-col gap-2 mb-6">
            {g.members.map((m) => (
              <div key={m.user_id} className="flex items-center gap-3 p-3 rounded-[10px] bg-cream">
                <div className="w-9 h-9 rounded-full flex items-center justify-center text-[11px] font-bold text-white shrink-0"
                  style={{ background: m.is_leader ? "var(--brand-accent)" : "var(--forest)" }}>
                  {(m.full_name || "?").split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-[13px] font-semibold text-ink truncate">{m.full_name}</span>
                    {m.is_leader && <Chip tone="gold">★ líder</Chip>}
                  </div>
                  <div className="text-[11px] text-soft truncate">{m.email} · RA {m.ra}</div>
                </div>
                {!m.is_leader && (
                  <button
                    type="button"
                    onClick={() => remove(m.user_id)}
                    disabled={busy === m.user_id}
                    className="text-[12px] font-semibold text-danger hover:opacity-80 cursor-pointer disabled:opacity-50"
                  >
                    {busy === m.user_id ? "…" : "Remover"}
                  </button>
                )}
              </div>
            ))}
          </div>

          <div className="text-[12px] font-semibold mb-2">Adicionar aluno</div>
          <Input
            value={search}
            onChange={setSearch}
            placeholder="Buscar por nome, e-mail ou RA…"
            right={
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--soft)" strokeWidth="2">
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" />
              </svg>
            }
          />
          {results.length > 0 && (
            <div className="flex flex-col gap-2 mt-3 max-h-[220px] overflow-y-auto pr-1">
              {results
                .filter((s) => !memberIds.has(s.id))
                .map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => add(s.id)}
                    disabled={s.has_group || busy === s.id}
                    className="flex items-center gap-3 p-3 rounded-[10px] text-left w-full transition-colors disabled:cursor-not-allowed disabled:opacity-50"
                    style={{
                      background: "#fff",
                      boxShadow: "inset 0 0 0 1px var(--hairline)",
                    }}
                  >
                    <div className="w-8 h-8 rounded-full flex items-center justify-center text-[11px] font-bold text-white"
                      style={{ background: "var(--forest)" }}>
                      {(s.full_name || "?").split(" ").map((p) => p[0]).slice(0, 2).join("").toUpperCase()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[13px] font-semibold text-ink truncate">{s.full_name}</div>
                      <div className="text-[11px] text-soft truncate">
                        {s.has_group ? "já está em outro grupo" : `${s.email} · RA ${s.ra}`}
                      </div>
                    </div>
                    <div className="text-[18px] text-soft">
                      {busy === s.id ? "…" : "+"}
                    </div>
                  </button>
                ))}
            </div>
          )}

          {g.recent_evidences.length > 0 && (
            <div className="mt-6">
              <div className="text-[12px] font-semibold mb-2">Últimas evidências</div>
              <div className="flex flex-col gap-1">
                {g.recent_evidences.map((e) => (
                  <div key={e.id} className="flex justify-between text-[12px] py-[6px] border-b border-hairline last:border-b-0">
                    <span className="capitalize">{e.category}</span>
                    <span className="text-soft font-mono">
                      {new Date(e.detected_at).toLocaleString("pt-BR")}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {err && (
            <div className="text-[12px] mt-4 px-3 py-2 rounded-lg" style={{ background: "var(--danger-bg)", color: "#7A2929" }}>
              {err}
            </div>
          )}
        </>
      )}
    </PanelShell>
  );
}
