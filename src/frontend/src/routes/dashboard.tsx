import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { RequireAuth } from "@/components/require-auth";
import { Btn, Card, Chip, Digits, PageHeader, Caret } from "@/components/le-ui";
import { BrandLogo } from "@/components/brand";
import { useEvidencesAggregate, useGroupsRanking, useTimeSeries, useMyGroup } from "@/lib/queries";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, Legend as RLegend } from "recharts";
import { useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/dashboard")({
  component: () => <RequireAuth><Dashboard /></RequireAuth>,
});

type PeriodKey = "7" | "30" | "60" | "all";
const PERIOD_OPTIONS: { value: PeriodKey; label: string; days: number | null }[] = [
  { value: "7", label: "Últimos 7 dias", days: 7 },
  { value: "30", label: "Últimos 30 dias", days: 30 },
  { value: "60", label: "Últimos 60 dias", days: 60 },
  { value: "all", label: "Todo período", days: null },
];

const COURSE_OPTIONS = [
  { value: "", label: "Todos os cursos" },
  { value: "Administração", label: "Administração" },
  { value: "Ciências Contábeis", label: "Ciências Contábeis" },
  { value: "Ciências Econômicas", label: "Ciências Econômicas" },
] as const;

function FilterChip<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const display = options.find((o) => o.value === value)?.label ?? "—";
  return (
    <label className="relative flex items-center gap-2 px-[14px] py-2 bg-white rounded-full border border-hairline text-[13px] cursor-pointer hover:border-brand-accent/60">
      <span className="text-soft text-[11px] font-semibold">{label}</span>
      <span className="font-semibold">{display}</span>
      <Caret />
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="absolute inset-0 opacity-0 cursor-pointer"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

function Legend({ color, label, value }: { color: string; label: string; value?: string }) {
  return (
    <div className="flex items-center gap-2 text-[12px]">
      <span className="w-[10px] h-[10px] rounded-[3px]" style={{ background: color }} />
      <span className="font-medium text-body">{label}</span>
      {value && <span className="text-soft font-mono">{value}</span>}
    </div>
  );
}

function CategoryGlyph({ tone }: { tone: "arroz" | "feijao" | "macarrao" }) {
  const c = { arroz: "var(--arroz)", feijao: "var(--feijao)", macarrao: "var(--macarrao)" }[tone];
  const bg = { arroz: "var(--arroz-light)", feijao: "var(--feijao-light)", macarrao: "var(--macarrao-light)" }[tone];
  return (
    <div className="w-9 h-9 rounded-[10px] flex items-center justify-center" style={{ background: bg }}>
      {tone === "arroz" && <svg width="18" height="18" viewBox="0 0 24 24" fill={c}><ellipse cx="8" cy="10" rx="2" ry="4" /><ellipse cx="14" cy="13" rx="2" ry="4" transform="rotate(20 14 13)" /><ellipse cx="11" cy="17" rx="2" ry="4" transform="rotate(-15 11 17)" /></svg>}
      {tone === "feijao" && <svg width="18" height="18" viewBox="0 0 24 24" fill={c}><path d="M7 6c-3 1-4 5-2 8s5 4 8 2c2-1 3-3 2-5-1-2-2-1-4-3-1-1-2-3-4-2z" /></svg>}
      {tone === "macarrao" && <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={c} strokeWidth="2.5" strokeLinecap="round"><path d="M5 6c2 0 2 12 4 12M10 6c2 0 2 12 4 12M15 6c2 0 2 12 4 12" /></svg>}
    </div>
  );
}

function StatCard({ title, subtitle, value, unit, delta, featured, tone = "green" }: {
  title: string; subtitle: string; value: string | number; unit: string; delta?: string; featured?: boolean;
  tone?: "green" | "arroz" | "feijao" | "macarrao";
}) {
  const accents = { green: "var(--brand-accent)", arroz: "var(--arroz)", feijao: "var(--feijao)", macarrao: "var(--macarrao)" };
  return (
    <Card
      accent={accents[tone]}
      className="p-4 sm:p-5"
      style={{ background: featured ? "var(--forest)" : "#fff", color: featured ? "#fff" : "var(--body-color)" }}
    >
      <div className="flex justify-between items-start mb-3 sm:mb-[18px] gap-2">
        <div className="min-w-0">
          <div className="text-[11px] sm:text-[12px] font-bold tracking-[1px] uppercase truncate" style={{ color: featured ? "rgba(255,255,255,0.7)" : "var(--soft)" }}>{title}</div>
          <div className="text-[11px] mt-1 truncate" style={{ color: featured ? "rgba(255,255,255,0.5)" : "var(--soft)" }}>{subtitle}</div>
        </div>
        {featured ? <BrandLogo size={32} /> : tone !== "green" && <CategoryGlyph tone={tone as "arroz" | "feijao" | "macarrao"} />}
      </div>
      <div className="flex items-baseline gap-[6px]">
        <span
          className="font-mono tabular tracking-tight text-[32px] sm:text-[38px] lg:text-[42px] font-bold leading-none"
          style={{ color: featured ? "#fff" : "var(--ink)", letterSpacing: "-0.03em" }}
        >{value}</span>
        <span className="text-sm font-medium" style={{ color: featured ? "rgba(255,255,255,0.7)" : "var(--soft)" }}>{unit}</span>
      </div>
      {delta && (
        <div className="flex items-center gap-[6px] mt-[10px] text-[12px] font-semibold"
          style={{ color: featured ? "rgba(255,255,255,0.85)" : "var(--success)" }}>
          <span>↗</span> {delta}
        </div>
      )}
    </Card>
  );
}

function Dashboard() {
  const { user } = useAuth();
  const isProfessor = user?.role === "professor";

  const [period, setPeriod] = useState<PeriodKey>("60");
  const [groupId, setGroupId] = useState<string>("");
  const [course, setCourse] = useState<string>("");

  const days = PERIOD_OPTIONS.find((p) => p.value === period)?.days ?? null;
  const since = useMemo(
    () => (days ? new Date(Date.now() - days * 86400000).toISOString() : undefined),
    [days]
  );

  const myGroup = useMyGroup();
  const myGroupId = myGroup.data?.group?.id;

  // Aluno: queries são forçadas pelo backend pro grupo dele; Grupo/Curso ficam ocultos
  const effectiveGroupId = isProfessor && groupId ? groupId : undefined;
  const effectiveCourse = isProfessor && course ? course : undefined;

  const agg = useEvidencesAggregate({ groupId: effectiveGroupId, since });
  const ranking = useGroupsRanking({ course: effectiveCourse, since });
  const series = useTimeSeries(days ?? 365, effectiveGroupId);

  const total = agg.data?.total ?? 0;
  const c = agg.data?.counts ?? { arroz: 0, feijao: 0, macarrao: 0, acucar: 0, oleo: 0, fuba: 0 };

  const groupOptions = useMemo(
    () => [
      { value: "", label: "Todos os grupos" },
      ...(ranking.data ?? []).map((g) => ({ value: g.id, label: g.name })),
    ],
    [ranking.data]
  );

  const isFiltering =
    period !== "60" || (isProfessor && (groupId !== "" || course !== ""));
  function resetFilters() {
    setPeriod("60");
    setGroupId("");
    setCourse("");
  }

  return (
    <div className="h-full overflow-auto pb-24 lg:pb-0">
      <PageHeader
        title="Dashboard"
        subtitle={`Visão geral da campanha 2026.1 · ${agg.isLoading ? "carregando…" : "atualizada agora"}`}
        right={
          <>
            <Chip tone="dark">● ao vivo</Chip>
            <Btn kind="ghost" sm>Exportar</Btn>
          </>
        }
      />

      <div className="px-4 sm:px-6 lg:px-8 py-4 lg:py-5 flex gap-2 sm:gap-3 items-center border-b border-hairline bg-cream overflow-x-auto flex-nowrap lg:flex-wrap">
        <FilterChip
          label="PERÍODO"
          value={period}
          options={PERIOD_OPTIONS.map(({ value, label }) => ({ value, label }))}
          onChange={setPeriod}
        />
        {isProfessor && (
          <>
            <FilterChip
              label="GRUPO"
              value={groupId}
              options={groupOptions}
              onChange={setGroupId}
            />
            <FilterChip
              label="CURSO"
              value={course}
              options={[...COURSE_OPTIONS]}
              onChange={setCourse}
            />
          </>
        )}
        <Btn
          kind={isFiltering ? "outline" : "ghost"}
          sm
          className="lg:ml-auto shrink-0"
          onClick={resetFilters}
          disabled={!isFiltering}
        >
          Limpar
        </Btn>
      </div>

      <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-4 lg:gap-6">
        {/* Stat cards */}
        <div className="grid gap-3 lg:gap-4 grid-cols-1 sm:grid-cols-2 xl:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <StatCard title="Total Geral" subtitle="Arrecadação total da campanha" value={total} unit="kg" delta={days ? `últimos ${days} dias` : "todo período"} featured />
          <StatCard title="Arroz" subtitle={`${c.arroz} registros`} value={c.arroz} unit="kg" tone="arroz" />
          <StatCard title="Feijão" subtitle={`${c.feijao} registros`} value={c.feijao} unit="kg" tone="feijao" />
          <StatCard title="Macarrão" subtitle={`${c.macarrao} registros`} value={c.macarrao} unit="kg" tone="macarrao" />
        </div>

        {/* Two columns */}
        <div className="grid gap-4 lg:gap-5 grid-cols-1 xl:grid-cols-[1.5fr_1fr]">
          <Card className="p-5 sm:p-6 lg:p-7">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h3 className="m-0 text-[18px] font-semibold tracking-[-0.01em]">Arrecadação ao longo do tempo</h3>
                <div className="text-[12px] text-soft mt-1">
                  Quilos acumulados por categoria · {PERIOD_OPTIONS.find((p) => p.value === period)?.label.toLowerCase()}
                </div>
              </div>
            </div>
            <div style={{ width: "100%", height: 220 }} className="sm:!h-[240px]">
              <ResponsiveContainer>
                <AreaChart data={series.data ?? []} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid stroke="var(--hairline)" vertical={false} />
                  <XAxis dataKey="label" stroke="var(--soft)" fontSize={10} />
                  <YAxis stroke="var(--soft)" fontSize={10} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid var(--hairline)" }} />
                  <Area type="monotone" dataKey="arroz" stroke="var(--arroz)" fill="var(--arroz)" fillOpacity={0.18} strokeWidth={2.5} />
                  <Area type="monotone" dataKey="feijao" stroke="var(--feijao)" fill="var(--feijao)" fillOpacity={0.18} strokeWidth={2.5} />
                  <Area type="monotone" dataKey="macarrao" stroke="var(--macarrao)" fill="var(--macarrao)" fillOpacity={0.18} strokeWidth={2.5} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-3 sm:gap-6 mt-4">
              <Legend color="var(--arroz)" label="Arroz" value={`${c.arroz} kg`} />
              <Legend color="var(--feijao)" label="Feijão" value={`${c.feijao} kg`} />
              <Legend color="var(--macarrao)" label="Macarrão" value={`${c.macarrao} kg`} />
            </div>
          </Card>

          <Card style={{ padding: 0 }}>
            <div className="px-4 sm:px-6 pt-5 sm:pt-6 pb-4 border-b border-hairline">
              <div className="flex justify-between items-center gap-2">
                <h3 className="m-0 text-[16px] sm:text-[18px] font-semibold tracking-[-0.01em]">Ranking de Grupos</h3>
                <span className="text-[12px] text-soft shrink-0">{ranking.data?.length ?? 0} grupos</span>
              </div>
            </div>
            <div>
              {(ranking.data ?? []).slice(0, 5).map((r, idx) => {
                const rank = idx + 1;
                const medal = rank === 1 ? "🥇" : rank === 2 ? "🥈" : rank === 3 ? "🥉" : "";
                const highlight = r.id === myGroupId;
                return (
                  <div key={r.id} className="flex items-center gap-3 sm:gap-[14px] px-4 sm:px-6 py-3 sm:py-[14px] border-b border-hairline"
                    style={{ background: highlight ? "var(--sage)" : "transparent" }}>
                    <div className="font-mono font-bold text-[13px] w-7 h-7 rounded-lg flex items-center justify-center"
                      style={{
                        background: rank <= 3 ? "var(--forest)" : "transparent",
                        color: rank <= 3 ? "#fff" : "var(--soft)",
                        border: rank > 3 ? "1px solid var(--hairline)" : undefined,
                      }}>{rank}</div>
                    <div className="flex-1 min-w-0 text-sm font-semibold truncate">
                      {r.name}
                      {highlight && <span className="ml-2 text-[10px] font-bold tracking-[1px]" style={{ color: "var(--green)" }}>VOCÊ</span>}
                    </div>
                    <Digits size={15} color="var(--ink)">{r.kg}</Digits>
                    <span className="text-[12px] text-soft">kg</span>
                    <span className="text-base w-[22px] hidden sm:inline">{medal}</span>
                  </div>
                );
              })}
              {(!ranking.data || ranking.data.length === 0) && (
                <div className="px-6 py-8 text-center text-soft text-sm">
                  {ranking.isLoading ? "Carregando…" : "Sem grupos para os filtros atuais."}
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Bottom — grouped bars */}
        <Card className="p-5 sm:p-6 lg:p-7">
          <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 mb-4 sm:mb-6">
            <div>
              <h3 className="m-0 text-[16px] sm:text-[18px] font-semibold tracking-[-0.01em]">Distribuição por Categoria — por Grupo</h3>
              <div className="text-[12px] text-soft mt-1">Quilos coletados por cada grupo</div>
            </div>
            <div className="flex gap-4">
              <Legend color="var(--brand-accent)" label="Total kg" />
            </div>
          </div>
          <GroupedBars data={ranking.data ?? []} />
        </Card>
        {user && <div className="text-[12px] text-soft text-center pb-2">Sessão de {user.full_name}</div>}
      </div>
    </div>
  );
}

function GroupedBars({ data }: { data: Array<{ id: string; name: string; kg: number }> }) {
  const top = data.slice(0, 6).map((g) => ({ name: g.name, total: g.kg }));
  return (
    <div style={{ width: "100%", height: 240 }} className="sm:!h-[280px]">
      <ResponsiveContainer>
        <BarChart data={top} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="var(--hairline)" vertical={false} />
          <XAxis dataKey="name" stroke="var(--soft)" fontSize={11} interval={0} angle={-15} textAnchor="end" height={50} />
          <YAxis stroke="var(--soft)" fontSize={10} />
          <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid var(--hairline)" }} />
          <RLegend />
          <Bar dataKey="total" name="Total kg" fill="var(--brand-accent)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
