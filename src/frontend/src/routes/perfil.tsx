import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { RequireAuth } from "@/components/require-auth";
import { Btn, Card, Chip, Field, Input, PageHeader } from "@/components/le-ui";
import { useAuth } from "@/lib/auth-context";
import { useMyGroup, useGroupsRanking } from "@/lib/queries";

export const Route = createFileRoute("/perfil")({
  component: () => <RequireAuth><Perfil /></RequireAuth>,
});

function StrengthMeter({ score = 3 }: { score?: number }) {
  const colors = ["var(--danger)", "#E08A2A", "var(--gold)", "var(--success)"];
  const labels = ["Fraca", "Média", "Boa", "Forte"];
  return (
    <div className="flex gap-1 mt-2 items-center">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="flex-1 h-1 rounded-sm"
          style={{ background: i < score ? colors[score - 1] : "var(--hairline)" }} />
      ))}
      <span className="ml-2 text-[11px] font-semibold" style={{ color: colors[score - 1] }}>{labels[score - 1]}</span>
    </div>
  );
}

function Perfil() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const myGroup = useMyGroup();
  const ranking = useGroupsRanking();
  const [showPwd, setShowPwd] = useState(false);
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [conf, setConf] = useState("");

  if (!user) return null;
  const initials = user.full_name.split(" ").map((s) => s[0]).slice(0, 2).join("").toUpperCase();
  const memberSince = "Membro da plataforma";
  const myGroupName = myGroup.data?.group?.name;
  const myKg = myGroup.data?.kg ?? 0;
  const myRank = myGroup.data ? (ranking.data ?? []).findIndex((g) => g.id === myGroup.data?.group?.id) + 1 : 0;

  const score = Math.min(4, Math.max(1,
    (next.length >= 8 ? 1 : 0) + (/[A-Z]/.test(next) ? 1 : 0) + (/[0-9]/.test(next) ? 1 : 0) + (/[^A-Za-z0-9]/.test(next) ? 1 : 0),
  )) as 1 | 2 | 3 | 4;

  return (
    <div className="h-screen overflow-auto">
      <PageHeader title="Perfil" subtitle="Suas informações de cadastro e segurança" />
      <div className="p-8 flex justify-center">
        <div className="max-w-[640px] w-full flex flex-col gap-5">
          {/* Profile */}
          <Card style={{ padding: 0 }}>
            <div className="relative px-8 pt-7 pb-5 text-white" style={{ background: "var(--forest)" }}>
              <svg className="absolute opacity-[.04]" style={{ right: -10, top: -10 }} width="180" height="180" viewBox="0 0 200 200">
                <circle cx="100" cy="100" r="96" fill="none" stroke="#fff" strokeWidth="2" />
              </svg>
              <div className="flex items-center gap-5 relative">
                <div className="w-[88px] h-[88px] rounded-full flex items-center justify-center text-[32px] font-bold border-[3px] border-white/15"
                  style={{ background: "var(--brand-accent)" }}>{initials}</div>
                <div>
                  <div className="flex items-center gap-[10px]">
                    <h2 className="m-0 text-[28px] font-semibold tracking-[-0.015em]">{user.full_name}</h2>
                    <Chip tone="sage">{user.role === "professor" ? "Professor" : "Aluno"}</Chip>
                  </div>
                  <div className="text-[13px] text-white/70 mt-1">
                    {user.email} {user.ra && <>· RA {user.ra}</>}
                  </div>
                  <div className="text-[12px] text-white/50 mt-[6px]">{memberSince}</div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-3 border-t border-hairline">
              {([
                ["CURSO", user.course || "—"],
                ["SEMESTRE", user.semester ? `${user.semester}º` : "—"],
                ["PERÍODO", user.period === "matutino" ? "Matutino" : "Noturno"],
              ] as const).map(([k, v]) => (
                <div key={k} className="px-6 py-5 border-r border-hairline last:border-r-0">
                  <div className="text-[10px] tracking-[1.5px] font-bold text-soft">{k}</div>
                  <div className="text-base font-semibold mt-[6px] text-ink">{v}</div>
                </div>
              ))}
            </div>
            <div className="px-6 py-4 flex justify-between items-center bg-cream">
              <div className="text-[13px] text-body">
                {myGroupName ? (
                  <>Grupo: <strong>{myGroupName}</strong> <span className="text-soft">· #{myRank || "—"} no ranking · {myKg} kg</span></>
                ) : (
                  <span className="text-soft">Sem grupo associado</span>
                )}
              </div>
              <Btn kind="ghost" sm>Editar perfil</Btn>
            </div>
          </Card>

          {/* Change password */}
          <Card style={{ padding: 28 }}>
            <button type="button" onClick={() => setShowPwd((v) => !v)} className="w-full flex justify-between items-center cursor-pointer text-left bg-transparent border-0">
              <div>
                <h3 className="m-0 text-[18px] font-semibold tracking-[-0.01em]">Alterar senha</h3>
                <div className="text-[13px] text-soft mt-1">Use uma senha forte que você não usa em outros lugares</div>
              </div>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--soft)" strokeWidth="2"
                style={{ transform: showPwd ? "rotate(180deg)" : "rotate(0deg)", transition: "transform .2s" }}>
                <polyline points="6 9 12 15 18 9" />
              </svg>
            </button>
            {showPwd && (
              <div className="mt-5 pt-5 border-t border-hairline">
                <Field label="Senha atual"><Input value={cur} onChange={setCur} type="password" placeholder="••••••••" /></Field>
                <Field label="Nova senha">
                  <Input value={next} onChange={setNext} type="password" placeholder="••••••••" focused />
                  {next && <StrengthMeter score={score} />}
                </Field>
                <Field label="Confirmar nova senha">
                  <Input value={conf} onChange={setConf} type="password" placeholder="••••••••"
                    right={conf && conf === next ? (
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
                    ) : null}
                  />
                </Field>
                <div className="flex justify-end mt-3">
                  <Btn kind="primaryDark">Salvar nova senha</Btn>
                </div>
              </div>
            )}
          </Card>

          {/* Danger zone */}
          <Card style={{ padding: 24, border: "1px solid var(--danger)", background: "#FCF7F7" }}>
            <div className="flex justify-between items-center">
              <div>
                <h3 className="m-0 text-[16px] font-semibold" style={{ color: "var(--danger)" }}>Sair da conta</h3>
                <div className="text-[13px] text-body mt-1">Você será desconectado desta sessão.</div>
              </div>
              <Btn kind="danger" onClick={() => { logout(); navigate({ to: "/login" }); }}>Logout</Btn>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
