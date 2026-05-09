import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { z } from "zod";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";
import { Btn, Field, Input, Caret } from "@/components/le-ui";
import { BrandLogo, FecapMark } from "@/components/brand";
import { useAuth } from "@/lib/auth-context";
import type { SignupInput, Course, Period } from "@/lib/auth-api";

export const Route = createFileRoute("/login")({
  component: LoginScreen,
});

function LoginScreen() {
  const [tab, setTab] = useState<"login" | "cadastro">("login");
  const { user, isLoading: loading } = useAuth();
  const navigate = useNavigate();
  useEffect(() => {
    if (!loading && user) navigate({ to: "/dashboard" });
  }, [user, loading, navigate]);

  return (
    <div className="flex w-full min-h-screen bg-white font-sans">
      {/* LEFT */}
      <div
        className="relative flex flex-col justify-between p-12 text-white overflow-hidden"
        style={{ width: "40%", background: "var(--forest)" }}
      >
        <svg className="absolute inset-0 opacity-[.06]" width="100%" height="100%">
          <defs>
            <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
              <circle cx="2" cy="2" r="1" fill="#fff" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#dots)" />
        </svg>
        <div className="flex items-center gap-3 relative">
          <FecapMark color="#fff" height={18} />
          <span className="w-px h-4 bg-white/30" />
          <span className="text-[12px] tracking-[1.5px] text-white/70">2026</span>
        </div>
        <div className="flex flex-col items-center gap-6 relative">
          <BrandLogo size={220} />
          <div className="text-center max-w-[360px]">
            <div className="text-[11px] tracking-[2px] font-bold text-white/55 mb-3">SISTEMA DE</div>
            <h1 className="m-0 text-[38px] font-semibold leading-[1.1] tracking-[-0.02em]">
              Arrecadação<br />de Alimentos
            </h1>
            <p className="mt-4 text-[15px] text-white/70 leading-[1.5]">
              Liderança coletiva movida a empatia. Cada quilo registrado é um aluno, um professor, um grupo somando.
            </p>
          </div>
        </div>
        <div className="flex justify-between items-center text-[12px] text-white/50 relative">
          <span>Campanha 2026.1</span>
          <span>v3.2</span>
        </div>
      </div>

      {/* RIGHT */}
      <div
        className="flex flex-col justify-center"
        style={{ width: "60%", background: "var(--cream)", padding: "64px 80px" }}
      >
        <div className="max-w-[480px] w-full mx-auto">
          <div className="mb-8">
            <div className="text-[12px] tracking-[1.5px] font-bold text-soft mb-2">BEM-VINDO DE VOLTA</div>
            <h2 className="m-0 text-[40px] font-semibold leading-[1.05] tracking-[-0.025em] text-ink">
              {tab === "login" ? "Entrar na sua conta" : "Criar sua conta"}
            </h2>
          </div>

          <div
            className="inline-flex p-1 rounded-full mb-8 bg-white border border-hairline"
          >
            {(["login", "cadastro"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="px-[22px] py-2 text-[13px] font-semibold rounded-full transition-colors"
                style={{
                  background: tab === t ? "var(--forest)" : "transparent",
                  color: tab === t ? "#fff" : "var(--body-color)",
                }}
              >
                {t === "login" ? "Entrar" : "Cadastrar"}
              </button>
            ))}
          </div>

          {tab === "login" ? <LoginForm /> : <CadastroForm onDone={() => setTab("login")} />}
        </div>
      </div>
    </div>
  );
}

function LoginForm() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await login(email, password);
      navigate({ to: "/dashboard" });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit}>
      <Field label="E-mail institucional">
        <Input value={email} onChange={setEmail} placeholder="seu.nome@edu.fecap.br" autoComplete="email" focused={!email} />
      </Field>
      <Field label="Senha">
        <Input
          value={password} onChange={setPassword} type="password" autoComplete="current-password"
          placeholder="••••••••"
          right={
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--soft)" strokeWidth="2">
              <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z" /><circle cx="12" cy="12" r="3" />
            </svg>
          }
        />
      </Field>
      {err && (
        <div className="text-[12px] mb-3 px-3 py-2 rounded-lg" style={{ background: "var(--danger-bg)", color: "#7A2929" }}>
          {err}
        </div>
      )}
      <Btn kind="primaryDark" full type="submit" disabled={busy} style={{ marginTop: 12, height: 48, fontSize: 15 }}>
        {busy ? "Entrando..." : "Entrar →"}
      </Btn>
      <div className="text-center mt-[18px]">
        <a className="text-[13px] font-medium text-green cursor-pointer" style={{ color: "var(--green)" }}>
          Esqueci minha senha
        </a>
      </div>
      <div
        className="mt-7 p-4 rounded-xl text-[12px] leading-[1.5]"
        style={{ background: "var(--sage)", color: "var(--forest)" }}
      >
        <strong className="block mb-1">Acesso restrito FECAP</strong>
        Use seu e-mail @edu.fecap.br para entrar. Estudantes e professores cadastrados pela coordenação.
      </div>
    </form>
  );
}

function StrengthMeter({ score = 3 }: { score?: number }) {
  const colors = ["var(--danger)", "#E08A2A", "var(--gold)", "var(--success)"];
  const labels = ["Fraca", "Média", "Boa", "Forte"];
  return (
    <div className="flex gap-1 mt-2 items-center">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="flex-1 h-1 rounded-sm"
          style={{ background: i < score ? colors[score - 1] : "var(--hairline)" }} />
      ))}
      <span className="ml-2 text-[11px] font-semibold" style={{ color: colors[score - 1] }}>
        {labels[score - 1]}
      </span>
    </div>
  );
}

const COURSES: Course[] = ["Administração", "Ciências Contábeis", "Ciências Econômicas"];

const baseSchema = {
  full_name: z.string().trim().min(3, "Informe seu nome completo").max(120),
  password: z.string().min(8, "Mínimo 8 caracteres").max(128),
  confirm: z.string(),
  period: z.enum(["matutino", "noturno"]),
};

const signupSchema = z
  .discriminatedUnion("role", [
    z.object({
      role: z.literal("aluno"),
      email: z
        .string()
        .trim()
        .toLowerCase()
        .email("E-mail inválido")
        .refine((v) => v.endsWith("@edu.fecap.br"), "Use o e-mail @edu.fecap.br"),
      ra: z.string().regex(/^\d{8}$/, "RA do aluno tem 8 dígitos"),
      course: z.enum(["Administração", "Ciências Contábeis", "Ciências Econômicas"]),
      semester: z.coerce.number().int().min(1).max(8),
      ...baseSchema,
    }),
    z.object({
      role: z.literal("professor"),
      email: z
        .string()
        .trim()
        .toLowerCase()
        .email("E-mail inválido")
        .refine((v) => v.endsWith("@fecap.br"), "Use o e-mail @fecap.br"),
      ra: z.string().regex(/^\d{6}$/, "RA do professor tem 6 dígitos"),
      ...baseSchema,
    }),
  ])
  .refine((d) => d.password === d.confirm, {
    message: "As senhas não conferem",
    path: ["confirm"],
  });

function raEntryHint(ra: string): string | null {
  if (ra.length < 4) return null;
  const yy = Number(ra.slice(0, 2));
  const mm = Number(ra.slice(2, 4));
  if (!yy || mm < 1 || mm > 12) return null;
  const date = new Date(2000 + yy, mm - 1, 1);
  return `Entrada em ${format(date, "MMMM 'de' yyyy", { locale: ptBR })}`;
}

function CadastroForm({ onDone }: { onDone: () => void }) {
  const { signup } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [f, setF] = useState({
    full_name: "",
    email: "",
    role: "aluno" as "aluno" | "professor",
    course: "Administração" as Course,
    semester: 3,
    period: "noturno" as Period,
    ra: "",
    password: "",
    confirm: "",
  });
  const set = <K extends keyof typeof f>(k: K, v: (typeof f)[K]) =>
    setF((s) => ({ ...s, [k]: v }));

  const score = Math.min(
    4,
    Math.max(
      1,
      (f.password.length >= 8 ? 1 : 0) +
        (/[A-Z]/.test(f.password) ? 1 : 0) +
        (/[0-9]/.test(f.password) ? 1 : 0) +
        (/[^A-Za-z0-9]/.test(f.password) ? 1 : 0),
    ),
  ) as 1 | 2 | 3 | 4;

  const isAluno = f.role === "aluno";
  const raMaxLen = isAluno ? 8 : 6;
  const raHint = useMemo(
    () => (isAluno ? raEntryHint(f.ra) : null),
    [isAluno, f.ra],
  );

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    const parsed = signupSchema.safeParse(f);
    if (!parsed.success) {
      setErr(parsed.error.issues[0]?.message ?? "Verifique os campos");
      return;
    }
    // Strip `confirm` and build payload conforme SignupInput
    const data = parsed.data;
    const payload: SignupInput =
      data.role === "aluno"
        ? {
            role: "aluno",
            email: data.email,
            ra: data.ra,
            full_name: data.full_name,
            password: data.password,
            period: data.period,
            course: data.course,
            semester: data.semester,
          }
        : {
            role: "professor",
            email: data.email,
            ra: data.ra,
            full_name: data.full_name,
            password: data.password,
            period: data.period,
          };

    setBusy(true);
    try {
      await signup(payload);
      navigate({ to: "/dashboard" });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="max-h-[540px] overflow-y-auto pr-1">
      <Field label="Nome completo">
        <Input value={f.full_name} onChange={(v) => set("full_name", v)} placeholder="Lucas Mendes da Silva" />
      </Field>
      <Field label="E-mail institucional" hint={isAluno ? "@edu.fecap.br" : "@fecap.br"}>
        <Input value={f.email} onChange={(v) => set("email", v)} placeholder={isAluno ? "seu.nome@edu.fecap.br" : "seu.nome@fecap.br"} focused />
      </Field>
      <div className="mb-[18px]">
        <label className="text-[12px] font-semibold mb-2 block" style={{ color: "rgb(76, 81, 75)" }}>Cargo</label>
        <div className="flex gap-[10px]">
          {(["aluno", "professor"] as const).map((k) => {
            const sel = f.role === k;
            return (
              <button
                type="button" key={k} onClick={() => set("role", k)}
                className="flex-1 px-4 py-[14px] rounded-xl text-[14px] font-semibold flex items-center gap-[10px] transition-all"
                style={{
                  background: sel ? "var(--sage)" : "#fff",
                  color: sel ? "var(--forest)" : "var(--body-color)",
                  boxShadow: `inset 0 0 0 ${sel ? 2 : 1}px ${sel ? "var(--brand-accent)" : "var(--hairline)"}`,
                }}
              >
                <span className="w-4 h-4 rounded-full" style={{ boxShadow: `inset 0 0 0 ${sel ? 5 : 1}px ${sel ? "var(--brand-accent)" : "var(--hairline)"}` }} />
                {k === "aluno" ? "Aluno" : "Professor"}
              </button>
            );
          })}
        </div>
      </div>
      {isAluno && (
        <div className="grid grid-cols-2 gap-3">
          <Field label="Curso">
            <div
              className="flex items-center gap-2 px-[14px] h-12 bg-white rounded-xl"
              style={{ boxShadow: `inset 0 0 0 1px var(--hairline)` }}
            >
              <select
                value={f.course}
                onChange={(e) => set("course", e.target.value as Course)}
                className="flex-1 border-none outline-none bg-transparent text-sm text-ink font-sans appearance-none"
              >
                {COURSES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
              <Caret />
            </div>
          </Field>
          <Field label="Semestre">
            <div
              className="flex items-center gap-2 px-[14px] h-12 bg-white rounded-xl"
              style={{ boxShadow: `inset 0 0 0 1px var(--hairline)` }}
            >
              <select
                value={f.semester}
                onChange={(e) => set("semester", Number(e.target.value))}
                className="flex-1 border-none outline-none bg-transparent text-sm text-ink font-sans appearance-none"
              >
                {[1, 2, 3, 4, 5, 6, 7, 8].map((n) => <option key={n} value={n}>{n}º</option>)}
              </select>
              <Caret />
            </div>
          </Field>
        </div>
      )}
      <div className="mb-[18px]">
        <label className="text-[12px] font-semibold mb-2 block" style={{ color: "rgb(81, 85, 79)" }}>Período</label>
        <div className="flex gap-[10px]">
          {(["matutino", "noturno"] as const).map((p) => {
            const sel = f.period === p;
            return (
              <button type="button" key={p} onClick={() => set("period", p)}
                className="flex-1 px-[14px] py-3 rounded-xl text-[14px] font-medium"
                style={{
                  background: "#fff",
                  color: sel ? "var(--forest)" : "var(--body-color)",
                  boxShadow: `inset 0 0 0 ${sel ? 2 : 1}px ${sel ? "var(--brand-accent)" : "var(--hairline)"}`,
                }}
              >{p[0].toUpperCase() + p.slice(1)}</button>
            );
          })}
        </div>
      </div>
      <Field label="RA" hint={isAluno ? "8 dígitos · AAMMXXXX" : "6 dígitos"}>
        <Input
          value={f.ra}
          onChange={(v) => set("ra", v.replace(/\D/g, "").slice(0, raMaxLen))}
          placeholder={isAluno ? "12345678" : "123456"}
        />
        {raHint && (
          <div className="text-[11px] mt-[6px]" style={{ color: "var(--green)" }}>
            {raHint}
          </div>
        )}
      </Field>
      <Field label="Senha">
        <Input value={f.password} onChange={(v) => set("password", v)} type="password" placeholder="••••••••" />
        {f.password && <StrengthMeter score={score} />}
      </Field>
      <Field label="Confirmar senha">
        <Input value={f.confirm} onChange={(v) => set("confirm", v)} type="password" placeholder="••••••••"
          right={f.confirm && f.confirm === f.password ? (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success)" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
          ) : null}
        />
      </Field>
      {err && (
        <div className="text-[12px] mb-3 px-3 py-2 rounded-lg" style={{ background: "var(--danger-bg)", color: "#7A2929" }}>
          {err}
        </div>
      )}
      <Btn kind="primaryDark" full type="submit" disabled={busy} style={{ height: 48, fontSize: 15, marginTop: 8 }}>
        {busy ? "Criando..." : "Criar conta →"}
      </Btn>
    </form>
  );
}
