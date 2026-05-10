import { cn } from "@/lib/utils";
import type { ReactNode, CSSProperties } from "react";

/* ============ Digits ============ */
export function Digits({
  children, size = 40, weight = 700, color, className,
}: { children: ReactNode; size?: number; weight?: number; color?: string; className?: string }) {
  return (
    <span
      className={cn("font-mono tabular tracking-tight", className)}
      style={{ fontSize: size, fontWeight: weight, color, lineHeight: 1, letterSpacing: "-0.03em" }}
    >
      {children}
    </span>
  );
}

/* ============ Btn ============ */
type BtnKind = "primary" | "primaryDark" | "outline" | "ghost" | "danger" | "onDark";
export function Btn({
  kind = "primary", children, full, sm, style, className, onClick, type = "button", disabled,
}: {
  kind?: BtnKind; children: ReactNode; full?: boolean; sm?: boolean; style?: CSSProperties;
  className?: string; onClick?: () => void; type?: "button" | "submit"; disabled?: boolean;
}) {
  const variants: Record<BtnKind, string> = {
    primary: "bg-brand-accent text-white hover:opacity-90",
    primaryDark: "bg-forest text-white hover:opacity-90",
    outline: "bg-transparent text-forest shadow-[inset_0_0_0_1px_var(--forest)]",
    ghost: "bg-transparent text-body shadow-[inset_0_0_0_1px_var(--hairline)] hover:bg-cream",
    danger: "bg-transparent text-danger shadow-[inset_0_0_0_1px_var(--danger)] hover:bg-danger-bg",
    onDark: "bg-white text-forest hover:opacity-90",
  };
  return (
    <button
      type={type} onClick={onClick} disabled={disabled}
      style={{ letterSpacing: "-0.01em", ...style }}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-all duration-150 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed",
        sm ? "px-[14px] py-[8px] text-[13px]" : "px-[20px] py-[11px] text-[14px]",
        full && "w-full",
        variants[kind],
        className,
      )}
    >{children}</button>
  );
}

/* ============ Card ============ */
export function Card({
  children, style, accent, className,
}: { children: ReactNode; style?: CSSProperties; accent?: string; className?: string }) {
  return (
    <div
      className={cn(
        "relative bg-white rounded-2xl border border-hairline overflow-hidden",
        "shadow-[0_1px_1px_rgba(20,56,36,0.04),0_0_0.5px_rgba(20,56,36,0.06)]",
        className,
      )}
      style={style}
    >
      {accent && (
        <div className="absolute top-0 bottom-0 left-0 w-1" style={{ background: accent }} />
      )}
      {children}
    </div>
  );
}

/* ============ Chip ============ */
type ChipTone =
  | "sage" | "arroz" | "feijao" | "macarrao" | "acucar" | "oleo" | "fuba"
  | "success" | "warning" | "danger" | "dark" | "gold";
export function Chip({ tone = "sage", children }: { tone?: ChipTone; children: ReactNode }) {
  const tones: Record<ChipTone, { bg: string; fg: string; ring?: string }> = {
    sage: { bg: "var(--sage)", fg: "var(--forest)" },
    arroz: { bg: "var(--arroz-light)", fg: "#7A5E1F" },
    feijao: { bg: "var(--feijao-light)", fg: "var(--feijao)" },
    macarrao: { bg: "var(--macarrao-light)", fg: "#8C3D1F" },
    acucar: { bg: "#F4ECDD", fg: "#6B4F1A" },
    oleo: { bg: "#FFF5D9", fg: "#8C6B14" },
    fuba: { bg: "#FBE6B5", fg: "#7A5414" },
    success: { bg: "var(--success-bg)", fg: "#1F5C36" },
    warning: { bg: "var(--warning-bg)", fg: "#7A5414" },
    danger: { bg: "var(--danger-bg)", fg: "#7A2929" },
    dark: { bg: "var(--forest)", fg: "#fff" },
    gold: { bg: "transparent", fg: "var(--gold)", ring: "var(--gold)" },
  };
  const t = tones[tone] ?? tones.sage;
  return (
    <span
      className="inline-flex items-center gap-[6px] px-[10px] py-[4px] rounded-full text-[12px] font-semibold"
      style={{
        background: t.bg, color: t.fg, letterSpacing: "-0.005em",
        boxShadow: t.ring ? `inset 0 0 0 1px ${t.ring}` : undefined,
      }}
    >{children}</span>
  );
}

/* ============ PageHeader ============ */
export function PageHeader({
  title, subtitle, right,
}: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-3 px-4 sm:px-6 lg:px-8 pt-5 sm:pt-6 lg:pt-8 pb-4 lg:pb-6 border-b border-hairline bg-cream">
      <div className="min-w-0">
        <div className="text-[11px] font-bold tracking-[1.5px] text-soft mb-1 lg:mb-2">LIDERANÇAS EMPÁTICAS</div>
        <h1 className="m-0 text-[24px] sm:text-[28px] lg:text-[34px] font-semibold leading-[1.1] tracking-[-0.02em] text-ink">{title}</h1>
        {subtitle && <div className="mt-[6px] text-[13px] sm:text-sm text-soft">{subtitle}</div>}
      </div>
      {right && <div className="flex items-center gap-[10px] flex-wrap">{right}</div>}
    </div>
  );
}

/* ============ Field & Input ============ */
export function Field({
  label, hint, children, status,
}: { label: string; hint?: string; children: ReactNode; status?: { text: string; color?: string } }) {
  return (
    <div className="mb-[18px]">
      <div className="flex justify-between mb-2">
        <label className="text-[12px] font-semibold text-body" style={{ letterSpacing: "-0.005em" }}>{label}</label>
        {hint && <span className="text-[11px] text-soft">{hint}</span>}
      </div>
      {children}
      {status && (
        <div className="text-[11px] mt-[6px] flex items-center gap-1" style={{ color: status.color || "var(--success)" }}>
          {status.text}
        </div>
      )}
    </div>
  );
}

export function Input({
  value, onChange, placeholder, type = "text", right, focused, autoComplete,
}: {
  value?: string; onChange?: (v: string) => void; placeholder?: string;
  type?: string; right?: ReactNode; focused?: boolean; autoComplete?: string;
}) {
  return (
    <div
      className="flex items-center gap-2 px-[14px] h-12 bg-white rounded-xl"
      style={{ boxShadow: focused ? `inset 0 0 0 2px var(--brand-accent)` : `inset 0 0 0 1px var(--hairline)` }}
    >
      <input
        type={type}
        value={value ?? ""}
        autoComplete={autoComplete}
        onChange={(e) => onChange?.(e.target.value)}
        placeholder={placeholder}
        className="flex-1 border-none outline-none bg-transparent text-sm text-ink font-sans"
      />
      {right}
    </div>
  );
}

/* ============ Caret ============ */
export function Caret() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--soft)" strokeWidth="2">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
