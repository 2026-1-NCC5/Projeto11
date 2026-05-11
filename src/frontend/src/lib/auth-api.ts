// Cliente HTTP para o backend FastAPI do Lideranças Empáticas.
// Backend usa cookies httpOnly — frontend NÃO armazena tokens no localStorage.
// O navegador injeta os cookies automaticamente em toda request com `credentials: "include"`.
//
// Configurável via VITE_API_BASE_URL.
//   Dev:        VITE_API_BASE_URL=http://localhost:8000
//   Vercel:     VITE_API_BASE_URL=https://seu-backend.ngrok.app  (na demo)

import {
  authHeader,
  clearAccessToken,
  setAccessToken,
} from "@/lib/auth-token";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API = `${BASE}/api/v1`;

// ── Tipos ─────────────────────────────────────────────────────────────────

export type UserRole = "aluno" | "professor";
export type Period = "matutino" | "noturno";
export type Course =
  | "Administração"
  | "Ciências Contábeis"
  | "Ciências Econômicas";

export type AuthUser = {
  id: string;
  email: string;
  full_name: string;
  ra: string;
  role: UserRole;
  course: Course | null;
  semester: number | null;
  period: Period;
  created_at: string;
};

// O backend retorna o access_token no body por conveniência (debug),
// mas a fonte de verdade é o cookie httpOnly. O frontend ignora esse campo.
type TokenResponse = {
  user: AuthUser;
  access_token: string;
  token_type: "bearer";
};

type MessageResponse = { message: string };

// ── Discriminated union: payload de signup difere por role ────────────────

export type SignupAlunoInput = {
  role: "aluno";
  email: string;            // deve terminar em @edu.fecap.br
  ra: string;               // 8 dígitos AAMMXXXX
  full_name: string;
  password: string;
  period: Period;
  course: Course;           // obrigatório para aluno
  semester: number;         // 1..8 obrigatório para aluno
};

export type SignupProfessorInput = {
  role: "professor";
  email: string;            // deve terminar em @fecap.br
  ra: string;               // 6 dígitos
  full_name: string;
  password: string;
  period: Period;
  // sem course/semester
};

export type SignupInput = SignupAlunoInput | SignupProfessorInput;

// ── Helper de fetch ───────────────────────────────────────────────────────

async function call<T>(
  path: string,
  init: Omit<RequestInit, "body"> & { body?: unknown } = {}
): Promise<T> {
  const { body, headers, ...rest } = init;

  const res = await fetch(`${API}${path}`, {
    ...rest,
    method: rest.method || "POST",
    credentials: "include", // cookies httpOnly (PC); mobile usa Bearer header
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "1",
      ...authHeader(),
      ...(headers || {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let detail: string;
    try {
      const errJson = await res.json();
      detail =
        typeof errJson.detail === "string"
          ? errJson.detail
          : JSON.stringify(errJson.detail);
    } catch {
      detail = res.statusText || `Erro ${res.status}`;
    }
    throw new ApiError(detail, res.status);
  }

  // /logout retorna { message }, demais retornam payloads tipados
  if (res.status === 204) return undefined as T;
  return res.json();
}

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
    this.name = "ApiError";
  }
}

// ── API ───────────────────────────────────────────────────────────────────

async function withTokenPersist(
  p: Promise<TokenResponse>
): Promise<TokenResponse> {
  const res = await p;
  if (res.access_token) setAccessToken(res.access_token);
  return res;
}

export const authApi = {
  /** Cadastra um usuário e já loga (cookie + Bearer no sessionStorage). */
  signup: (input: SignupInput) =>
    withTokenPersist(call<TokenResponse>("/auth/register", { body: input })),

  /** Login por e-mail e senha. */
  login: (email: string, password: string) =>
    withTokenPersist(
      call<TokenResponse>("/auth/login", {
        body: { email, password },
      })
    ),

  /** Rotaciona o refresh token e atualiza o access_token armazenado. */
  refresh: () =>
    withTokenPersist(call<TokenResponse>("/auth/refresh", { body: {} })),

  /** Revoga o refresh token, limpa cookies e o token local. */
  logout: async () => {
    try {
      return await call<MessageResponse>("/auth/logout", { body: {} });
    } finally {
      clearAccessToken();
    }
  },

  /** Retorna o usuário autenticado (cookie OU Bearer). */
  me: () => call<AuthUser>("/auth/me", { method: "GET" }),
};
