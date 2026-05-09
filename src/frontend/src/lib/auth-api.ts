// Cliente HTTP para o backend FastAPI do Lideranças Empáticas.
// Backend usa cookies httpOnly — frontend NÃO armazena tokens no localStorage.
// O navegador injeta os cookies automaticamente em toda request com `credentials: "include"`.
//
// Configurável via VITE_API_BASE_URL.
//   Dev:        VITE_API_BASE_URL=http://localhost:8000
//   Vercel:     VITE_API_BASE_URL=https://seu-backend.ngrok.app  (na demo)

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
    credentials: "include", // CRÍTICO: envia/recebe cookies httpOnly
    headers: {
      "Content-Type": "application/json",
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

export const authApi = {
  /** Cadastra um usuário e já loga (cookies setados no response). */
  signup: (input: SignupInput) =>
    call<TokenResponse>("/auth/register", { body: input }),

  /** Login por e-mail e senha. Cookies httpOnly são setados pelo backend. */
  login: (email: string, password: string) =>
    call<TokenResponse>("/auth/login", { body: { email, password } }),

  /** Rotaciona o refresh token. O cookie é injetado automaticamente. */
  refresh: () => call<TokenResponse>("/auth/refresh", { body: {} }),

  /** Revoga o refresh token e limpa cookies. */
  logout: () => call<MessageResponse>("/auth/logout", { body: {} }),

  /** Retorna o usuário autenticado pelo cookie atual. */
  me: () => call<AuthUser>("/auth/me", { method: "GET" }),
};
