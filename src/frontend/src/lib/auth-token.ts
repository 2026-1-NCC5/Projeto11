// Armazena o access_token em sessionStorage como fallback ao cookie httpOnly.
// Necessário porque iOS Safari/Chrome mobile bloqueiam cookies cross-site
// (frontend Vercel <-> backend ngrok) mesmo com SameSite=None; Secure.
// sessionStorage some quando a aba é fechada — limita o risco vs. localStorage.

const KEY = "le_access_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setAccessToken(token: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(KEY, token);
  } catch {
    // storage cheio ou indisponível — silencioso, cookie ainda funciona em PC
  }
}

export function clearAccessToken(): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.removeItem(KEY);
  } catch {
    // noop
  }
}

export function authHeader(): Record<string, string> {
  const t = getAccessToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}
