// AuthContext — fonte de verdade é o backend (cookie httpOnly).
// Não armazenamos tokens no localStorage. Em cada mount, perguntamos
// ao backend "quem está logado?" via /auth/me. Isso preserva sessão
// no F5 sem expor JWT ao JavaScript.
//
// Use em componentes:
//   const { user, login, logout, signup, isAuthenticated, isLoading } = useAuth();

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import {
  ApiError,
  authApi,
  type AuthUser,
  type SignupInput,
} from "@/lib/auth-api";

type AuthContextValue = {
  user: AuthUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;       // true enquanto faz hydrate inicial
  login: (email: string, password: string) => Promise<AuthUser>;
  signup: (input: SignupInput) => Promise<AuthUser>;
  logout: () => Promise<void>;
  refetch: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Hydrate inicial: pergunta ao backend se há sessão válida.
  // Se o cookie expirou mas o refresh ainda é válido, tenta refresh.
  const hydrate = useCallback(async () => {
    try {
      const me = await authApi.me();
      setUser(me);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        // Tenta renovar antes de desistir
        try {
          await authApi.refresh();
          const me = await authApi.me();
          setUser(me);
        } catch {
          setUser(null);
        }
      } else {
        setUser(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    hydrate();
  }, [hydrate]);

  const login = useCallback(async (email: string, password: string) => {
    const res = await authApi.login(email, password);
    setUser(res.user);
    return res.user;
  }, []);

  const signup = useCallback(async (input: SignupInput) => {
    const res = await authApi.signup(input);
    setUser(res.user);
    return res.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      setUser(null);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated: user !== null,
      isLoading,
      login,
      signup,
      logout,
      refetch: hydrate,
    }),
    [user, isLoading, login, signup, logout, hydrate]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth deve ser usado dentro de <AuthProvider>");
  return ctx;
}
