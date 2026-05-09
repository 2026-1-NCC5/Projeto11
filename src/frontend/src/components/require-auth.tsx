import { Navigate } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { useAuth } from "@/lib/auth-context";
import { AppShell } from "./app-shell";

export function RequireAuth({ children, professorOnly }: { children: ReactNode; professorOnly?: boolean }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user) return <Navigate to="/login" />;
  if (professorOnly && user.role !== "professor") return <Navigate to="/dashboard" />;
  return <AppShell>{children}</AppShell>;
}
