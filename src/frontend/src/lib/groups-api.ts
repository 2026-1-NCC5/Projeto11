// Cliente HTTP para os endpoints de grupos do FastAPI.
// Usa cookies httpOnly via credentials: "include" — mesmo padrão do auth-api.

import { ApiError } from "@/lib/auth-api";
import { authHeader } from "@/lib/auth-token";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API = `${BASE}/api/v1`;

export type GroupSummary = {
  id: string;
  name: string;
  created_at: string;
  created_by: string;
  member_count: number;
  kg: number;
};

export type GroupListResponse = {
  groups: GroupSummary[];
  total_groups: number;
  total_students: number;
  complete_groups: number;
};

export type GroupMember = {
  user_id: string;
  full_name: string;
  email: string;
  ra: string;
  course: string | null;
  semester: number | null;
  joined_at: string;
  is_leader: boolean;
};

export type EvidenceBrief = {
  id: string;
  category: "arroz" | "feijao" | "acucar" | "macarrao" | "oleo" | "fuba";
  detected_at: string;
  confidence: number;
  frame_url: string;
};

export type GroupDetail = {
  id: string;
  name: string;
  created_at: string;
  created_by: string;
  members: GroupMember[];
  kg: number;
  recent_evidences: EvidenceBrief[];
};

export type StudentSearch = {
  id: string;
  full_name: string;
  email: string;
  ra: string;
  course: string | null;
  semester: number | null;
  has_group: boolean;
};

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "1",
      ...authHeader(),
      ...(init.headers || {}),
    },
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

  if (res.status === 204) return undefined as T;
  return res.json();
}

export const groupsApi = {
  listAdmin: () => call<GroupListResponse>("/groups"),

  myGroup: () => call<GroupDetail | null>("/groups/me"),

  detail: (id: string) => call<GroupDetail>(`/groups/${id}`),

  create: (input: { name: string; member_ids: string[] }) =>
    call<GroupSummary>("/groups", {
      method: "POST",
      body: JSON.stringify(input),
    }),

  addMember: (groupId: string, userId: string) =>
    call<GroupMember>(`/groups/${groupId}/members`, {
      method: "POST",
      body: JSON.stringify({ user_id: userId }),
    }),

  removeMember: (groupId: string, userId: string) =>
    call<void>(`/groups/${groupId}/members/${userId}`, { method: "DELETE" }),

  searchStudents: (q: string, limit = 30) =>
    call<StudentSearch[]>(
      `/users/students?q=${encodeURIComponent(q)}&limit=${limit}`
    ),
};
