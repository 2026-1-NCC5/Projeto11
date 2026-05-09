// Cliente HTTP para evidences via FastAPI.
// Privacidade: para alunos, o backend força group_id = grupo do usuário.

import { ApiError } from "@/lib/auth-api";

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const API = `${BASE}/api/v1`;

export type FoodCategory =
  | "arroz"
  | "feijao"
  | "acucar"
  | "macarrao"
  | "oleo"
  | "fuba";

export type EvidenceFeedItem = {
  id: string;
  category: FoodCategory;
  detected_at: string;
  confidence: number;
  frame_url: string;
  group_id: string;
};

export type CategoryCounts = {
  arroz: number;
  feijao: number;
  acucar: number;
  macarrao: number;
  oleo: number;
  fuba: number;
};

export type EvidenceAggregate = {
  counts: CategoryCounts;
  total: number;
};

export type GroupRankingItem = {
  id: string;
  name: string;
  created_at: string;
  kg: number;
};

async function call<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    let detail: string;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      detail = res.statusText || `Erro ${res.status}`;
    }
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

export const evidencesApi = {
  feed: (opts: {
    groupId?: string;
    category?: FoodCategory;
    since?: string;
    limit?: number;
  } = {}) => {
    const qs = new URLSearchParams();
    if (opts.groupId) qs.set("group_id", opts.groupId);
    if (opts.category) qs.set("category", opts.category);
    if (opts.since) qs.set("since", opts.since);
    if (opts.limit !== undefined) qs.set("limit", String(opts.limit));
    const tail = qs.toString();
    return call<EvidenceFeedItem[]>(`/evidences${tail ? `?${tail}` : ""}`);
  },

  aggregate: (opts: { groupId?: string } = {}) => {
    const qs = opts.groupId ? `?group_id=${opts.groupId}` : "";
    return call<EvidenceAggregate>(`/evidences/aggregate${qs}`);
  },

  ranking: () => call<GroupRankingItem[]>("/groups/ranking"),
};
