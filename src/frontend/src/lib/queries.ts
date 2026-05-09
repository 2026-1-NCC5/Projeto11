import { useQuery } from "@tanstack/react-query";
import { useAuth } from "@/lib/auth-context";
import { groupsApi } from "@/lib/groups-api";
import { evidencesApi, type FoodCategory } from "@/lib/evidences-api";

export type { FoodCategory };

export function useEvidencesAggregate(opts?: { groupId?: string }) {
  return useQuery({
    queryKey: ["evidences-agg", opts?.groupId ?? "all"],
    queryFn: async () => {
      const res = await evidencesApi.aggregate({ groupId: opts?.groupId });
      return { counts: res.counts, total: res.total };
    },
  });
}

export function useEvidencesFeed(opts?: {
  groupId?: string;
  limit?: number;
  category?: FoodCategory | "all";
}) {
  return useQuery({
    queryKey: [
      "evidences-feed",
      opts?.groupId ?? "all",
      opts?.limit ?? 200,
      opts?.category ?? "all",
    ],
    queryFn: () =>
      evidencesApi.feed({
        groupId: opts?.groupId,
        category: opts?.category && opts.category !== "all" ? opts.category : undefined,
        limit: opts?.limit ?? 200,
      }),
  });
}

export function useGroupsRanking() {
  return useQuery({
    queryKey: ["groups-ranking"],
    queryFn: () => evidencesApi.ranking(),
  });
}

export function useMyGroup() {
  const { user } = useAuth();
  return useQuery({
    enabled: !!user,
    queryKey: ["my-group", user?.id],
    queryFn: async () => {
      const detail = await groupsApi.myGroup();
      if (!detail || !user) return null;
      return {
        group: {
          id: detail.id,
          name: detail.name,
          created_at: detail.created_at,
          created_by: detail.created_by,
        },
        members: detail.members.map((m) => ({
          user_id: m.user_id,
          full_name: m.full_name,
          email: m.email,
          ra: m.ra,
          isLeader: m.is_leader,
          isMe: m.user_id === user.id,
        })),
        kg: detail.kg,
      };
    },
  });
}

export function useAllGroupsAdmin() {
  return useQuery({
    queryKey: ["groups-admin"],
    queryFn: async () => {
      const data = await groupsApi.listAdmin();
      const list = data.groups.map((g) => ({
        id: g.id,
        name: g.name,
        created_at: g.created_at,
        count: g.member_count,
        kg: g.kg,
        status: g.member_count >= 4 ? "ativo" : "incompleto",
        memberIds: [] as string[],
      }));
      return {
        groups: list,
        totals: {
          totalGroups: data.total_groups,
          totalStudents: data.total_students,
          completeGroups: data.complete_groups,
        },
      };
    },
  });
}

export function useTimeSeries(days = 60, groupId?: string) {
  return useQuery({
    queryKey: ["timeseries", days, groupId ?? "all"],
    queryFn: async () => {
      const since = new Date(Date.now() - days * 86400000).toISOString();
      const data = await evidencesApi.feed({
        groupId,
        since,
        limit: 1000,
      });
      const buckets = 12;
      const start = Date.now() - days * 86400000;
      const span = days * 86400000;
      type Series = { label: string; arroz: number; feijao: number; macarrao: number };
      const series: Series[] = [];
      for (let i = 0; i < buckets; i++) {
        const d = new Date(start + (i * span) / (buckets - 1));
        series.push({
          label: `${String(d.getDate()).padStart(2, "0")}/${String(d.getMonth() + 1).padStart(2, "0")}`,
          arroz: 0,
          feijao: 0,
          macarrao: 0,
        });
      }
      for (const r of data) {
        if (r.category !== "arroz" && r.category !== "feijao" && r.category !== "macarrao") continue;
        const t = new Date(r.detected_at).getTime();
        const idx = Math.min(buckets - 1, Math.max(0, Math.round(((t - start) / span) * (buckets - 1))));
        series[idx][r.category] += 1;
      }
      let a = 0, f = 0, m = 0;
      return series.map((s) => {
        a += s.arroz; f += s.feijao; m += s.macarrao;
        return { label: s.label, arroz: a, feijao: f, macarrao: m };
      });
    },
  });
}
