import { keepPreviousData, useQuery } from "@tanstack/react-query"

import { api, unwrap, type EvidenceQuery, type FaersQuery, type GraphQuery } from "./client"

// The graph is frozen while the server runs, so every view stays fresh for the session.
export function useOverview() {
  return useQuery({ queryKey: ["overview"], queryFn: () => unwrap(api.GET("/api/overview")), staleTime: Infinity })
}

export function useGraphView(query: GraphQuery, enabled = true) {
  return useQuery({
    queryKey: ["graph", query],
    queryFn: () => unwrap(api.POST("/api/graph", { body: query })),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    enabled,
  })
}

export function usePatterns() {
  return useQuery({ queryKey: ["patterns"], queryFn: () => unwrap(api.GET("/api/patterns")), staleTime: Infinity })
}

export function useFaers(query: FaersQuery) {
  return useQuery({
    queryKey: ["faers", query],
    queryFn: () => unwrap(api.POST("/api/faers", { body: query })),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

export function useEvidence(query: EvidenceQuery | null) {
  return useQuery({
    queryKey: ["evidence", query],
    queryFn: () => unwrap(api.POST("/api/evidence", { body: query as EvidenceQuery })),
    enabled: query !== null,
    staleTime: Infinity,
  })
}

export function useAskStatus() {
  return useQuery({ queryKey: ["ask-status"], queryFn: () => unwrap(api.GET("/api/ask/status")), staleTime: 30_000 })
}
