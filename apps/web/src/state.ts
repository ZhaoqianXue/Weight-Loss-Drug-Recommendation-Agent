import type { GraphQuery, Platform } from "@/api/client"

export type Level = GraphQuery["level"]
export type Show = GraphQuery["show"]

/** What the knowledge-graph explorer shows. null means "all". */
export type Explorer = {
  platform: Platform
  level: Level
  show: Show
  drugs: string[] | null
  ageGroups: string[] | null
  sexes: string[] | null
  soc: string | null
  topN: number
}

export const INITIAL_EXPLORER: Explorer = {
  platform: "reddit",
  level: "brand",
  show: "both",
  drugs: null,
  ageGroups: null,
  sexes: null,
  soc: null,
  topN: 4,
}

export type NodeSizes = { drug: number; purpose: number; side_effect: number }
export const INITIAL_SIZES: NodeSizes = { drug: 8, purpose: 5, side_effect: 5 }

export function toGraphQuery(state: Explorer): GraphQuery {
  return {
    platform: state.platform,
    level: state.level,
    show: state.show,
    drugs: state.drugs,
    age_groups: state.ageGroups,
    sexes: state.sexes,
    soc: state.soc,
    top_n: state.topN,
  }
}

/** A node or link the reader selected in the graph; the evidence panel shows its statements. */
export type Selection = {
  target: "side_effect" | "purpose"
  code: string | null
  label: string
  drug: string | null
}

// Tool filters the explorer cannot apply, named for the reader.
const NOT_IN_EXPLORER: Record<string, string> = {
  intensities: "intensity",
  impacts: "impact",
  actions: "action taken",
  require_purposes: "required purpose",
  require_side_effects: "required side effect",
  named_drugs_only: "named drugs only",
  wording: "wording",
}

/** The tool filters of a call that the explorer leaves out when it shows that call. */
export function omittedFilters(args: Record<string, unknown>) {
  const omitted = Object.keys(NOT_IN_EXPLORER).filter((key) => args[key] !== undefined && args[key] !== null && args[key] !== false)
  const targets = Array.isArray(args.targets) ? (args.targets as { level?: string }[]) : []
  if (targets.some((t) => t.level === "pt" || t.level === "hlt" || t.level === "hlgt")) omitted.push("targets")
  return omitted.map((key) => NOT_IN_EXPLORER[key] ?? "term")
}

/**
 * Explorer filters that show what an agent tool call counted: its platform (when one), drug level and drugs, role,
 * SOC target and population filters. null when the call has nothing the graph can show.
 */
export function explorerFromCall(tool: string, args: Record<string, unknown>): Partial<Explorer> | null {
  if (tool !== "query_graph" && tool !== "get_evidence") return null
  const list = (value: unknown) => (Array.isArray(value) && value.length ? (value as unknown[]).map(String) : null)
  const update: Partial<Explorer> = { drugs: null, ageGroups: list(args.age_groups), sexes: list(args.sexes), soc: null }
  const platforms = list(args.platforms)
  if (platforms?.length === 1) update.platform = platforms[0] as Explorer["platform"]
  const brands = list(args.brands)?.filter((b) => b !== "unspecified")
  const ingredients = list(args.ingredients)?.filter((i) => i !== "unspecified")
  if (brands?.length) Object.assign(update, { level: "brand", drugs: brands })
  else if (ingredients?.length) Object.assign(update, { level: "ingredient", drugs: ingredients })
  if (args.target === "side_effect" || args.target === "purpose") update.show = args.target
  const soc = Array.isArray(args.targets) ? (args.targets as { level?: string; code?: string }[]).find((t) => t.level === "soc") : undefined
  if (soc?.code) update.soc = soc.code
  return update
}

// Explorer filters in the address bar, so a view can be linked and reopened exactly. Defaults are left out.
const PARAMS = { platform: "platform", level: "level", show: "show", drugs: "drugs", ageGroups: "age", sexes: "sex", soc: "soc", topN: "top" } as const

export function explorerFromSearch(search: string): Explorer {
  const params = new URLSearchParams(search)
  const pick = <T extends string>(key: string, allowed: readonly T[], fallback: T) => {
    const value = params.get(key) as T | null
    return value && allowed.includes(value) ? value : fallback
  }
  const list = (key: string) => (params.has(key) ? (params.get(key) ?? "").split(",").filter(Boolean) : null)
  const top = Number(params.get(PARAMS.topN))
  return {
    platform: pick(PARAMS.platform, ["reddit", "webmd"] as const, INITIAL_EXPLORER.platform),
    level: pick(PARAMS.level, ["brand", "ingredient"] as const, INITIAL_EXPLORER.level),
    show: pick(PARAMS.show, ["both", "side_effect", "purpose"] as const, INITIAL_EXPLORER.show),
    drugs: list(PARAMS.drugs),
    ageGroups: list(PARAMS.ageGroups),
    sexes: list(PARAMS.sexes),
    soc: params.get(PARAMS.soc) || null,
    topN: Number.isInteger(top) && top >= 1 && top <= 10 ? top : INITIAL_EXPLORER.topN,
  }
}

export function explorerToSearch(state: Explorer) {
  const params = new URLSearchParams()
  for (const [field, key] of Object.entries(PARAMS) as [keyof Explorer, string][]) {
    const value = state[field]
    if (value === INITIAL_EXPLORER[field] || value === null) continue
    params.set(key, Array.isArray(value) ? value.join(",") : String(value))
  }
  const text = params.toString()
  return text ? `?${text}` : ""
}
