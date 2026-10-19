import type { GraphNode } from "@/api/client"
import type { NodeSizes } from "@/state"

// Colors of the original page: one per brand, one for purposes (the old "condition" yellow), one for side effects.
export const DRUG_COLORS: Record<string, string> = {
  Wegovy: "#EFC29E",
  Ozempic: "#6D7EA2",
  Rybelsus: "#9DB8A9",
  Zepbound: "#FFB347",
  Mounjaro: "#B28DFF",
  Victoza: "#77DD77",
  Trulicity: "#8056A6",
  Saxenda: "#FF6961",
}
// An ingredient takes the color of its first-approved brand in the palette above.
export const INGREDIENT_COLORS: Record<string, string> = {
  semaglutide: DRUG_COLORS.Ozempic,
  tirzepatide: DRUG_COLORS.Mounjaro,
  liraglutide: DRUG_COLORS.Victoza,
  dulaglutide: DRUG_COLORS.Trulicity,
}
export const DEFAULT_DRUG_COLOR = "#D36F66"
export const PURPOSE_COLOR = "#FFDA66"
export const SIDE_EFFECT_COLOR = "#B89EC8"

export const GRAPH_THEME = {
  light: { surface: "#fffefc", text: "#26364b", title: "#1a365d", muted: "#68798c", link: "#8d9cae", linkOpacity: 0.24, ring: "#fffefc", selected: "#1a365d" },
  dark: { surface: "#131c2b", text: "#e0e6ef", title: "#c9d8ec", muted: "#8d9bb0", link: "#7d8ea6", linkOpacity: 0.2, ring: "#131c2b", selected: "#eef3f9" },
}

export type GraphMode = keyof typeof GRAPH_THEME

export function drugColor(name: string) {
  return DRUG_COLORS[name] ?? INGREDIENT_COLORS[name] ?? DEFAULT_DRUG_COLOR
}

export function nodeColor(node: GraphNode) {
  if (node.kind === "drug") return drugColor(node.label)
  return node.kind === "purpose" ? PURPOSE_COLOR : SIDE_EFFECT_COLOR
}

/** Largest record count of a node over its roles (a drug has one per role shown). */
export function nodeRecords(node: GraphNode) {
  return Math.max(0, ...Object.values(node.records))
}

/**
 * Radius = base size from the sliders + up to 16px by the square root of records, relative to the largest node of the
 * same kind in view (so Reddit and WebMD views both use the full range).
 */
export function makeRadius(nodes: GraphNode[], sizes: NodeSizes) {
  const largest: Record<string, number> = {}
  for (const node of nodes) largest[node.kind] = Math.max(largest[node.kind] ?? 1, nodeRecords(node))
  return (node: GraphNode) => sizes[node.kind] * 2 + 16 * Math.sqrt(nodeRecords(node) / (largest[node.kind] ?? 1))
}

export function legendDrugs(nodes: GraphNode[]) {
  return nodes.filter((node) => node.kind === "drug").map((node) => ({ label: node.label, color: drugColor(node.label) }))
}
