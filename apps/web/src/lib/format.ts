/** "1 record", "2,727 records". */
export function countLabel(count: number, singular: string, plural = `${singular}s`) {
  return `${count.toLocaleString()} ${count === 1 ? singular : plural}`
}

export function formatNumber(value: number | null | undefined) {
  return value === null || value === undefined ? "–" : value.toLocaleString()
}

/** "12.5%"; a share that rounds to 0.0% but has records reads "<0.1%". */
export function formatPercent(value: number | null | undefined, count?: number | null) {
  if (value === null || value === undefined) return "–"
  if (value < 0.05 && (count ?? 0) > 0) return "<0.1%"
  return `${value.toFixed(1)}%`
}

export const PLATFORM_LABEL: Record<string, string> = { reddit: "Reddit", webmd: "WebMD" }
export const PLATFORM_UNIT: Record<string, string> = { reddit: "records", webmd: "reviews" }
export const ROLE_LABEL: Record<string, string> = { side_effect: "Side effect", purpose: "Purpose", drug: "Drug" }

const VALUE_LABELS: Record<string, string> = {
  female: "Female",
  male: "Male",
  other: "Other",
  "not stated": "Not stated",
  unspecified: "Unspecified",
  daily_life: "Daily life",
  medical_visit: "Medical visit",
  emergency_visit: "Emergency visit",
  hospitalization: "Hospitalization",
  dose_reduced: "Dose reduced",
  escalation_paused: "Escalation paused",
  dose_skipped: "Dose skipped",
  stopped: "Stopped",
  name: "named",
  community: "community",
  profile: "profile",
}

/** Human labels for categorical values from the graph (sex, impact, action...). Ingredients get a capital letter. */
export function valueLabel(value: string | null | undefined) {
  if (value === null || value === undefined) return "–"
  return VALUE_LABELS[value] ?? value.charAt(0).toUpperCase() + value.slice(1)
}

/** "unspecified" means different things for brands (no brand named) and ingredients (none could be assigned). */
export function drugLabel(level: "brand" | "ingredient", value: string) {
  if (value !== "unspecified") return valueLabel(value)
  return level === "brand" ? "No brand named" : "Ingredient not identified"
}

export function scopeLabel(scope: string) {
  if (scope === "all") return "All four ingredients"
  const [, name] = scope.split(":")
  return valueLabel(name)
}
