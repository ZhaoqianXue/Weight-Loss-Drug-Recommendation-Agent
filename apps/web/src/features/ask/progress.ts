import type { AskEvent, AskResult } from "@/api/client"
import { PLATFORM_LABEL, valueLabel } from "@/lib/format"

export type Step = {
  key: string
  kind: "model" | "tool"
  status: "running" | "done" | "error"
  title: string
  details: string[]
  note?: string | null
  seconds?: number | null
}

export type Turn = {
  id: number
  question: string
  status: "running" | "done" | "error"
  steps: Step[]
  result?: AskResult
  error?: string
  startedAt: number
}

type Args = Record<string, unknown>

const list = (value: unknown) => (Array.isArray(value) ? (value as unknown[]).map(String) : [])

/** "1 term" or "2 terms" (and "1 term group" for a MedDRA group): the codes themselves mean nothing to a reader. */
function concepts(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) return []
  const items = value as { level?: string }[]
  const groups = items.filter((c) => c.level && c.level !== "pt").length
  const terms = items.length - groups
  return [terms > 0 && `${terms} ${terms === 1 ? "term" : "terms"}`, groups > 0 && `${groups} term ${groups === 1 ? "group" : "groups"}`].filter(Boolean) as string[]
}

/** Readable chips for the filters a tool call used. */
function details(args: Args) {
  const out: string[] = []
  const platforms = list(args.platforms)
  if (platforms.length) out.push(platforms.map((p) => PLATFORM_LABEL[p] ?? p).join(" + "))
  for (const key of ["brands", "ingredients", "scopes"]) out.push(...list(args[key]).map((v) => valueLabel(v.replace(/^\w+:/, ""))))
  out.push(...list(args.age_groups).map((v) => `age ${v}`), ...list(args.sexes).map(valueLabel))
  out.push(...concepts(args.targets))
  if (concepts(args.require_purposes).length) out.push(`with a stated purpose`)
  out.push(...list(args.actions).map((v) => `action: ${valueLabel(v)}`), ...list(args.impacts).map((v) => `impact: ${valueLabel(v)}`))
  out.push(...list(args.intensities).map((v) => `intensity: ${valueLabel(v)}`))
  if (list(args.group_by).length) out.push(`by ${list(args.group_by).join(" × ")}`)
  if (typeof args.wording === "string") out.push(`“${args.wording}”`)
  if (typeof args.sentence === "string") out.push(`sentences with “${args.sentence}”`)
  return out
}

export function toolTitle(tool: string, args: Args) {
  switch (tool) {
    case "link_terms":
      return `Linking “${String(args.text ?? "")}” to MedDRA ${args.kind === "purpose" ? "purpose" : "side-effect"} terms`
    case "query_graph":
      return `Counting ${args.target === "purpose" ? "purpose" : "side-effect"} records in the graph`
    case "get_evidence":
      return "Reading example statements"
    case "compare_faers":
      return "Comparing with FAERS reports"
    case "run_python":
      return "Computing over the statement tables"
    default:
      return tool
  }
}

/** Apply one streamed event to a turn. */
export function applyEvent(turn: Turn, event: AskEvent): Turn {
  switch (event.type) {
    case "queued": {
      const title = `Waiting for a free agent: ${event.ahead} ${event.ahead === 1 ? "question" : "questions"} from other visitors start first`
      return { ...turn, steps: [...turn.steps, { key: "queued", kind: "model", status: "running", title, details: [] }] }
    }
    case "model": {
      const key = `model-${event.round}`
      // The question has left the queue once the model is working on it.
      const steps = turn.steps
        .filter((s) => s.key !== key)
        .map((s) => (s.key === "queued" || (s.key === "model-0" && key !== "model-0") ? { ...s, status: "done" as const } : s))
      if (event.status === "running") {
        const title =
          event.round === 0 ? "Reconnecting to the model service" : event.round === 1 ? "Reading the question" : "Deciding the next step"
        return { ...turn, steps: [...steps, { key, kind: "model", status: "running", title, details: [] }] }
      }
      // A response without tool calls is the final answer, which the verifier checks next.
      if (event.tool_calls === 0) {
        return { ...turn, steps: [...steps, { key, kind: "model", status: "running", title: "Writing the answer and verifying every number and quote", details: [] }] }
      }
      return { ...turn, steps }
    }
    case "tool": {
      const args = (event.arguments ?? {}) as Args
      const step: Step = {
        key: event.step,
        kind: "tool",
        status: event.status,
        title: toolTitle(event.tool, args),
        details: details(args),
        note:
          event.status === "error"
            ? event.detail
            : event.status === "done"
              ? event.tool === "run_python"
                ? `${Math.round(event.seconds ?? 0)} s`
                : `${event.rows ?? 0} rows`
              : null,
        seconds: event.seconds,
      }
      const exists = turn.steps.some((s) => s.key === event.step)
      return { ...turn, steps: exists ? turn.steps.map((s) => (s.key === event.step ? step : s)) : [...turn.steps, step] }
    }
    case "result":
      return { ...turn, status: "done", result: event.result, steps: turn.steps.map((s) => ({ ...s, status: s.status === "running" ? "done" : s.status })) }
    case "error":
      return { ...turn, status: "error", error: event.message }
    default:
      return turn
  }
}
