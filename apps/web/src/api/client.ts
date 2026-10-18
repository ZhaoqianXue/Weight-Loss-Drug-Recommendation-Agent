import createClient from "openapi-fetch"

import type { components, paths } from "./schema"

// Types come from the FastAPI schema (pnpm gen:api); never edit schema.d.ts by hand.
type Schemas = components["schemas"]
export type Overview = Schemas["Overview"]
export type Soc = Schemas["Soc"]
export type GraphQuery = Schemas["GraphQuery"]
export type GraphView = Schemas["GraphView"]
export type GraphNode = Schemas["GraphNode"]
export type GraphLink = Schemas["GraphLink"]
export type GraphSummary = Schemas["GraphSummary"]
export type Patterns = Schemas["Patterns"]
export type PatternSection = Schemas["PatternSection"]
export type PatternRow = Schemas["PatternRow"]
export type PopulationCell = Schemas["PopulationCell"]
export type FaersQuery = Schemas["FaersQuery"]
export type FaersView = Schemas["FaersView"]
export type FaersRow = Schemas["FaersRow"]
export type EvidenceQuery = Schemas["EvidenceQuery"]
export type EvidenceView = Schemas["EvidenceView"]
export type Statement = Schemas["Statement"]
export type AskStatus = Schemas["AskStatus"]
export type AskResult = Schemas["AskResult"]
export type Citation = Schemas["Citation"]
export type Quote = Schemas["Quote"]
export type ToolCall = Schemas["ToolCall"]
export type AskEvent = Schemas["AskEvent"]
export type ToolEvent = Schemas["ToolEvent"]
export type ModelEvent = Schemas["ModelEvent"]
export type Platform = GraphQuery["platform"]
export type Target = EvidenceQuery["target"]

export const api = createClient<paths>({ baseUrl: "" })

/** Unwrap an openapi-fetch result, turning HTTP errors into exceptions for TanStack Query. */
export async function unwrap<T>(request: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await request
  if (data === undefined) {
    const detail = (error as { detail?: unknown } | undefined)?.detail
    throw new Error(typeof detail === "string" ? detail : `The server answered ${response.status}.`)
  }
  return data
}

/**
 * Ask the agent one question. The server streams progress as server-sent events (one AskEvent per `data:` line);
 * `onEvent` receives each one in order. Resolves when the stream ends.
 */
export async function streamAsk(question: string, onEvent: (event: AskEvent) => void, signal?: AbortSignal) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  })
  if (!response.ok || !response.body) {
    throw new Error(`The agent could not be reached (${response.status}).`)
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    let end = buffer.indexOf("\n\n")
    while (end >= 0) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n")
      if (data) onEvent(JSON.parse(data) as AskEvent)
      end = buffer.indexOf("\n\n")
    }
  }
}
