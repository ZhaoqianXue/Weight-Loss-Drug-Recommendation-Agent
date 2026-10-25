import type { AskResult } from "@/api/client"

import { toolTitle } from "./progress"

const SOURCE_FIELDS = ["platform", "scope", "brand", "ingredient", "age_group", "sex", "pt", "concept", "soc", "intensity", "impact", "action_taken"]

/** The answer as Markdown: text with its [cN] tags, then one source line per tag, the quotes and the run details. */
export function answerMarkdown(result: AskResult) {
  const lines = [`**Q:** ${result.question ?? ""}`, "", result.answer.trim(), ""]
  if (result.citations.length) {
    lines.push("**Sources**", "")
    for (const c of result.citations) {
      const row = c.source ?? {}
      const where = SOURCE_FIELDS.filter((k) => row[k] !== undefined && row[k] !== null)
        .map((k) => `${k.replaceAll("_", " ")} ${String(row[k])}`)
        .join(", ")
      const call = c.call === "t0" ? "graph overview" : `graph query ${c.call.replace(/^t/, "")}`
      if (c.tool === "run_python") {
        const shown = (row.output_lines as string[] | undefined)?.[0]
        lines.push(`- [${c.id}] ${c.value} (step ${c.call.replace(/^t/, "")}: computed in code over the statement tables${shown ? `; output "${shown.trim()}"` : ""})`)
        continue
      }
      lines.push(`- [${c.id}] ${c.field.replaceAll("_", " ")} = ${c.value} (${call}, row ${c.row}${c.tool ? `: ${toolTitle(c.tool, c.arguments).toLowerCase()}` : ""}${where ? `; ${where}` : ""})`)
    }
    lines.push("")
  }
  if (result.quotes.length) {
    lines.push("**Quoted statements**", "")
    for (const q of result.quotes) {
      const s = q.statement
      lines.push(`> “${q.text}” (${[s?.platform, s?.record_id, s?.date].filter(Boolean).join(", ")}; statement ${q.edge_id})`, "")
    }
  }
  if (result.limitations.length) lines.push("**Limits**", "", ...result.limitations.map((l) => `- ${l}`), "")
  lines.push(
    `_AskGLP1 ${result.agent_version ?? ""}, ${result.model ?? ""} (${result.reasoning_effort ?? ""}), run ${result.run_id ?? ""}; ${result.verified ? "every number and quote verified against the graph" : "NOT verified"}._`
  )
  return lines.join("\n")
}

export function downloadJson(result: AskResult) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }))
  const link = document.createElement("a")
  link.href = url
  link.download = `askglp1-${result.run_id ?? "answer"}.json`
  link.click()
  URL.revokeObjectURL(url)
}
