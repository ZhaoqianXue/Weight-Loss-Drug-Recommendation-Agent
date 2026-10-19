import { BadgeCheck, Check, ChevronRight, CircleAlert, Copy, FileJson, Network } from "lucide-react"
import { useState } from "react"
import Markdown from "react-markdown"
import remarkGfm from "remark-gfm"

import type { AskResult, Citation } from "@/api/client"
import { StatementView } from "@/components/statement"
import { Button } from "@/components/ui/button"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { PLATFORM_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"
import { explorerFromCall, omittedFilters, type Explorer } from "@/state"

import { answerMarkdown, downloadJson } from "./export"
import { toolTitle } from "./progress"

const TYPE_LABEL = {
  answered: { text: "Answered from the data", className: "text-ink" },
  limited: { text: "Partly answerable from the data", className: "text-[#7a4b00] dark:text-[#f0c674]" },
  declined: { text: "Not answered: medical advice, or outside the data", className: "text-prose" },
} as const

// Row fields worth showing in a citation card, in reading order, with the names a reader knows.
const FIELD_LABEL: Record<string, string> = {
  platform: "Platform",
  scope: "Drug scope",
  brand: "Brand",
  ingredient: "Ingredient",
  age_group: "Age group",
  sex: "Sex",
  pt: "MedDRA PT",
  concept: "Term",
  common_wordings: "Common wordings",
  name: "Term",
  hlt: "HLT",
  hlgt: "HLGT",
  soc: "SOC",
  intensity: "Intensity",
  impact: "Impact",
  action_taken: "Action taken",
  records: "Records",
  denominator: "Out of",
  percent: "Share (%)",
  faers_reports: "FAERS reports",
  faers_denominator: "FAERS reports with a clinical-event term",
  faers_percent: "FAERS share (%)",
  reddit_records: "Reddit records",
  reddit_denominator: "Reddit records with a clinical-event side effect",
  reddit_percent: "Reddit share (%)",
  webmd_records: "WebMD reviews",
  webmd_denominator: "WebMD reviews with a clinical-event side effect",
  webmd_percent: "WebMD share (%)",
  records_reddit: "Reddit records",
  records_webmd: "WebMD reviews",
  universe_records: "Records in scope",
  excluded_missing_age: "Left out, age not stated",
  excluded_missing_sex: "Left out, sex not stated",
  records_processed: "Records processed",
  records_with_side_effect: "With a side effect",
  records_with_purpose: "With a purpose",
  // Counted before any age or sex filter, and for any stated value (sex "other" included).
  records_with_age_stated: "Age stated (any value)",
  records_with_sex_stated: "Sex stated (any value)",
  universe_records_with_profile_purpose: "With a WebMD profile condition",
  record_window: "Dates",
  code: "MedDRA code",
  pt_code: "MedDRA code",
  sparse: "Denominator under 10",
  small: "Denominator under 30",
  reddit_small: "Reddit group under 30",
  webmd_small: "WebMD group under 30",
}

function display(key: string, value: unknown) {
  if (value === null || value === undefined) return "–"
  if (key === "platform") return PLATFORM_LABEL[String(value)] ?? String(value)
  if (typeof value === "number") return key.endsWith("percent") ? `${value}%` : value.toLocaleString()
  if (typeof value === "boolean") return value ? "yes" : "no"
  if (Array.isArray(value)) return value.map((v) => valueLabel(String(v))).join(", ")
  return valueLabel(String(value))
}

/** "Graph query 3" for tool call t3, "Graph overview" for t0. */
export function callName(call: string) {
  return call === "t0" ? "Graph overview" : `Graph query ${call.replace(/^t/, "")}`
}

/** A footnote-style marker after a number; hovering or focusing it shows the tool row the number comes from. */
function CitationMark({ citation, number, children }: { citation: Citation | undefined; number: number; children?: React.ReactNode }) {
  if (!citation) return <>{children}<sup className="text-destructive">?</sup></>
  const source = citation.source ?? {}
  // Flags are shown only when they apply; empty values are left out.
  const fields = Object.keys(FIELD_LABEL).filter((key) => key in source && source[key] !== false && source[key] !== null && source[key] !== "")
  const label = `Source of this number: ${callName(citation.call)}, row ${citation.row}, ${FIELD_LABEL[citation.field] ?? citation.field}`
  return (
    <HoverCard openDelay={60} closeDelay={80}>
      <HoverCardTrigger asChild>
        {children ? (
          // The cited number itself is the link to its source row; an inline span, so it never breaks from "(" or "of".
          <span
            tabIndex={0}
            aria-label={`${String(children)}. ${label}`}
            className="cursor-help rounded-[3px] font-medium text-foreground underline decoration-[#9bb8d8] decoration-dotted decoration-[1.5px] underline-offset-[3px] outline-none hover:bg-card hover:text-ink hover:decoration-ink focus-visible:bg-card focus-visible:ring-2 focus-visible:ring-ring/50 dark:decoration-[#6f96bf]"
          >
            {children}
          </span>
        ) : (
          <button
            type="button"
            className="relative -top-[0.45em] ml-px rounded-[3px] px-[2px] font-mono text-[10px] leading-none text-[#5b7aa3] outline-none hover:bg-card hover:text-ink focus-visible:bg-card focus-visible:text-ink focus-visible:ring-2 focus-visible:ring-ring/50 dark:text-[#9bb8d8]"
            aria-label={label}
          >
            {number}
          </button>
        )}
      </HoverCardTrigger>
      <HoverCardContent align="start" collisionPadding={12} className="w-80 p-0 text-[12.5px]">
        <div className="border-b border-hairline px-3 py-2">
          <p className="label-caps">
            {callName(citation.call)} · row {citation.row}
          </p>
          <p className="mt-0.5 font-medium text-ink">{citation.tool ? toolTitle(citation.tool, citation.arguments) : "Size of the graph"}</p>
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 px-3 py-2">
          {fields.map((key) => (
            <div key={key} className={cn("contents", key === citation.field && "[&>*]:font-medium [&>*]:text-ink")}>
              <dt className="text-muted-foreground">{FIELD_LABEL[key]}</dt>
              <dd className="text-right tabular-nums">{display(key, source[key])}</dd>
            </div>
          ))}
        </dl>
        <p className="border-t border-hairline px-3 py-1.5 text-meta text-muted-foreground">
          The cited value, <span className="font-medium text-ink">{display(citation.field, Number.isNaN(Number(citation.value)) ? citation.value : Number(citation.value))}</span>,
          was checked against this row before the answer was shown.
        </p>
      </HoverCardContent>
    </HoverCard>
  )
}

/** Thousands separators for integers the agent wrote without them (only before a citation tag, so years and doses stay). */
function readable(text: string) {
  return text.replace(/(?<![\d.,])(\d{4,})(?=\s*\[c\d+\])/g, (n) => Number(n).toLocaleString("en-US"))
}

// A number right before a citation tag: "1,207", "32.4%", "<0.1%", "$0.01".
const CITED_NUMBER = /((?:[<>≤≥~]\s?)?\$?\d[\d,]*(?:\.\d+)?%?)\s*\[(c\d+)\]/g

/** Answer text in which each cited number links to its source row; a tag after anything else becomes a small marker. */
function AnswerText({ result }: { result: AskResult }) {
  const byId = new Map(result.citations.map((c) => [c.id, c]))
  const linked = readable(result.answer).replace(CITED_NUMBER, "[$1](#cite-$2)")
  // Remaining markers are numbered in reading order; the agent's own tag ids stay in the copied text and the JSON record.
  const order = new Map<string, number>()
  for (const match of linked.matchAll(/\[(c\d+)\]/g)) if (!order.has(match[1])) order.set(match[1], order.size + 1)
  const text = linked.replace(/\s*\[(c\d+)\]/g, "[$1](#cite-$1)")
  return (
    <div className="answer-prose">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith("#cite-")) {
              const id = href.slice(6)
              const marker = /^c\d+$/.test(String(children))
              return (
                <CitationMark citation={byId.get(id)} number={order.get(id) ?? 0}>
                  {marker ? undefined : children}
                </CitationMark>
              )
            }
            return (
              <a href={href} target="_blank" rel="noreferrer">
                {children}
              </a>
            )
          },
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={label} onClick={onClick} className="size-7 rounded-[7px] text-muted-foreground hover:text-ink">
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function Disclosure({ title, children, defaultOpen = false }: { title: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="group border-t border-hairline pt-2.5" open={defaultOpen}>
      <summary className="flex cursor-pointer list-none items-center gap-1 text-meta text-ink hover:underline">
        <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
        {title}
      </summary>
      <div className="mt-2 grid gap-2">{children}</div>
    </details>
  )
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`

export function AnswerCard({ result, onShowInGraph }: { result: AskResult; onShowInGraph: (update: Partial<Explorer>) => void }) {
  const type = TYPE_LABEL[result.answer_type ?? "answered"]
  // The first counting call that worked is what the answer is about.
  const graphCall = result.tool_calls.find((call) => !call.error && explorerFromCall(call.tool, call.arguments))
  const graphView = graphCall ? explorerFromCall(graphCall.tool, graphCall.arguments) : null
  const omitted = graphCall ? omittedFilters(graphCall.arguments) : []
  // What the graph view differs in from the answer's query.
  const otherFilters = omitted.filter((o) => o !== "term")
  const graphNote = [
    omitted.includes("term") && "shows these drugs' top terms",
    otherFilters.length > 0 && `without the ${otherFilters.join(" and ")} filter`,
  ]
    .filter(Boolean)
    .join(", ")
  const checked = [
    result.citations.length > 0 && plural(result.citations.length, "number"),
    result.quotes.length > 0 && plural(result.quotes.length, "quote"),
  ].filter(Boolean) as string[]
  const queries = result.tool_calls.filter((call) => !call.error).length
  const [copied, setCopied] = useState(false)
  const copy = () =>
    void navigator.clipboard.writeText(answerMarkdown(result)).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  return (
    <div className="grid gap-3">
      <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px]">
        <span className={cn("font-medium", type.className)}>{type.text}</span>
        {result.verified
          ? checked.length > 0 && (
              <span className="inline-flex items-center gap-1 text-[#2f6b45] dark:text-[#9fd8b0]" title="Every cited number matches its graph row and every quote matches its source sentences">
                <BadgeCheck className="size-3.5" />
                {checked.join(" and ")} checked against the graph
              </span>
            )
          : (
              <span className="inline-flex items-center gap-1 text-[#8a1f1f] dark:text-[#f4b4b4]">
                <CircleAlert className="size-3.5" />
                Not verified: treat the numbers with care
              </span>
            )}
      </p>
      <AnswerText result={result} />
      {!result.verified && result.problems.length > 0 && (
        <ul className="grid gap-0.5 rounded-lg border border-[#f4b4b4] bg-[#fff5f5] px-3 py-2 text-meta text-[#8a1f1f] dark:border-[#8a1f1f] dark:bg-transparent dark:text-[#f4b4b4]">
          {result.problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        {graphView && result.answer_type !== "declined" && (
          <>
            <Button variant="outline" size="sm" className="h-7 gap-1.5 rounded-full bg-card px-2.5 text-[12.5px]" onClick={() => onShowInGraph(graphView)}>
              <Network className="size-3.5" />
              Show in graph
            </Button>
            {graphNote && <span className="text-meta text-muted-foreground">{graphNote}</span>}
          </>
        )}
        <span className="ml-auto flex items-center">
          <IconAction label={copied ? "Copied" : "Copy as Markdown, with sources"} onClick={copy}>
            {copied ? <Check /> : <Copy />}
          </IconAction>
          <IconAction label="Download the full answer record (JSON)" onClick={() => downloadJson(result)}>
            <FileJson />
          </IconAction>
        </span>
      </div>
      {result.quotes.length > 0 && (
        <Disclosure title={`Quoted statements (${result.quotes.length})`} defaultOpen>
          {result.quotes.map((quote) =>
            quote.statement ? (
              <StatementView key={`${quote.edge_id}-${quote.text}`} statement={quote.statement} highlight={quote.text} variant="quote" />
            ) : (
              <p key={quote.edge_id} className="text-meta text-muted-foreground">
                “{quote.text}” ({quote.edge_id})
              </p>
            )
          )}
        </Disclosure>
      )}
      {result.limitations.length > 0 && (
        <Disclosure title="Limits of this answer">
          <ul className="grid list-disc gap-1 pl-4 text-meta text-prose">
            {result.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </Disclosure>
      )}
      {result.tool_calls.length > 0 && (
      <Disclosure title={`How it was answered (${queries} graph ${queries === 1 ? "query" : "queries"})`}>
        <ol className="grid gap-1.5 text-meta">
          {result.tool_calls.map((call) => (
            <li key={call.call} className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5">
              <span className="font-mono text-subtle">{call.call.replace(/^t/, "")}.</span>
              <span className={cn("text-prose", call.error && "text-muted-foreground")}>
                {toolTitle(call.tool, call.arguments)}
                {call.error ? <span className="text-muted-foreground"> · rejected, retried</span> : <span className="text-muted-foreground"> · {plural(call.rows, "row")}</span>}
              </span>
              {call.error && <span className="col-start-2 text-muted-foreground">{call.error}</span>}
            </li>
          ))}
        </ol>
        <p className="font-mono text-[11px] text-subtle">
          {result.model} ({result.reasoning_effort} reasoning) · {result.agent_version} · {result.seconds ?? "–"} s · ${result.cost_usd.toFixed(4)}
          {result.repair_used && " · revised once after a failed check"} · run {result.run_id}
        </p>
      </Disclosure>
      )}
    </div>
  )
}
