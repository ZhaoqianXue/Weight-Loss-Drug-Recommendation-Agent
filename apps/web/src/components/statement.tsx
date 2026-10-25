import { Check, Copy, ExternalLink } from "lucide-react"
import { useState } from "react"

import type { Statement } from "@/api/client"
import { drugLabel, PLATFORM_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"

// Case and curly quotes do not matter when finding a phrase; the replacements keep every index in place.
const fold = (text: string) => text.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"')

/** The sentence with every occurrence of each phrase marked (the writer's wording, or the quotes taken from it). */
function Marked({ text, phrases, color }: { text: string; phrases: (string | null | undefined)[]; color: string }) {
  const lower = fold(text)
  const spans: [number, number][] = []
  for (const phrase of phrases) {
    // A quote may end with punctuation the sentence does not repeat.
    const full = fold(phrase?.trim() ?? "")
    const needle = lower.includes(full) ? full : full.replace(/[.!?…]+$/, "")
    // Nothing to point at when the phrase is (nearly) the whole sentence.
    if (needle.length < 2 || needle.length >= lower.trim().length * 0.9) continue
    for (let at = lower.indexOf(needle); at >= 0; at = lower.indexOf(needle, at + needle.length)) spans.push([at, at + needle.length])
  }
  if (spans.length === 0) return <>{text}</>
  spans.sort((a, b) => a[0] - b[0])
  const merged: [number, number][] = []
  for (const [start, end] of spans) {
    const last = merged.at(-1)
    if (last && start <= last[1]) last[1] = Math.max(last[1], end)
    else merged.push([start, end])
  }
  const parts: React.ReactNode[] = []
  let from = 0
  for (const [start, end] of merged) {
    if (start > from) parts.push(text.slice(from, start))
    parts.push(
      <mark
        key={start}
        className="rounded-[2px] text-foreground [box-decoration-break:clone]"
        style={{ background: `color-mix(in oklab, ${color} 38%, transparent)`, boxShadow: `0 0 0 1.5px color-mix(in oklab, ${color} 38%, transparent)` }}
      >
        {text.slice(start, end)}
      </mark>
    )
    from = end
  }
  parts.push(text.slice(from))
  return <>{parts}</>
}

/** A Reddit post opens on reddit.com; a comment carries only its own id, which is not enough for a link. */
function sourceUrl(statement: Statement) {
  const id = statement.record_id ?? ""
  return statement.platform === "reddit" && id.startsWith("t3_") ? `https://www.reddit.com/comments/${id.slice(3)}/` : null
}

// "2.5mg" and "2.5 mg" read the same; first letters are capitalised.
const tidy = (value: string) => valueLabel(value.replace(/(\d)\s*mg\b/gi, "$1 mg"))

function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      type="button"
      title={copied ? "Copied" : `Copy the statement id (${id})`}
      aria-label={copied ? "Copied" : "Copy the statement id"}
      onClick={() =>
        void navigator.clipboard.writeText(id).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1200)
        })
      }
      className="inline-flex size-5 items-center justify-center rounded-[4px] text-subtle hover:bg-tint hover:text-ink"
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
    </button>
  )
}

type Props = {
  statement: Statement
  /** Passages to mark instead of the writer's wording (the agent's quotes from this statement). */
  highlight?: string | string[]
  /** Leave out what the surrounding panel already says. */
  hideTerm?: boolean
  hidePlatform?: boolean
  /** "card" stands alone; "quote" sits in a list inside an answer. */
  variant?: "card" | "quote"
  /** Color of the rule beside the sentences and of the marked wording (the node color of the term's role). */
  accent?: string
  className?: string
}

/** One patient statement: its source sentences and where it comes from. */
export function StatementView({ statement, highlight, hideTerm = false, hidePlatform = false, variant = "card", accent = "#9bb8d8", className }: Props) {
  const drug =
    statement.brand && statement.brand !== "unspecified"
      ? statement.brand
      : statement.ingredient
        ? `${drugLabel("ingredient", statement.ingredient)}${statement.ingredient_basis === "community" ? " (from the community)" : ""}`
        : null
  const writer = [statement.writer_age_group, statement.writer_sex].filter((v) => v && v !== "not stated").map((v) => valueLabel(v))
  const details = [
    statement.intensity && `Intensity: ${tidy(statement.intensity)}`,
    statement.impact && `Impact: ${tidy(statement.impact)}`,
    ...statement.action_taken.map((action) => `Action: ${tidy(action)}`),
    statement.dose && `Dose: ${tidy(statement.dose)}`,
    statement.duration && `Duration: ${tidy(statement.duration)}`,
  ].filter(Boolean) as string[]
  const meta = [
    !hideTerm && (statement.pt ?? "No MedDRA term"),
    drug && valueLabel(drug),
    !hidePlatform && (PLATFORM_LABEL[statement.platform ?? ""] ?? statement.platform),
    statement.date,
    writer.length > 0 && `Writer: ${writer.join(", ")}`,
  ].filter(Boolean) as string[]
  const url = sourceUrl(statement)
  return (
    <figure className={cn("grid gap-1.5 border-l-2 py-0.5 pl-3", variant === "card" && "pb-1", className)} style={{ borderColor: accent }}>
      {statement.evidence.length > 0 ? (
        <blockquote className="grid gap-1 text-[13px] leading-[1.55] text-foreground">
          {statement.evidence.map((sentence, index) => (
            <p key={index}>
              <Marked text={sentence} phrases={highlight === undefined ? [statement.wording] : [highlight].flat()} color={accent} />
            </p>
          ))}
        </blockquote>
      ) : (
        // The graph does not show what writers aged 13-18 wrote; the statement still counts.
        <p className="text-[12.5px] text-muted-foreground italic" title={statement.evidence_withheld ?? undefined}>
          {statement.evidence_withheld ? "Sentences not shown: the writer is aged 13–18." : "No source sentences."}
        </p>
      )}
      <figcaption className="grid gap-1.5 text-[12px] text-muted-foreground">
        <span className="flex items-start gap-2">
          <span className="flex min-w-0 flex-1 flex-wrap items-center gap-x-1.5">
            {meta.map((part, index) => (
              <span key={part} className={cn(index > 0 && "before:mr-1.5 before:text-subtle before:content-['·']", index === 0 && !hideTerm && "text-ink")}>
                {part}
              </span>
            ))}
          </span>
          {/* Where the statement comes from: the Reddit post when it has one, and the statement id. */}
          <span className="-my-0.5 flex shrink-0 items-center gap-0.5">
            {url && (
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                title="Open the post on Reddit"
                aria-label="Open the post on Reddit"
                className="inline-flex size-5 items-center justify-center rounded-[4px] text-subtle hover:bg-tint hover:text-ink"
              >
                <ExternalLink className="size-3" aria-hidden />
              </a>
            )}
            <CopyId id={statement.edge_id} />
          </span>
        </span>
        {details.length > 0 && (
          <span className="flex flex-wrap gap-1" title="Details the extraction model read from the whole post or review">
            {details.map((detail) => (
              <span key={detail} className="rounded-full border border-hairline bg-card px-1.5 py-px text-[11.5px] text-prose">
                {detail}
              </span>
            ))}
          </span>
        )}
      </figcaption>
    </figure>
  )
}
