import { Check, Copy, ExternalLink } from "lucide-react"
import { useState } from "react"

import type { Statement } from "@/api/client"
import { drugLabel, PLATFORM_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"

/** The sentence with every case-insensitive occurrence of `phrase` marked (the writer's wording, or a quote). */
function Marked({ text, phrase }: { text: string; phrase: string | null | undefined }) {
  const needle = phrase?.trim().toLowerCase()
  if (!needle || needle.length < 2) return <>{text}</>
  const parts: React.ReactNode[] = []
  const lower = text.toLowerCase()
  let from = 0
  let at = lower.indexOf(needle)
  while (at >= 0) {
    if (at > from) parts.push(text.slice(from, at))
    parts.push(
      <mark key={at} className="rounded-[2px] bg-tint text-ink shadow-[0_0_0_2px_var(--tint)] [box-decoration-break:clone]">
        {text.slice(at, at + needle.length)}
      </mark>
    )
    from = at + needle.length
    at = lower.indexOf(needle, from)
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
      title={`Statement id ${id}`}
      onClick={() =>
        void navigator.clipboard.writeText(id).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1200)
        })
      }
      className="inline-flex items-center gap-1 rounded-[4px] text-subtle hover:text-ink"
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Copied" : "Copy statement id"}
    </button>
  )
}

type Props = {
  statement: Statement
  /** Passage to mark instead of the writer's wording (an agent quote). */
  highlight?: string
  /** Leave out what the surrounding panel already says. */
  hideTerm?: boolean
  hidePlatform?: boolean
  /** "card" stands alone; "quote" sits in a list inside an answer. */
  variant?: "card" | "quote"
  className?: string
}

/** One patient statement: its source sentences and where it comes from. */
export function StatementView({ statement, highlight, hideTerm = false, hidePlatform = false, variant = "card", className }: Props) {
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
    <figure
      className={cn(
        "grid gap-1.5",
        variant === "card" ? "rounded-[10px] border border-hairline bg-card px-3 py-2.5" : "border-l-2 border-chip-border py-0.5 pl-3",
        className
      )}
    >
      <blockquote className="grid gap-1 text-[12.5px] leading-[1.55] text-foreground">
        {statement.evidence.map((sentence, index) => (
          <p key={index}>
            <Marked text={sentence} phrase={highlight ?? statement.wording} />
          </p>
        ))}
      </blockquote>
      <figcaption className="grid gap-1.5 text-[11.5px] text-muted-foreground">
        <span className="flex flex-wrap items-center gap-x-1.5">
          {meta.map((part, index) => (
            <span key={part} className={cn(index > 0 && "before:mr-1.5 before:text-subtle before:content-['·']", index === 0 && !hideTerm && "text-ink")}>
              {part}
            </span>
          ))}
        </span>
        {details.length > 0 && (
          <span className="flex flex-wrap gap-1" title="Details the extraction model read from the whole post or review">
            {details.map((detail) => (
              <span key={detail} className="rounded-full border border-hairline bg-card px-1.5 py-px text-[11px] text-prose">
                {detail}
              </span>
            ))}
          </span>
        )}
        <span className="flex flex-wrap items-center gap-x-3 text-[10.5px]">
          {url && (
            <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-0.5 text-ink underline-offset-2 hover:underline">
              Open the post on Reddit
              <ExternalLink className="size-3" aria-hidden />
            </a>
          )}
          <CopyId id={statement.edge_id} />
        </span>
      </figcaption>
    </figure>
  )
}
