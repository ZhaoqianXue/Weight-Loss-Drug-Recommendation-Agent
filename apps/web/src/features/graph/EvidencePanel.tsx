import { Loader2, MessageSquareText, X } from "lucide-react"

import type { Platform } from "@/api/client"
import { useEvidence } from "@/api/queries"
import { StatementView } from "@/components/statement"
import { Button } from "@/components/ui/button"
import { formatNumber, PLATFORM_LABEL, PLATFORM_UNIT, ROLE_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"
import type { Explorer, Level, Selection } from "@/state"

type Props = {
  selection: Selection
  platform: Platform
  level: Level
  filters: Pick<Explorer, "ageGroups" | "sexes" | "soc">
  onClose: () => void
  onAsk: (question: string) => void
  className?: string
}

const SOURCE_TEXT: Record<string, string> = { reddit: "Reddit posts and comments", webmd: "WebMD reviews" }

/** The question the agent gets for a node or link, worded around what writers report. */
function question(selection: Selection, platform: Platform) {
  const where = SOURCE_TEXT[platform]
  const drug = selection.drug ? valueLabel(selection.drug) : null
  if (!selection.code) return `Which side effects and purposes do ${where} report for ${drug}?`
  if (selection.target === "purpose")
    return drug
      ? `In how many ${where} do writers say they use ${drug} for ${selection.label}, and how do they describe it?`
      : `Which drugs do writers of ${where} say they use for ${selection.label}?`
  return drug
    ? `How often do ${where} report ${selection.label} with ${drug}, and how do writers describe it?`
    : `How often do ${where} report ${selection.label} for each drug?`
}

/** Example statements behind a graph node or link, under the same filters as the graph, and the way into the agent. */
export function EvidencePanel({ selection, platform, level, filters, onClose, onAsk, className }: Props) {
  const evidence = useEvidence({
    target: selection.target,
    platform,
    level,
    drug: selection.drug,
    code: selection.code,
    limit: 6,
    age_groups: filters.ageGroups,
    sexes: filters.sexes,
    soc: selection.code ? null : filters.soc,
  })
  const data = evidence.data
  const filtered = filters.ageGroups !== null || filters.sexes !== null
  return (
    <aside aria-label="Example statements" className={cn("z-30 flex flex-col", className)}>
      <div className="flex items-start gap-2 border-b border-hairline px-4 py-3">
        <div className="grid min-w-0 flex-1 gap-0.5">
          <p className="label-caps">
            {selection.code ? ROLE_LABEL[selection.target] : "Drug"} · {PLATFORM_LABEL[platform]}
          </p>
          <h3 className="font-display text-[19px] leading-[1.2] text-ink">
            {selection.drug && selection.code ? `${valueLabel(selection.drug)} → ${selection.label}` : valueLabel(selection.label)}
          </h3>
          {data && (
            <p className="text-meta text-muted-foreground">
              {formatNumber(data.matching_records)} {PLATFORM_UNIT[platform]} report{" "}
              {selection.code ? selection.label : `a ${ROLE_LABEL[selection.target].toLowerCase()}`}
              {selection.drug ? ` for ${valueLabel(selection.drug)}` : level === "brand" ? " with a named brand" : ""}
              {filtered ? " among writers matching the age and sex filters" : ""}. {data.statements.length} examples follow, one per{" "}
              {platform === "reddit" ? "post or comment" : "review"}.
            </p>
          )}
        </div>
        <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
          <X />
        </Button>
      </div>
      <div className="grid min-h-0 flex-1 content-start gap-2.5 overflow-y-auto px-4 py-3">
        {evidence.isLoading && (
          <p className="flex items-center gap-2 text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" /> Reading statements…
          </p>
        )}
        {evidence.error && <p className="text-destructive">{evidence.error.message}</p>}
        {data?.statements.map((statement) => (
          <StatementView key={statement.edge_id} statement={statement} hideTerm={Boolean(selection.code)} hidePlatform />
        ))}
        {data && data.statements.length === 0 && <p className="text-muted-foreground">No statements match.</p>}
      </div>
      <div className="border-t border-hairline px-4 py-3">
        <Button className="w-full" onClick={() => onAsk(question(selection, platform))}>
          <MessageSquareText />
          Ask the agent about {selection.code ? selection.label : valueLabel(selection.label)}
        </Button>
      </div>
    </aside>
  )
}
