import { Info, Loader2, MessageSquareText } from "lucide-react"
import { useState } from "react"

import type { FaersQuery, FaersRow, Overview } from "@/api/client"
import { useFaers } from "@/api/queries"
import { SectionHeading } from "@/components/layout"
import { Segmented } from "@/components/segmented"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { formatNumber, formatPercent, scopeLabel, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"

const ALL_AGES = "__all__"
const SOURCES = [
  { key: "faers", label: "FAERS reports", short: "FAERS", color: "var(--ink)" },
  { key: "reddit", label: "Reddit records", short: "Reddit", color: "#6f96bf" },
  { key: "webmd", label: "WebMD reviews", short: "WebMD", color: "#9bb8d8" },
] as const
type Source = (typeof SOURCES)[number]["key"]

// The comparison rules applied by the graph tool, in plain words.
const RULES = [
  "FAERS counts reports in the validated primary configuration: a study drug is the suspect or interacting drug and its identity was validated. Reports are counted from the drug's FDA approval to 2026-06-30.",
  "Only clinical-event terms are compared. Product issues, medication errors, off-label use and misuse, therapeutic responses and interactions, overdoses and underdoses, and pregnancy exposures are left out on every side.",
  "A FAERS report does not establish that the drug caused the reaction; Reddit and WebMD shares come from what writers chose to post. The sources are shown side by side, never combined or divided.",
  "FAERS age is the age at reaction onset; Reddit and WebMD age is the writer's stated age. FAERS has no “other” sex.",
  "Ingredient scopes cover every product of the ingredient (on Reddit, also references assigned from a single-drug community); brand scopes need the brand named.",
]

function share(row: FaersRow, key: Source) {
  const count = key === "faers" ? row.faers_reports : key === "reddit" ? row.reddit_records : row.webmd_records
  // The agent flags a Reddit or WebMD denominator under 30 as small; FAERS denominators are not flagged.
  const small = key === "reddit" ? row.reddit_small : key === "webmd" ? row.webmd_small : false
  return { count, percent: row[`${key}_percent`], denominator: row[`${key}_denominator`], small: Boolean(small) }
}

function Bar({ percent, scale, color, count }: { percent?: number | null; scale: number; color: string; count?: number | null }) {
  const width = count ? Math.max(1, (100 * (percent ?? 0)) / scale) : 0
  return (
    <div className="mt-1 h-1 rounded-full bg-hairline">
      <div className="h-1 rounded-full" style={{ width: `${Math.min(100, width)}%`, background: color }} />
    </div>
  )
}

export function FaersSection({ overview, onAsk }: { overview: Overview | undefined; onAsk: (question: string) => void }) {
  const [query, setQuery] = useState<FaersQuery>({ scope: "all", rank_by: "faers", top_n: 12, sex: null, age_group: null })
  const faers = useFaers(query)
  const data = faers.data
  const rows: FaersRow[] = data?.rows ?? []
  // One scale for the three sources, so equal bar lengths mean equal shares.
  const scale = Math.max(1, ...rows.flatMap((r) => [r.faers_percent ?? 0, r.reddit_percent ?? 0, r.webmd_percent ?? 0]))
  const summary = (data?.summary ?? {}) as Record<string, number | string | null>
  const n = (key: Source) => summary[`${key}_denominator`] as number | null
  const update = (change: Partial<FaersQuery>) => setQuery((current) => ({ ...current, ...change }))
  const scopeText = query.scope === "all" ? "GLP-1 drugs" : scopeLabel(query.scope)
  const ask = (row: FaersRow) => onAsk(`How does ${row.concept} with ${scopeText} compare between Reddit, WebMD and FAERS reports?`)
  // Each column has one denominator, so the first row says whether that source is a small group here.
  const smallSources = rows.length ? SOURCES.filter((s) => share(rows[0], s.key).small) : []
  const excluded = (field: "sex" | "age") =>
    SOURCES.map((s) => ({ source: s.short, count: summary[`${s.key}_excluded_missing_${field}`] as number | null })).filter((e) => (e.count ?? 0) > 0)

  return (
    <section aria-label="Compared with FAERS" className="grid gap-2.5">
      <SectionHeading
        title={
          <>
            Compared with FAERS
            {faers.isFetching && <Loader2 className="size-4 animate-spin text-muted-foreground" aria-label="Updating" />}
          </>
        }
        meta="The same MedDRA terms in FDA spontaneous reports, Reddit posts and comments, and WebMD reviews, each as a share of its own source."
      />
      <div className="relative overflow-hidden rounded-xl border border-border bg-card">
        {faers.isFetching && data && (
          <span role="status" className="absolute top-[68px] left-1/2 z-10 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-meta text-ink">
            <Loader2 className="size-3.5 animate-spin" />
            Updating the comparison…
          </span>
        )}
        <div role="group" aria-label="FAERS comparison filters" className="flex flex-wrap items-center gap-2 border-b border-border bg-info px-3 py-2.5">
          <Select value={query.scope} onValueChange={(scope) => update({ scope })}>
            <SelectTrigger aria-label="Drug" className="h-[34px] gap-1.5 pr-[9px] pl-[11px]">
              <span className="text-meta text-muted-foreground">Drug</span>
              <SelectValue className="font-medium text-ink">{scopeLabel(query.scope)}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All four ingredients</SelectItem>
              <SelectGroup>
                <SelectLabel>Ingredients</SelectLabel>
                {(overview?.ingredients ?? []).map((ingredient) => (
                  <SelectItem key={ingredient} value={`ingredient:${ingredient}`}>
                    {valueLabel(ingredient)}
                  </SelectItem>
                ))}
              </SelectGroup>
              <SelectGroup>
                <SelectLabel>Brands</SelectLabel>
                {(overview?.brands ?? []).map(({ brand }) => (
                  <SelectItem key={brand} value={`brand:${brand}`}>
                    {brand}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
          <Segmented
            label="Rank terms by"
            value={query.rank_by ?? "faers"}
            onChange={(rank_by) => update({ rank_by })}
            options={[
              { value: "faers", label: "Top in FAERS" },
              { value: "reddit", label: "Top in Reddit" },
              { value: "webmd", label: "Top in WebMD" },
            ]}
          />
          <Segmented
            label="Sex"
            value={query.sex ?? "all"}
            onChange={(sex) => update({ sex: sex === "all" ? null : sex })}
            options={[
              { value: "all", label: "Both sexes" },
              { value: "female", label: "Female" },
              { value: "male", label: "Male" },
            ]}
          />
          <Select value={query.age_group ?? ALL_AGES} onValueChange={(age) => update({ age_group: age === ALL_AGES ? null : age })}>
            <SelectTrigger aria-label="Age group" className="h-[34px] gap-1.5 pr-[9px] pl-[11px]">
              <span className="text-meta text-muted-foreground">Age</span>
              <SelectValue className="font-medium text-ink">{query.age_group ?? "All"}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_AGES}>All ages</SelectItem>
              {(overview?.age_groups ?? []).map((age) => (
                <SelectItem key={age} value={age}>
                  {age}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Wide screens: a table with one column per source. */}
        <div className={cn("hidden transition-opacity sm:block", faers.isFetching && data && "opacity-50")}>
          <table className="w-full border-collapse text-left">
            <thead>
              <tr className="border-b border-hairline text-meta text-muted-foreground">
                <th scope="col" className="px-3 py-2 font-normal">
                  MedDRA Preferred Term
                </th>
                {SOURCES.map((source) => (
                  <th key={source.key} scope="col" className="w-[24%] px-3 py-2 font-normal">
                    <span className="flex items-center gap-1.5">
                      <span className="size-2 rounded-full" style={{ background: source.color }} />
                      {source.label}
                      <span className="ml-auto font-mono text-[10.5px] text-subtle" title="Denominator of the shares in this column">
                        n {formatNumber(n(source.key))}
                      </span>
                    </span>
                  </th>
                ))}
                <th className="w-8" aria-label="Ask" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.code ?? row.concept} className="group border-b border-hairline last:border-0 hover:bg-info">
                  <th scope="row" className="px-3 py-2 text-left text-[12.5px] font-normal text-foreground">
                    {row.concept}
                  </th>
                  {SOURCES.map((source) => {
                    const v = share(row, source.key)
                    return (
                      <td key={source.key} className="px-3 py-2 align-middle">
                        <div className="flex items-baseline justify-between gap-2 tabular-nums">
                          <span className={cn("text-[12.5px] text-foreground", v.small && "text-muted-foreground")}>{formatPercent(v.percent, v.count)}</span>
                          <span className="text-meta text-muted-foreground">{formatNumber(v.count)}</span>
                        </div>
                        <Bar percent={v.percent} count={v.count} scale={scale} color={source.color} />
                      </td>
                    )
                  })}
                  <td className="pr-2">
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label={`Ask the agent about ${row.concept}`}
                      title={`Ask the agent about ${row.concept}`}
                      className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                      onClick={() => ask(row)}
                    >
                      <MessageSquareText />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Phones: one block per term, the three sources stacked. */}
        <ol className={cn("grid transition-opacity sm:hidden", faers.isFetching && data && "opacity-50")}>
          {rows.map((row) => (
            <li key={row.code ?? row.concept} className="grid gap-1.5 border-b border-hairline px-3 py-2.5 last:border-0">
              <button type="button" className="text-left text-[13px] text-foreground" onClick={() => ask(row)}>
                {row.concept}
              </button>
              {SOURCES.map((source) => {
                const v = share(row, source.key)
                return (
                  <div key={source.key} className="grid grid-cols-[64px_minmax(0,1fr)_auto] items-center gap-2 text-meta">
                    <span className="text-muted-foreground">{source.short}</span>
                    <Bar percent={v.percent} count={v.count} scale={scale} color={source.color} />
                    <span className="text-right tabular-nums">
                      {formatPercent(v.percent, v.count)} · {formatNumber(v.count)}
                    </span>
                  </div>
                )
              })}
            </li>
          ))}
        </ol>
        {faers.isLoading && <p className="px-3 py-6 text-center text-muted-foreground">Reading the FAERS reports…</p>}
      </div>

      {data && (
        <div className="grid gap-1.5 text-[12px] text-prose">
          <p className="flex gap-1.5">
            <Info className="mt-px size-3.5 shrink-0 text-ink" aria-hidden />
            <span>
              Each n counts the reports, records or reviews with at least one clinical-event side effect, dated {data.window}. The FAERS scope holds{" "}
              {formatNumber(summary.faers_reports_in_scope as number)} reports in all; reports that list only non-clinical terms are not in its n.
              Reddit and WebMD count side effects only, so weight loss stated as a goal is not counted here, and some FAERS terms may describe the
              condition being treated (for example, Blood glucose increased).
            </span>
          </p>
          {(query.sex || query.age_group) && (
            <p className="pl-5 text-[#7a4b00] dark:text-[#f0c674]">
              {[
                excluded("sex").length > 0 && `The sex filter leaves out ${excluded("sex").map((e) => `${formatNumber(e.count)} ${e.source}`).join(", ")} with no stated sex.`,
                excluded("age").length > 0 && `The age filter leaves out ${excluded("age").map((e) => `${formatNumber(e.count)} ${e.source}`).join(", ")} with no stated age.`,
              ]
                .filter(Boolean)
                .join(" ")}{" "}
              Reddit age and sex are self-disclosed, so what remains there is a small, self-selected subset.
            </p>
          )}
          {smallSources.length > 0 && (
            <p className="pl-5 text-[#7a4b00] dark:text-[#f0c674]">
              {smallSources.length === 1
                ? `The ${smallSources[0].short} n is under 30 here: a small group, so its shares (in grey) are unstable.`
                : `The ${smallSources.map((s) => s.short).join(" and ")} n are under 30 here: small groups, so their shares (in grey) are unstable.`}
            </p>
          )}
          <details className="group pl-5">
            <summary className="cursor-pointer list-none text-ink hover:underline">Comparison rules</summary>
            <ul className="mt-1.5 grid list-disc gap-1 pl-4 text-muted-foreground">
              {RULES.map((rule) => (
                <li key={rule}>{rule}</li>
              ))}
            </ul>
          </details>
        </div>
      )}
      {faers.error && <p className="text-destructive">{faers.error.message}</p>}
    </section>
  )
}
