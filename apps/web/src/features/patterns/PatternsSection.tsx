import type { PatternRow, PatternSection, Patterns, PopulationCell } from "@/api/client"
import { SectionHeading } from "@/components/layout"
import { Skeleton } from "@/components/ui/skeleton"
import { drugColor, PURPOSE_COLOR, SIDE_EFFECT_COLOR } from "@/features/graph/style"
import { drugLabel, formatNumber, formatPercent, PLATFORM_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"

const PLATFORMS = ["reddit", "webmd"] as const
const SEXES = ["female", "male", "other", "not stated"]
// Cards side by side have similar lengths: ten-row lists together, five-row lists together.
const ORDER = ["side_effects", "purposes", "socs", "brands", "ingredients", "impacts"]
// Groups that are not a concept ("no brand named", "not stated") go last and are muted.
const RESIDUAL = new Set(["unspecified", "not stated"])

const CARD: Record<string, { title: string; unit: string; note?: string }> = {
  side_effects: { title: "Side effects", unit: "% of records with a side effect" },
  purposes: {
    title: "Purposes",
    unit: "% of records with a purpose",
    note: "One intent can map to several terms: weight loss to Weight decreased, Weight control or Obesity; diabetes to Diabetes mellitus, Type 2 diabetes mellitus or Diabetes mellitus management. WebMD purposes include the reviewer's profile condition.",
  },
  socs: { title: "Side effects by system organ class", unit: "% of records with a side effect" },
  brands: { title: "Brands named", unit: "% of records with a side effect", note: "A record can name several brands, so shares add up to more than 100%." },
  ingredients: {
    title: "Ingredients",
    unit: "% of records with a side effect",
    note: "Includes Reddit references assigned from a single-drug community. A record can mention several drugs.",
  },
  impacts: { title: "Reported impact of side effects", unit: "% of records with a side effect" },
}

function barColor(section: string, label: string) {
  if (section === "purposes") return PURPOSE_COLOR
  if (section === "ingredients" || section === "brands") return drugColor(label)
  if (section === "impacts") return "var(--ink)"
  return SIDE_EFFECT_COLOR
}

function label(section: string, value: string) {
  if (section === "ingredients") return drugLabel("ingredient", value)
  if (section === "brands") return drugLabel("brand", value)
  return valueLabel(value)
}

function Ranked({ section, rows, scale }: { section: string; rows: PatternRow[]; scale: number }) {
  const ordered = [...rows].sort(
    (a, b) => Number(RESIDUAL.has(a.label)) - Number(RESIDUAL.has(b.label)) || (b.percent ?? 0) - (a.percent ?? 0)
  )
  if (rows.length === 0) return <p className="text-meta text-muted-foreground">No records.</p>
  return (
    <ol className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-2">
      {ordered.map((row) => {
        const residual = RESIDUAL.has(row.label)
        const width = row.records ? Math.max(1.5, (100 * (row.percent ?? 0)) / scale) : 0
        return (
          <li key={row.label} className={cn("grid min-w-0 grid-cols-[minmax(0,1fr)] gap-1", residual && "text-muted-foreground")}>
            <div className="flex min-w-0 items-baseline justify-between gap-3">
              <span className="min-w-0 truncate" title={label(section, row.label)}>
                {label(section, row.label)}
              </span>
              <span className="shrink-0 text-meta text-muted-foreground tabular-nums">
                <span className={residual ? undefined : "text-foreground"}>{formatPercent(row.percent, row.records)}</span> · {formatNumber(row.records)}
              </span>
            </div>
            <div className="h-1 rounded-full bg-hairline">
              <div
                className="h-1 rounded-full"
                style={{ width: `${Math.min(100, width)}%`, background: residual ? "var(--chip-border)" : barColor(section, row.label) }}
              />
            </div>
          </li>
        )
      })}
    </ol>
  )
}

function SectionCard({ section }: { section: PatternSection }) {
  const card = CARD[section.key] ?? { title: section.title, unit: section.unit }
  // Impact is mostly not stated: show the stated values and give the not-stated share in the column header.
  const hideNotStated = section.key === "impacts"
  const rows = (platform: string) => section.rows.filter((r) => r.platform === platform && !(hideNotStated && r.label === "not stated"))
  // One scale for both sources in a card, so equal bars mean equal shares.
  const scale = Math.max(1, ...section.rows.filter((r) => !RESIDUAL.has(r.label)).map((r) => r.percent ?? 0))
  return (
    <div className="grid content-start gap-3 rounded-xl border border-border bg-card px-4 py-3.5">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="label-caps">{card.title}</h3>
        <span className="text-[11px] text-muted-foreground">{card.unit}</span>
      </div>
      <div className="grid gap-x-6 gap-y-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {PLATFORMS.map((platform) => {
          const all = section.rows.filter((r) => r.platform === platform)
          const notStated = all.find((r) => r.label === "not stated")
          return (
            <div key={platform} className="grid min-w-0 content-start gap-2">
              <p className="flex items-baseline justify-between gap-2 border-b border-hairline pb-1 text-meta">
                <span className="font-medium text-foreground">{PLATFORM_LABEL[platform]}</span>
                <span className="font-mono text-[10.5px] text-subtle" title="Denominator of the shares in this column">
                  {hideNotStated && notStated && <>not stated {formatPercent(notStated.percent)} · </>}n {formatNumber(all[0]?.denominator)}
                </span>
              </p>
              <Ranked section={section.key} rows={rows(platform)} scale={scale} />
            </div>
          )
        })}
      </div>
      {card.note && <p className="text-meta text-muted-foreground">{card.note}</p>}
    </div>
  )
}

function PopulationCard({ cells, summary, ageGroups }: { cells: PopulationCell[]; summary: Patterns["population_summary"]; ageGroups: string[] }) {
  const ages = [...ageGroups, "not stated"]
  return (
    <div className="grid content-start gap-3 rounded-xl border border-border bg-card px-4 py-3.5 xl:col-span-2">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="label-caps">Side-effect records by the writer's stated age and sex</h3>
        <span className="text-[11px] text-muted-foreground">records</span>
      </div>
      <div className="grid gap-x-10 gap-y-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        {PLATFORMS.map((platform) => {
          const mine = cells.filter((c) => c.platform === platform)
          const known = mine.filter((c) => c.age_group !== "not stated" && c.sex !== "not stated")
          const max = Math.max(1, ...known.map((c) => c.records))
          const value = (age: string, sex: string) => mine.find((c) => c.age_group === age && c.sex === sex)?.records ?? 0
          const s = summary.find((x) => x.platform === platform)
          return (
            <div key={platform} className="grid min-w-0 content-start gap-2">
              <p className="flex items-baseline justify-between gap-2 border-b border-hairline pb-1 text-meta">
                <span className="font-medium text-foreground">{PLATFORM_LABEL[platform]}</span>
                {s && (
                  <span className="text-[11px] text-muted-foreground">
                    both stated in {formatNumber(s.age_and_sex_stated)} of {formatNumber(s.records)}
                  </span>
                )}
              </p>
              <table className="w-full table-fixed border-separate border-spacing-[2px] text-[11px] tabular-nums">
                <thead>
                  <tr className="text-muted-foreground">
                    <th className="w-20 text-left font-normal" />
                    {SEXES.map((sex) => (
                      <th key={sex} scope="col" className="font-normal">
                        {valueLabel(sex)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ages.map((age) => (
                    <tr key={age}>
                      <th scope="row" className="pr-1 text-left font-normal whitespace-nowrap text-muted-foreground">
                        {valueLabel(age)}
                      </th>
                      {SEXES.map((sex) => {
                        const n = value(age, sex)
                        const stated = age !== "not stated" && sex !== "not stated"
                        const strength = stated ? Math.sqrt(n / max) : 0
                        return (
                          <td
                            key={sex}
                            className={cn("rounded-[4px] px-1 py-[3px] text-center", !stated && "border border-hairline text-muted-foreground")}
                            style={
                              stated
                                ? {
                                    background: `color-mix(in oklab, var(--ink) ${Math.round(strength * 70)}%, var(--info))`,
                                    color: strength > 0.55 ? "var(--primary-foreground)" : undefined,
                                  }
                                : undefined
                            }
                            title={`${PLATFORM_LABEL[platform]}, ${valueLabel(age)}, ${valueLabel(sex)}: ${formatNumber(n)} records`}
                          >
                            {n ? formatNumber(n) : "0"}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        })}
      </div>
      <p className="text-meta text-muted-foreground">
        Shaded cells have both age and sex stated; outlined cells miss one or both. Reddit age and sex are known only where writers state them
        about themselves, so the shaded Reddit cells are a small, self-selected subset; WebMD takes both from the reviewer profile.
      </p>
    </div>
  )
}

export function PatternsSection({ patterns, ageGroups }: { patterns: Patterns | undefined; ageGroups: string[] }) {
  const sections = patterns ? [...patterns.sections].sort((a, b) => ORDER.indexOf(a.key) - ORDER.indexOf(b.key)) : []
  return (
    <section aria-label="Patterns" className="grid gap-3">
      <SectionHeading title="Patterns" meta="The most reported terms in each source, over the whole graph." />
      {patterns ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {sections.map((section) => (
            <SectionCard key={section.key} section={section} />
          ))}
          <PopulationCard cells={patterns.population} summary={patterns.population_summary} ageGroups={ageGroups} />
        </div>
      ) : (
        <div className="grid gap-4 xl:grid-cols-2">
          {Array.from({ length: 4 }, (_, index) => (
            <Skeleton key={index} className="h-72 rounded-xl bg-info" />
          ))}
        </div>
      )}
    </section>
  )
}
