import type { Overview } from "@/api/client"
import { SectionHeading } from "@/components/layout"
import { formatNumber } from "@/lib/format"

function Item({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[110px_minmax(0,1fr)] gap-3 border-b border-hairline py-2">
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="text-prose">{children}</dd>
    </div>
  )
}

const day = (value: string | null | undefined) => (value ? value.slice(0, 10) : "–")

// How to read every number on the page; these restate the counting rules the graph tools apply.
const READING = [
  "Counts are records: a Reddit post or comment, or a WebMD review. They are not people, and one person can write several.",
  "A term counts once per record. Shares are out of the records that report any side effect (or any purpose), so they show what writers chose to mention, not incidence, prevalence or risk.",
  "Reddit and WebMD are separate sources and are never added together.",
  "A brand counts only where the writer names it. An ingredient can also be assigned from a Reddit community about a single GLP-1 drug.",
  "Reddit age and sex are known only where writers state them about themselves; WebMD takes them from the reviewer profile.",
  "Mentions without an adequate MedDRA term are kept out of term counts.",
  "FAERS holds spontaneous reports from patients, clinicians and manufacturers. Its shares are shown next to the Reddit and WebMD shares and never combined with or divided by them.",
  "Term names follow MedDRA, which uses British spelling (for example, Diarrhoea).",
]

/** Sources, versions and how to read the numbers, from the graph and build manifests. */
export function AboutSection({ overview }: { overview: Overview }) {
  const v = overview.versions
  const reddit = overview.platforms.find((p) => p.platform === "reddit")
  const webmd = overview.platforms.find((p) => p.platform === "webmd")
  const release = /openFDA release ([\d-]+)/.exec(overview.faers_status)
  return (
    <section aria-label="About the data" className="grid gap-3">
      <SectionHeading title="About the data" meta="Sources, versions and the rules behind every number on this page." />
      <div className="grid gap-x-10 gap-y-6 rounded-xl border border-border bg-card px-5 py-4 xl:grid-cols-2">
        <div className="grid content-start gap-2">
          <h3 className="label-caps">Sources and versions</h3>
          <dl className="text-[13px]">
            <Item term="Drugs">
              {overview.ingredients.map((ingredient, index) => (
                <span key={ingredient}>
                  {index > 0 && "; "}
                  {ingredient.charAt(0).toUpperCase() + ingredient.slice(1)} (
                  {overview.brands
                    .filter((b) => b.ingredient === ingredient)
                    .map((b) => b.brand)
                    .join(", ")}
                  ){ingredient === "tirzepatide" && ", a dual GIP/GLP-1 receptor agonist"}
                </span>
              ))}
            </Item>
            {reddit && (
              <Item term="Reddit">
                {formatNumber(reddit.records_processed)} posts and comments analysed; {formatNumber(reddit.records_with_side_effect)} report a
                side effect and {formatNumber(reddit.records_with_purpose)} a purpose
              </Item>
            )}
            {webmd && (
              <Item term="WebMD">
                {formatNumber(webmd.records_processed)} reviews; {formatNumber(webmd.records_with_side_effect)} report a side effect and{" "}
                {formatNumber(webmd.records_with_purpose)} a purpose
              </Item>
            )}
            <Item term="FAERS">
              {formatNumber(overview.faers_reports)} reports in which a study drug is the suspect or interacting drug and its identity was
              validated (the primary configuration)
              {release && (
                <>
                  , openFDA release <span className="whitespace-nowrap">{release[1]}</span>
                </>
              )}
            </Item>
            <Item term="Extraction">
              <span className="whitespace-nowrap">{v.extraction_model ?? "–"}</span> with{" "}
              <span className="whitespace-nowrap">{v.extraction_prompt ?? "–"}</span>
            </Item>
            <Item term="Terminology">MedDRA {v.meddra ?? "–"}: Preferred Terms (PTs) and their system organ class (SOC)</Item>
            <Item term="Graph">
              <span className="whitespace-nowrap">{v.kg ?? "–"}</span>, built <span className="whitespace-nowrap">{day(v.kg_built)}</span>
            </Item>
            <Item term="Agent">
              <span className="whitespace-nowrap">{v.agent ?? "–"}</span>, {v.agent_model ?? "–"}
            </Item>
          </dl>
        </div>
        <div className="grid content-start gap-2">
          <h3 className="label-caps">How to read the numbers</h3>
          <ul className="grid gap-0 text-[13px] text-prose">
            {READING.map((line) => (
              <li key={line} className="border-b border-hairline py-2">
                {line}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  )
}
