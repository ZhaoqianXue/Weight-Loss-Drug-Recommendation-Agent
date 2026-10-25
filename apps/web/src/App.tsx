import { Info, Loader2, Moon, Sun } from "lucide-react"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { useGraphView, useOverview, usePatterns } from "@/api/queries"
import { BrandMark } from "@/components/brand-mark"
import { SectionHeading } from "@/components/layout"
import { useTheme } from "@/components/theme-provider"
import { AboutSection } from "@/features/about/AboutSection"
import { AskPanel } from "@/features/ask/AskPanel"
import { FaersSection } from "@/features/faers/FaersSection"
import { ExplorerToolbar } from "@/features/filters/ExplorerToolbar"
import { GraphCard } from "@/features/graph/GraphCard"
import { PatternsSection } from "@/features/patterns/PatternsSection"
import { useResolvedTheme } from "@/hooks/use-resolved-theme"
import type { Overview } from "@/api/client"
import { cn } from "@/lib/utils"
import {
  formatNumber,
  PLATFORM_LABEL,
  PLATFORM_UNIT,
  valueLabel,
} from "@/lib/format"
import {
  explorerFromSearch,
  explorerToSearch,
  INITIAL_EXPLORER,
  INITIAL_SIZES,
  toGraphQuery,
  type Explorer,
} from "@/state"

const ROLE_PHRASE = {
  both: "purposes and side effects",
  side_effect: "side effects",
  purpose: "purposes",
} as const

const SECTIONS = [
  { id: "graph", label: "Knowledge graph" },
  { id: "faers", label: "Compared with FAERS" },
  { id: "patterns", label: "Patterns" },
  { id: "about", label: "About the data" },
] as const

/** What the graph shows, including every filter in force, so a screenshot carries its own caption. */
function GraphMeta({
  explorer,
  overview,
}: {
  explorer: Explorer
  overview: Overview | undefined
}) {
  const soc = explorer.soc
    ? (overview?.socs.find((s) => s.code === explorer.soc)?.name ??
      explorer.soc)
    : null
  const filters = [
    explorer.drugs &&
      `${explorer.drugs.length} ${explorer.level === "brand" ? (explorer.drugs.length === 1 ? "brand" : "brands") : explorer.drugs.length === 1 ? "ingredient" : "ingredients"}: ${explorer.drugs.map(valueLabel).join(", ")}`,
    explorer.ageGroups &&
      `writers aged ${explorer.ageGroups.map(valueLabel).join(", ")}`,
    explorer.sexes &&
      `sex ${explorer.sexes.map((s) => valueLabel(s).toLowerCase()).join(", ")}`,
    soc && `${soc} only`,
  ].filter(Boolean)
  return (
    <>
      {PLATFORM_LABEL[explorer.platform]}: the{" "}
      {explorer.topN === 1 ? "" : `${explorer.topN} `}most reported{" "}
      {ROLE_PHRASE[explorer.show]} of each {explorer.level}, as MedDRA Preferred
      Terms
      {filters.length > 0 && (
        <span className="text-[#7a4b00] dark:text-[#f0c674]">
          {" "}
          · {filters.join(" · ")}
        </span>
      )}
    </>
  )
}

/** The section in view: the last one whose top has passed under the navigation bar. */
function useActiveSection(pane: React.RefObject<HTMLDivElement | null>) {
  const [active, setActive] = useState<string>(SECTIONS[0].id)
  useEffect(() => {
    const el = pane.current
    if (!el) return
    const update = () => {
      const top = el.getBoundingClientRect().top + 72
      const atEnd = el.scrollTop + el.clientHeight >= el.scrollHeight - 4
      let current: string = SECTIONS[0].id
      for (const { id } of SECTIONS) {
        const section = document.getElementById(id)
        if (section && section.getBoundingClientRect().top <= top) current = id
      }
      setActive(atEnd ? SECTIONS.at(-1)!.id : current)
    }
    update()
    el.addEventListener("scroll", update, { passive: true })
    return () => el.removeEventListener("scroll", update)
  }, [pane])
  return active
}

export function App() {
  const [explorer, setExplorer] = useState<Explorer>(() =>
    explorerFromSearch(window.location.search)
  )
  useEffect(() => {
    const search = explorerToSearch(explorer)
    if (search !== window.location.search)
      window.history.replaceState(
        null,
        "",
        `${window.location.pathname}${search}`
      )
  }, [explorer])
  const [sizes, setSizes] = useState(INITIAL_SIZES)
  const [askRequest, setAskRequest] = useState<{
    id: number
    question: string
  } | null>(null)
  const overview = useOverview()
  const query = useMemo(() => toGraphQuery(explorer), [explorer])
  // An empty multi-select means "none": nothing to draw, so no request.
  const blocked = [explorer.drugs, explorer.ageGroups, explorer.sexes].some(
    (choice) => choice !== null && choice.length === 0
  )
    ? "Select at least one value in each filter."
    : null
  const graph = useGraphView(query, blocked === null)
  const patterns = usePatterns()
  const mode = useResolvedTheme()
  const { setTheme } = useTheme()
  const nextAsk = useRef(1)
  const pane = useRef<HTMLDivElement>(null)
  const active = useActiveSection(pane)

  const updateExplorer = useCallback(
    (update: Partial<Explorer>) =>
      setExplorer((current) => ({ ...current, ...update })),
    []
  )
  const ask = useCallback(
    (question: string) => setAskRequest({ id: nextAsk.current++, question }),
    []
  )
  const graphSection = useRef<HTMLElement>(null)
  // An answer's "Show in graph": apply what it counted and bring the graph into view.
  const showInGraph = useCallback((update: Partial<Explorer>) => {
    setExplorer({ ...INITIAL_EXPLORER, ...update })
    requestAnimationFrame(() =>
      graphSection.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      })
    )
  }, [])

  const data = overview.data
  const view = graph.data
  const meta = data
    ? [
        ...data.platforms.map((p) => ({
          value: formatNumber(p.records_processed),
          label:
            p.platform === "reddit"
              ? "Reddit posts and comments"
              : "WebMD reviews",
        })),
        { value: formatNumber(data.faers_reports), label: "FAERS reports" },
        {
          value: data.versions.meddra ?? "–",
          label: "MedDRA version",
          before: "MedDRA",
        },
      ]
    : []
  const summary = view?.summary ?? []
  const excluded = summary.filter(
    (s) => (s.records_before_age_sex_filter ?? 0) > (s.universe_records ?? 0)
  )
  const uncoded = summary.filter(
    (s) => (s.universe_records_with_uncoded_mention ?? 0) > 0
  )
  const sourceText =
    explorer.platform === "reddit" ? "posts and comments" : "reviews"

  return (
    <div className="flex min-h-svh flex-col bg-background lg:h-svh lg:overflow-hidden">
      <header className="shrink-0 bg-masthead text-masthead-fg dark:border-b dark:border-masthead-line">
        <div className="flex flex-wrap items-center gap-x-7 gap-y-2.5 px-4 py-3.5 sm:px-[22px]">
          <div className="flex items-center gap-2.5">
            <BrandMark onMasthead className="size-[30px]" />
            <h1 className="font-display text-[26px] leading-[1.15] font-normal tracking-[-0.015em]">
              AskGLP1
            </h1>
            <span className="hidden text-[13px] text-masthead-muted min-[1600px]:inline">
              What people report about GLP-1 receptor agonists on Reddit and
              WebMD, next to FAERS
            </span>
          </div>
          <ul className="order-last grid basis-full grid-cols-2 gap-x-4 gap-y-1 text-[12.5px] text-masthead-muted sm:flex sm:flex-wrap sm:items-center sm:gap-x-0 sm:gap-y-1.5 xl:order-none xl:ml-auto xl:basis-auto">
            {data ? (
              meta.map((item) => (
                <li
                  key={item.label}
                  className="sm:border-l sm:border-masthead-line sm:px-3.5 sm:first:border-l-0 sm:first:pl-0"
                >
                  {"before" in item ? (
                    <>
                      {item.before}{" "}
                      <span className="font-mono text-masthead-fg">
                        {item.value}
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="font-mono text-masthead-fg">
                        {item.value}
                      </span>{" "}
                      {item.label}
                    </>
                  )}
                </li>
              ))
            ) : (
              <li className="border-l border-masthead-line px-3.5 max-xl:border-l-0 max-xl:pl-0">
                {overview.error
                  ? overview.error.message
                  : "Connecting to the knowledge graph…"}
              </li>
            )}
          </ul>
          <button
            type="button"
            aria-label={
              mode === "dark" ? "Switch to light mode" : "Switch to dark mode"
            }
            onClick={() => setTheme(mode === "dark" ? "light" : "dark")}
            className="ml-auto inline-flex size-8 items-center justify-center rounded-lg border border-masthead-line text-masthead-muted hover:text-masthead-fg xl:ml-0"
          >
            {mode === "dark" ? (
              <Sun className="size-4" />
            ) : (
              <Moon className="size-4" />
            )}
          </button>
        </div>
      </header>

      {/*
        App shell after ReviewPilot: the content pane fills the width and scrolls on its own, and the agent is docked on the
        right at full height (30% of the window, 380 to 560px). Below lg the page stacks, with the agent under the graph.
      */}
      <main className="flex min-h-0 flex-1 flex-col gap-8 pt-5 pb-10 lg:flex-row lg:gap-0 lg:p-0">
        <div
          ref={pane}
          className="contents lg:block lg:min-w-0 lg:flex-1 lg:[scrollbar-gutter:stable] lg:overflow-y-auto lg:px-[22px] lg:pb-10"
        >
          {/* Sections of the page; the one in view is underlined. */}
          <nav
            aria-label="Sections"
            className="sticky top-0 z-40 -mx-[22px] hidden h-10 items-stretch gap-6 border-b border-hairline bg-background/95 px-[22px] backdrop-blur-sm lg:flex"
          >
            {SECTIONS.map((s) => (
              <a
                key={s.id}
                href={`#${s.id}`}
                aria-current={active === s.id ? "location" : undefined}
                onClick={(event) => {
                  event.preventDefault()
                  document
                    .getElementById(s.id)
                    ?.scrollIntoView({ behavior: "smooth", block: "start" })
                }}
                className={cn(
                  "-mb-px inline-flex items-center border-b-2 border-transparent text-[13px] text-muted-foreground transition-colors hover:text-ink",
                  active === s.id && "border-ink font-medium text-ink"
                )}
              >
                {s.label}
              </a>
            ))}
          </nav>
          <div className="contents lg:block lg:space-y-10 lg:pt-4">
            <section
              id="graph"
              ref={graphSection}
              aria-label="Knowledge graph"
              className="order-1 grid scroll-mt-14 gap-2.5 px-4 sm:px-[22px] lg:px-0"
            >
              <SectionHeading
                title={
                  <>
                    Knowledge graph
                    {graph.isFetching && (
                      <Loader2
                        className="size-4 animate-spin text-muted-foreground"
                        aria-label="Updating"
                      />
                    )}
                  </>
                }
                meta={<GraphMeta explorer={explorer} overview={data} />}
              />
              <div
                role="region"
                aria-label="Knowledge graph explorer"
                className="@container flex min-w-0 flex-col overflow-hidden rounded-xl border border-border bg-card"
              >
                <ExplorerToolbar
                  overview={data}
                  state={explorer}
                  onChange={updateExplorer}
                  onReset={() => setExplorer(INITIAL_EXPLORER)}
                />
                <GraphCard
                  view={view}
                  fetching={graph.isFetching}
                  error={graph.error}
                  blocked={blocked}
                  filters={{
                    ageGroups: explorer.ageGroups,
                    sexes: explorer.sexes,
                    soc: explorer.soc,
                  }}
                  sizes={sizes}
                  onSizesChange={setSizes}
                  onAsk={ask}
                />
              </div>
              <p className="flex gap-1.5 text-[12.5px] text-prose">
                <Info
                  className="mt-px size-3.5 shrink-0 text-ink"
                  aria-hidden
                />
                <span>
                  A line's share is the drug's {sourceText} reporting the term,
                  each once, out of its {sourceText} with any{" "}
                  {explorer.show === "purpose"
                    ? "purpose"
                    : explorer.show === "side_effect"
                      ? "side effect"
                      : "statement of that kind"}
                  : what writers chose to report, not how often effects occur.
                  {uncoded.length > 0 && (
                    <>
                      {" "}
                      Not drawn: mentions with no adequate MedDRA term (in{" "}
                      {uncoded
                        .map(
                          (s) =>
                            `${formatNumber(s.universe_records_with_uncoded_mention)} of ${formatNumber(s.universe_records)} ${s.target === "purpose" ? "purpose" : "side-effect"} ${PLATFORM_UNIT[explorer.platform]}`
                        )
                        .join(" and ")}
                      ).
                    </>
                  )}
                  {excluded.map((s) => (
                    <span key={`x-${s.target}`}>
                      {" "}
                      The age and sex filters keep{" "}
                      {formatNumber(s.universe_records)} of{" "}
                      {formatNumber(s.records_before_age_sex_filter)}{" "}
                      {s.target === "purpose" ? "purpose" : "side-effect"}{" "}
                      {PLATFORM_UNIT[explorer.platform]}
                      {(s.excluded_missing_age ?? 0) > 0 &&
                        `; ${formatNumber(s.excluded_missing_age)} have no stated age`}
                      {(s.excluded_missing_sex ?? 0) > 0 &&
                        `; ${formatNumber(s.excluded_missing_sex)} have no stated sex`}
                      .
                    </span>
                  ))}
                </span>
              </p>
            </section>

            <div
              id="faers"
              className="order-3 scroll-mt-14 px-4 sm:px-[22px] lg:px-0"
            >
              <FaersSection overview={data} onAsk={ask} />
            </div>

            <div
              id="patterns"
              className="order-3 scroll-mt-14 px-4 sm:px-[22px] lg:px-0"
            >
              <PatternsSection
                patterns={patterns.data}
                ageGroups={data?.age_groups ?? []}
              />
            </div>

            <div
              id="about"
              className="order-3 scroll-mt-14 px-4 sm:px-[22px] lg:px-0"
            >
              {data && <AboutSection overview={data} />}
            </div>
          </div>
        </div>

        <aside
          aria-label="AskGLP1 agent"
          className="order-2 mx-4 h-[680px] overflow-hidden rounded-xl border border-border sm:mx-[22px] lg:mx-0 lg:h-auto lg:w-[clamp(380px,30vw,560px)] lg:shrink-0 lg:rounded-none lg:border-y-0 lg:border-r-0"
        >
          <AskPanel request={askRequest} onShowInGraph={showInGraph} />
        </aside>
      </main>
    </div>
  )
}

export default App
