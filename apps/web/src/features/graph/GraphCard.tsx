import { CircleHelp, ImageDown, Loader2, Maximize2, Minus, Plus, SlidersHorizontal } from "lucide-react"
import { Fragment, useCallback, useRef, useState } from "react"

import type { GraphView } from "@/api/client"
import { Segmented } from "@/components/segmented"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useElementSize } from "@/hooks/use-element-size"
import { useResolvedTheme } from "@/hooks/use-resolved-theme"
import { formatNumber, formatPercent, PLATFORM_UNIT, ROLE_LABEL, valueLabel } from "@/lib/format"
import { cn } from "@/lib/utils"
import type { Explorer, NodeSizes, Selection } from "@/state"

import { EvidencePanel } from "./EvidencePanel"
import { GraphCanvas, type GraphCanvasHandle, type Hover } from "./GraphCanvas"
import { GraphTooltip } from "./GraphTooltip"
import { drugColor, PURPOSE_COLOR, SIDE_EFFECT_COLOR, type GraphMode } from "./style"

type Props = {
  view: GraphView | undefined
  /** A new view is being computed; the current one stays visible, dimmed. */
  fetching: boolean
  error: Error | null
  blocked: string | null
  /** The explorer's population and SOC filters, so example statements match the view. */
  filters: Pick<Explorer, "ageGroups" | "sexes" | "soc">
  sizes: NodeSizes
  onSizesChange: (sizes: NodeSizes) => void
  onAsk: (question: string) => void
}

type Picked = Selection & { id: string }

const ICON_BUTTON = "size-7 rounded-[7px] text-ink hover:text-ink"

function ToolbarButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label={label} onClick={onClick} className={ICON_BUTTON}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function SizeSlider({ id, label, value, onChange }: { id: string; label: string; value: number; onChange: (value: number) => void }) {
  return (
    <div className="grid gap-2">
      <div className="flex justify-between text-meta">
        <Label htmlFor={id} className="text-meta font-normal text-prose">
          {label}
        </Label>
        <span className="font-mono text-muted-foreground">{value}</span>
      </div>
      <Slider id={id} min={1} max={10} step={1} value={[value]} onValueChange={([next]) => onChange(next)} />
    </div>
  )
}

function NodeSizesButton({ sizes, onChange }: { sizes: NodeSizes; onChange: (sizes: NodeSizes) => void }) {
  return (
    <Popover>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <Button variant="ghost" size="icon-sm" aria-label="Node sizes" className={ICON_BUTTON}>
              <SlidersHorizontal />
            </Button>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent>Node sizes</TooltipContent>
      </Tooltip>
      <PopoverContent align="end" className="grid w-64 gap-4 p-4">
        <SizeSlider id="size-drug" label="Drug" value={sizes.drug} onChange={(drug) => onChange({ ...sizes, drug })} />
        <SizeSlider id="size-purpose" label="Purpose" value={sizes.purpose} onChange={(purpose) => onChange({ ...sizes, purpose })} />
        <SizeSlider id="size-effect" label="Side effect" value={sizes.side_effect} onChange={(side_effect) => onChange({ ...sizes, side_effect })} />
      </PopoverContent>
    </Popover>
  )
}

/**
 * The view as a matrix: terms (grouped by role, most reported first) by drugs, each cell the drug's share. A blank cell
 * means the term is not among that drug's top terms here, not that nobody reports it.
 */
function LinkTable({ view, selected, onSelect }: { view: GraphView; selected: string | null; onSelect: (s: Picked) => void }) {
  const drugs = view.nodes.filter((n) => n.kind === "drug")
  const byPair = new Map(view.links.map((l) => [`${l.source}->${l.target}`, l]))
  const roles = (["purpose", "side_effect"] as const).filter((role) => view.nodes.some((n) => n.kind === role))
  const unit = PLATFORM_UNIT[view.platform]
  return (
    <div className="absolute inset-0 overflow-auto bg-card">
      <table className="w-full min-w-[640px] border-collapse text-left text-[13px]">
        <thead className="sticky top-0 z-10 bg-card shadow-[0_1px_0_var(--border)]">
          <tr>
            <th scope="col" className="px-4 py-2.5 text-meta font-normal text-muted-foreground">
              MedDRA Preferred Term
            </th>
            {drugs.map((drug) => (
              <th key={drug.id} scope="col" className="px-1.5 py-2.5 text-center font-medium whitespace-nowrap text-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <span className="size-2 rounded-full" style={{ background: drugColor(drug.label) }} />
                  {valueLabel(drug.label)}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        {roles.map((role) => {
          const terms = view.nodes.filter((n) => n.kind === role).sort((a, b) => (b.records[role] ?? 0) - (a.records[role] ?? 0))
          const max = Math.max(1, ...view.links.filter((l) => l.kind === role).map((l) => l.percent ?? 0))
          return (
            <tbody key={role}>
              <tr className="border-t border-border bg-info text-meta">
                <th scope="rowgroup" className="px-4 py-1.5 font-medium text-prose">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="size-2 rounded-full" style={{ background: role === "purpose" ? PURPOSE_COLOR : SIDE_EFFECT_COLOR }} />
                    <span>
                      {role === "purpose" ? "Purposes" : "Side effects"}
                      <span className="font-normal text-muted-foreground">: share of the drug's {unit} with a {ROLE_LABEL[role].toLowerCase()}</span>
                    </span>
                  </span>
                </th>
                {drugs.map((drug) => (
                  <td key={drug.id} className="px-1.5 py-1.5 text-center font-mono text-[11px] text-subtle" title={`${valueLabel(drug.label)} ${unit} with a ${ROLE_LABEL[role].toLowerCase()}`}>
                    n {formatNumber(drug.records[role])}
                  </td>
                ))}
              </tr>
              {terms.map((term) => {
                const termId = term.id
                return (
                  <tr key={term.id} className="border-t border-hairline">
                    <th scope="row" className="px-4 py-1 font-normal">
                      <button
                        type="button"
                        onClick={() => onSelect({ id: termId, target: role, code: term.code ?? null, label: term.label, drug: null })}
                        className={cn("rounded-[4px] text-left text-foreground outline-none hover:text-ink hover:underline focus-visible:ring-2 focus-visible:ring-ring/50", selected === termId && "font-medium text-ink")}
                      >
                        {term.label}
                      </button>
                    </th>
                    {drugs.map((drug) => {
                      const link = byPair.get(`${drug.id}->${term.id}`)
                      if (!link) return <td key={drug.id} className="px-1.5 py-1 text-center text-subtle" aria-label="Not among this drug's top terms">·</td>
                      const id = `${drug.id}->${term.id}`
                      const strength = (link.percent ?? 0) / max
                      return (
                        <td key={drug.id} className="px-1 py-[3px]">
                          <button
                            type="button"
                            title={`${valueLabel(drug.label)} → ${term.label}: ${formatNumber(link.records)} of ${formatNumber(link.denominator)} ${unit}${link.sparse ? ", too few to compare" : link.small ? ", a small group" : ""}`}
                            onClick={() => onSelect({ id, target: role, code: term.code ?? null, label: term.label, drug: drug.label })}
                            className={cn(
                              "w-full rounded-[6px] px-1.5 py-1 text-center tabular-nums outline-none hover:ring-1 hover:ring-ring focus-visible:ring-2 focus-visible:ring-ring",
                              selected === id && "ring-2 ring-ink",
                              (link.sparse || link.small) && "italic"
                            )}
                            style={{
                              background: `color-mix(in oklab, var(--ink) ${Math.round(strength * 58)}%, var(--card))`,
                              color: strength > 0.62 ? "var(--primary-foreground)" : undefined,
                            }}
                          >
                            {formatPercent(link.percent, link.records)}
                          </button>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          )
        })}
      </table>
    </div>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-[5px]">
      <span className="size-2 rounded-full" style={{ background: color }} />
      {label}
    </span>
  )
}

/** The canvas (or table) with the evidence panel beside it, then one bar for the view switch, legend and view controls. */
export function GraphCard({ view, fetching, error, blocked, filters, sizes, onSizesChange, onAsk }: Props) {
  const canvas = useRef<GraphCanvasHandle>(null)
  const [zoom, setZoom] = useState(1)
  const [hover, setHover] = useState<Hover | null>(null)
  const [selection, setSelection] = useState<Picked | null>(null)
  const [chosenLayout, setLayout] = useState<"graph" | "table" | null>(null)
  // On a narrow card the graph's labels cannot fit, so the table is the default there.
  const area = useRef<HTMLDivElement>(null)
  const { width: areaWidth } = useElementSize(area)
  const layout = chosenLayout ?? (areaWidth > 0 && areaWidth < 640 ? "table" : "graph")
  const mode: GraphMode = useResolvedTheme()
  const onZoom = useCallback((scale: number) => setZoom(scale), [])
  const onHover = useCallback((next: Hover | null) => setHover(next), [])
  const onSelect = useCallback((next: Picked) => setSelection((current) => (current?.id === next.id ? null : next)), [])

  const nodes = view && !blocked ? view.nodes : []
  const links = view && !blocked ? view.links : []
  const unit = PLATFORM_UNIT[view?.platform ?? "reddit"]
  const empty = blocked ?? (view && links.length === 0 ? "No statements match these filters." : null)
  const kinds = new Set(nodes.map((n) => n.kind))

  return (
    <>
      {/* Tall enough that heading, filters, canvas and bar fill the first screen, within limits. */}
      <div ref={area} className="relative flex h-[clamp(440px,calc(100svh-372px),880px)] overflow-hidden">
        {fetching && view && (
          <span role="status" className="absolute top-3 left-1/2 z-40 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-meta text-ink">
            <Loader2 className="size-3.5 animate-spin" />
            Updating the graph…
          </span>
        )}
        <div
          className={cn("relative min-w-0 flex-1 transition-opacity", fetching && view && "opacity-50")}
        >
          {error ? (
            <p className="p-4 text-destructive">The graph could not be loaded: {error.message}</p>
          ) : layout === "table" && view && !blocked ? (
            <LinkTable view={view} selected={selection?.id ?? null} onSelect={onSelect} />
          ) : (
            <GraphCanvas
              ref={canvas}
              nodes={nodes}
              links={links}
              sizes={sizes}
              mode={mode}
              selected={selection?.id ?? null}
              drugTitle={view?.level === "ingredient" ? "Ingredients" : "Brands"}
              onZoom={onZoom}
              onHover={onHover}
              onSelect={onSelect}
            />
          )}
          {empty && <p className="pointer-events-none absolute inset-x-0 top-1/2 text-center text-muted-foreground">{empty}</p>}
          {hover && layout === "graph" && <GraphTooltip hover={hover} unit={unit} />}
        </div>
        {/* Beside the graph, which re-lays out narrower so the selected node stays in view; over it on narrow cards. */}
        {selection && view && (
          <EvidencePanel
            selection={selection}
            platform={view.platform}
            level={view.level}
            filters={filters}
            onClose={() => setSelection(null)}
            onAsk={onAsk}
            className={cn("w-[min(400px,100%)] shrink-0 border-l border-border bg-card", areaWidth < 640 && "absolute inset-y-0 right-0")}
          />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-border bg-info px-3 py-2 text-meta text-prose">
        <Segmented
          label="Show the graph as"
          value={layout}
          onChange={setLayout}
          options={[
            { value: "graph", label: "Graph" },
            { value: "table", label: "Table" },
          ]}
        />
        {layout === "graph" && (
          <Fragment>
            {kinds.has("purpose") && <LegendDot color={PURPOSE_COLOR} label="Purpose" />}
            {kinds.has("side_effect") && <LegendDot color={SIDE_EFFECT_COLOR} label="Side effect" />}
            <span className="text-muted-foreground">
              Circle: {unit} · Line: share of the drug's {unit}
            </span>
            <div role="toolbar" aria-label="Graph view" className="ml-auto flex items-center gap-0.5">
              <ToolbarButton label="Zoom out" onClick={() => canvas.current?.zoomBy(0.7)}>
                <Minus />
              </ToolbarButton>
              <span className="w-10 text-center font-mono text-[11.5px] text-ink">{Math.round(zoom * 100)}%</span>
              <ToolbarButton label="Zoom in" onClick={() => canvas.current?.zoomBy(1.3)}>
                <Plus />
              </ToolbarButton>
              <ToolbarButton label="Reset zoom" onClick={() => canvas.current?.resetZoom()}>
                <Maximize2 />
              </ToolbarButton>
              <span className="mx-0.5 h-4 w-px bg-border" aria-hidden />
              <NodeSizesButton sizes={sizes} onChange={onSizesChange} />
              <ToolbarButton label="Save the graph as PNG" onClick={() => void canvas.current?.exportPng("askglp1_knowledge_graph.png")}>
                <ImageDown />
              </ToolbarButton>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" aria-label="How to use the graph" className={ICON_BUTTON}>
                    <CircleHelp />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top" align="end" className="max-w-64">
                  Terms are MedDRA Preferred Terms. Circle size grows with the number of {unit} (compare within a column); line width is the
                  share of the drug's {unit} that report the term. Hover a node to see its shares, click a node or a line to read example
                  statements, drag a node to move it, and pinch or hold Ctrl and scroll to zoom.
                </TooltipContent>
              </Tooltip>
            </div>
          </Fragment>
        )}
        {layout === "table" && view && (
          <span className="text-muted-foreground">
            Darker cells are larger shares; italics mark a group under 30. A dot: not among that drug's top {view.top_n}. Click a cell or a term to read
            example statements.
          </span>
        )}
      </div>
    </>
  )
}
