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
import { PURPOSE_COLOR, SIDE_EFFECT_COLOR, type GraphMode } from "./style"

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

/** The same links as a table, grouped by drug, largest first within each drug and role. */
function LinkTable({ view, selected, onSelect }: { view: GraphView; selected: string | null; onSelect: (s: Picked) => void }) {
  const nodes = new Map(view.nodes.map((n) => [n.id, n]))
  const order = new Map(view.nodes.filter((n) => n.kind === "drug").map((n, i) => [n.id, i]))
  const rows = [...view.links].sort(
    (a, b) => (order.get(a.source) ?? 0) - (order.get(b.source) ?? 0) || a.kind.localeCompare(b.kind) || b.records - a.records
  )
  const unit = PLATFORM_UNIT[view.platform]
  return (
    <div className="absolute inset-0 overflow-auto bg-card">
      <table className="w-full border-collapse text-left text-[12.5px]">
        <thead className="sticky top-0 z-10 bg-card text-meta text-muted-foreground shadow-[0_1px_0_var(--hairline)]">
          <tr>
            <th scope="col" className="px-4 py-2 font-normal">
              {view.level === "brand" ? "Brand" : "Ingredient"}
            </th>
            <th scope="col" className="hidden px-3 py-2 font-normal sm:table-cell">
              Role
            </th>
            <th scope="col" className="px-3 py-2 font-normal">
              MedDRA Preferred Term
            </th>
            <th scope="col" className="hidden px-3 py-2 font-normal md:table-cell">
              System organ class
            </th>
            <th scope="col" className="px-3 py-2 text-right font-normal">
              {unit === "records" ? "Records" : "Reviews"}
            </th>
            <th scope="col" className="hidden px-3 py-2 text-right font-normal sm:table-cell" title="The drug's records that report any statement of this role">
              Out of
            </th>
            <th scope="col" className="px-4 py-2 text-right font-normal">
              Share
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((link, index) => {
            const drug = nodes.get(link.source)
            const pt = nodes.get(link.target)
            if (!drug || !pt) return null
            const id = `${link.source}->${link.target}`
            const first = index === 0 || rows[index - 1].source !== link.source
            const pick = () => onSelect({ id, target: link.kind, code: pt.code ?? null, label: pt.label, drug: drug.label })
            return (
              <tr
                key={id}
                tabIndex={0}
                onClick={pick}
                onKeyDown={(event) => event.key === "Enter" && pick()}
                className={cn(
                  "cursor-pointer outline-none hover:bg-info focus-visible:bg-tint",
                  first ? "border-t border-border" : "border-t border-hairline",
                  selected === id && "bg-tint"
                )}
              >
                <td className="px-4 py-1.5 font-medium text-foreground">{first ? valueLabel(drug.label) : ""}</td>
                <td className="hidden px-3 py-1.5 text-muted-foreground sm:table-cell">{ROLE_LABEL[link.kind]}</td>
                <td className="px-3 py-1.5 text-foreground">{pt.label}</td>
                <td className="hidden max-w-56 truncate px-3 py-1.5 text-muted-foreground md:table-cell" title={pt.soc ?? undefined}>
                  {pt.soc ?? "–"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">{formatNumber(link.records)}</td>
                <td className="hidden px-3 py-1.5 text-right text-muted-foreground tabular-nums sm:table-cell">{formatNumber(link.denominator)}</td>
                <td className="px-4 py-1.5 text-right tabular-nums">
                  {formatPercent(link.percent, link.records)}
                  {link.sparse ? (
                    <span className="ml-1 text-meta text-muted-foreground" title="Fewer than 10 of the drug's records report any statement of this role">
                      few records
                    </span>
                  ) : (
                    link.small && (
                      <span className="ml-1 text-meta text-muted-foreground" title="Fewer than 30 of the drug's records report any statement of this role, so the share is unstable">
                        small group
                      </span>
                    )
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
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
      <div ref={area} className="relative flex h-[clamp(440px,calc(100svh-340px),880px)] overflow-hidden">
        {fetching && view && (
          <span role="status" className="absolute top-3 left-1/2 z-40 inline-flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-meta text-ink">
            <Loader2 className="size-3.5 animate-spin" />
            Updating the graph…
          </span>
        )}
        <div
          className={cn("relative min-w-0 flex-1 transition-opacity", fetching && view && "opacity-50")}
          style={layout === "graph" ? { backgroundImage: "radial-gradient(circle, var(--dot) 1px, transparent 1.3px)", backgroundSize: "20px 20px" } : undefined}
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
        {/* Beside the graph, which re-lays out narrower, so the selected node stays in view; over it on narrow cards. */}
        {selection && view && (
          <EvidencePanel
            selection={selection}
            platform={view.platform}
            level={view.level}
            filters={filters}
            onClose={() => setSelection(null)}
            onAsk={onAsk}
            className="w-[min(380px,100%)] shrink-0 border-l border-border bg-card max-[39.99rem]:absolute max-[39.99rem]:inset-y-0 max-[39.99rem]:right-0"
          />
        )}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-border bg-info px-3 py-2 text-meta text-prose">
        <Segmented
          label="Show the graph as"
          value={layout}
          onChange={setLayout}
          className="h-[30px]"
          options={[
            { value: "graph", label: "Graph" },
            { value: "table", label: "Table" },
          ]}
        />
        {layout === "graph" && (
          <Fragment>
            {kinds.has("purpose") && <LegendDot color={PURPOSE_COLOR} label="Purpose" />}
            {kinds.has("side_effect") && <LegendDot color={SIDE_EFFECT_COLOR} label="Side effect" />}
            <span className="text-muted-foreground" title={`Terms are MedDRA Preferred Terms. Circle size and line width grow with ${unit}; sizes compare within a column.`}>
              Size and width: {unit}
            </span>
            <div role="toolbar" aria-label="Graph view" className="ml-auto flex items-center gap-0.5">
              <ToolbarButton label="Zoom out" onClick={() => canvas.current?.zoomBy(0.7)}>
                <Minus />
              </ToolbarButton>
              <span className="w-10 text-center font-mono text-[11px] text-ink">{Math.round(zoom * 100)}%</span>
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
                  Terms are MedDRA Preferred Terms. Circle size and line width grow with the number of {unit}; sizes compare within a column.
                  Hover a node to highlight its links, click a node or a line to read example statements, drag a node to move it, and pinch
                  or hold Ctrl and scroll to zoom.
                </TooltipContent>
              </Tooltip>
            </div>
          </Fragment>
        )}
        {layout === "table" && <span className="text-muted-foreground">Click a row to read example statements.</span>}
      </div>
    </>
  )
}
