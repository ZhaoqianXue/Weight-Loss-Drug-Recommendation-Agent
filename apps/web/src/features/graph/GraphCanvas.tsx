import * as d3 from "d3"
import { useEffect, useImperativeHandle, useRef, type Ref } from "react"

import type { GraphLink, GraphNode } from "@/api/client"
import { useElementSize } from "@/hooks/use-element-size"
import { formatPercent, valueLabel } from "@/lib/format"
import type { NodeSizes, Selection } from "@/state"

import { exportSvgAsPng } from "./export-png"
import { GRAPH_THEME, drugColor, makeRadius, nodeColor, type GraphMode } from "./style"

type Pointer = { x: number; y: number; width: number; height: number }
export type Hover = ({ kind: "node"; node: GraphNode } | { kind: "link"; link: GraphLink; source: GraphNode; target: GraphNode }) &
  Pointer

export type GraphCanvasHandle = {
  zoomBy: (factor: number) => void
  resetZoom: () => void
  exportPng: (filename: string) => Promise<void>
}

type Props = {
  nodes: GraphNode[]
  links: GraphLink[]
  sizes: NodeSizes
  mode: GraphMode
  selected: string | null
  /** Title of the drug column, e.g. "Brands". */
  drugTitle: string
  onZoom: (scale: number) => void
  onHover: (hover: Hover | null) => void
  onSelect: (selection: Selection & { id: string }) => void
  ref?: Ref<GraphCanvasHandle>
}

type Placed = GraphNode & { x: number; y: number; r: number; side: "left" | "center" | "right" }

const LABEL_FONT = "'Hanken Grotesk Variable', 'Hanken Grotesk', system-ui, sans-serif"
const PADDING = { top: 52, bottom: 28 }
const MAX_RADIUS = 26

/**
 * Three columns: purposes on the left, drugs in the middle, side effects on the right (two columns when one role is
 * hidden). Drugs keep catalog order; each PT sits at the mean height of the drugs it links to, weighted by records,
 * then nodes are spread evenly in that order, which keeps crossings few and the figure the same for the same view.
 */
function layout(nodes: GraphNode[], links: GraphLink[], width: number, height: number, radius: (n: GraphNode) => number) {
  const kinds = new Set(nodes.map((n) => n.kind))
  const hasPurpose = kinds.has("purpose")
  const hasEffect = kinds.has("side_effect")
  // Outer columns sit just far enough in to fit their longest label (within limits); the drug column sits between.
  const context = document.createElement("canvas").getContext("2d")
  const longest = (kind: string) => {
    if (!context) return 160
    context.font = `400 13px ${LABEL_FONT}`
    return Math.max(0, ...nodes.filter((n) => n.kind === kind).map((n) => context.measureText(n.label).width))
  }
  const inset = (kind: string, min: number, max: number) => Math.min(max, Math.max(min, (longest(kind) + 46) / width))
  const both = hasPurpose && hasEffect
  const left = hasPurpose ? inset("purpose", 0.14, both ? 0.36 : 0.45) : 0
  const right = hasEffect ? inset("side_effect", 0.14, both ? 0.36 : 0.45) : 0
  // With one role hidden, the two columns sit symmetrically about the centre, at most 680px apart, and drug labels go
  // on the outer side of their nodes, so the figure mirrors.
  const half = Math.min(0.5 - Math.max(left, right, inset("drug", 0.12, 0.4)), 340 / width)
  const column = both
    ? { purpose: left, side_effect: 1 - right, drug: (left + 1 - right) / 2 }
    : hasPurpose
      ? { purpose: 0.5 - half, side_effect: 1, drug: 0.5 + half }
      : { purpose: 0, side_effect: 0.5 + half, drug: 0.5 - half }
  const drugSide: Placed["side"] = both ? "center" : hasPurpose ? "right" : "left"
  const usable = height - PADDING.top - PADDING.bottom
  const spread = (list: GraphNode[]) => {
    const step = usable / Math.max(list.length, 1)
    return new Map(list.map((n, i) => [n.id, PADDING.top + step * (i + 0.5)]))
  }
  const drugs = nodes.filter((n) => n.kind === "drug")
  const drugY = spread(drugs)
  const placed = new Map<string, Placed>()
  for (const n of drugs) placed.set(n.id, { ...n, x: width * column.drug, y: drugY.get(n.id)!, r: 0, side: drugSide })
  for (const kind of ["purpose", "side_effect"] as const) {
    const members = nodes.filter((n) => n.kind === kind)
    const center = new Map<string, number>()
    for (const n of members) {
      let sum = 0
      let weight = 0
      for (const l of links) {
        if (l.target !== n.id) continue
        const y = drugY.get(l.source)
        if (y === undefined) continue
        sum += y * l.records
        weight += l.records
      }
      center.set(n.id, weight ? sum / weight : height / 2)
    }
    members.sort((a, b) => center.get(a.id)! - center.get(b.id)! || a.label.localeCompare(b.label))
    const ys = spread(members)
    for (const n of members)
      placed.set(n.id, { ...n, x: width * column[kind], y: ys.get(n.id)!, r: 0, side: kind === "purpose" ? "left" : "right" })
  }
  // Radii fit their column (PT labels sit beside the node, drug labels under it, about 18px): when the largest node of a
  // column would not fit, the whole column shrinks by one factor, so sizes stay comparable within it.
  const step = (kind: string) => usable / Math.max(nodes.filter((n) => n.kind === kind).length, 1)
  const room = (kind: string) => Math.min(MAX_RADIUS, kind === "drug" && drugSide === "center" ? (step(kind) - 28) / 2 : step(kind) * 0.42)
  const largest: Record<string, number> = {}
  for (const p of placed.values()) largest[p.kind] = Math.max(largest[p.kind] ?? 0, radius(p))
  for (const p of placed.values()) p.r = Math.max(p.kind === "drug" ? 5 : 3, radius(p) * Math.min(1, room(p.kind) / largest[p.kind]))
  return { placed, column }
}

export function GraphCanvas({ nodes, links, sizes, mode, selected, drugTitle, onZoom, onHover, onSelect, ref }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null)
  const { width, height } = useElementSize(containerRef)
  const theme = GRAPH_THEME[mode]

  useImperativeHandle(ref, () => ({
    zoomBy: (factor) => {
      if (svgRef.current && zoomRef.current)
        d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.scaleBy, factor)
    },
    resetZoom: () => {
      if (svgRef.current && zoomRef.current)
        d3.select(svgRef.current).transition().duration(300).call(zoomRef.current.transform, d3.zoomIdentity)
    },
    exportPng: (filename) => (svgRef.current ? exportSvgAsPng(svgRef.current, theme.surface, filename) : Promise.resolve()),
  }))

  useEffect(() => {
    const svgElement = svgRef.current
    if (!svgElement || !width || !height) return
    const svg = d3.select(svgElement)
    svg.selectAll("*").remove()
    if (!nodes.length) return

    const { placed, column } = layout(nodes, links, width, height, makeRadius(nodes, sizes))
    const items = [...placed.values()]
    // A link's width is its share: the drug's records reporting the term out of its records with any statement of
    // that role (node size is records). Wide links are drawn first, so thin ones stay visible on top.
    const share = (e: { data: GraphLink }) => e.data.percent ?? 0
    const edges = links
      .filter((l) => placed.has(l.source) && placed.has(l.target))
      .map((l) => ({ data: l, source: placed.get(l.source)!, target: placed.get(l.target)! }))
      .sort((a, b) => share(b) - share(a))
    type Edge = (typeof edges)[number]
    const maxShare = Math.max(1, ...edges.map(share))
    const edgeWidth = (e: Edge) => 0.75 + 6.5 * (share(e) / maxShare)
    // Room for a label between its node and the canvas edge; drug labels under their node get a fixed width.
    const labelRoom = (p: Placed) => {
      if (p.side === "left") return p.x - p.r - 14
      if (p.side === "right") return width - p.x - p.r - 14
      return 150
    }

    // Column titles stay fixed while the graph zooms and pans.
    const kinds = new Set(nodes.map((n) => n.kind))
    const count = (kind: string) => items.filter((p) => p.kind === kind).length
    const titles = [
      ...(kinds.has("purpose") ? [{ x: column.purpose, text: `Purposes · ${count("purpose")}` }] : []),
      { x: column.drug, text: `${drugTitle} · ${count("drug")}` },
      ...(kinds.has("side_effect") ? [{ x: column.side_effect, text: `Side effects · ${count("side_effect")}` }] : []),
    ]
    svg
      .append("g")
      .attr("font-family", LABEL_FONT)
      .attr("pointer-events", "none")
      .selectAll("text")
      .data(titles)
      .join("text")
      .attr("x", (t) => t.x * width)
      .attr("y", 24)
      .attr("text-anchor", "middle")
      .attr("font-size", 11)
      .attr("font-weight", 600)
      .attr("letter-spacing", "0.06em")
      .attr("fill", theme.title)
      .text((t) => t.text.toUpperCase())

    const root = svg.append("g").attr("font-family", LABEL_FONT)
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 5])
      .filter((event: Event) => event.type !== "wheel" || (event as WheelEvent).ctrlKey || (event as WheelEvent).metaKey)
      .on("zoom", (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => {
        root.attr("transform", event.transform.toString())
        onZoom(event.transform.k)
      })
    zoomRef.current = zoom
    svg.call(zoom).call(zoom.transform, d3.zoomIdentity)

    const pointer = (event: MouseEvent): Pointer => {
      const box = svgElement.getBoundingClientRect()
      return { x: event.clientX - box.left, y: event.clientY - box.top, width, height }
    }
    const path = (e: Edge) => {
      const [a, b] = e.source.x < e.target.x ? [e.source, e.target] : [e.target, e.source]
      const mid = (a.x + b.x) / 2
      return `M${a.x},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`
    }

    const edgeLayer = root.append("g").attr("fill", "none")
    const edge = edgeLayer
      .selectAll<SVGPathElement, Edge>("path")
      .data(edges)
      .join("path")
      .attr("d", path)
      .attr("stroke", theme.link)
      .attr("stroke-opacity", theme.linkOpacity)
      .attr("stroke-width", edgeWidth)
      .attr("cursor", "pointer")
      .on("mouseover", (event: MouseEvent, e) => {
        highlight(e.source.id, e.target.id)
        onHover({ kind: "link", link: e.data, source: e.source, target: e.target, ...pointer(event) })
      })
      .on("mousemove", (event: MouseEvent, e) => onHover({ kind: "link", link: e.data, source: e.source, target: e.target, ...pointer(event) }))
      .on("mouseout", () => {
        highlight(null)
        onHover(null)
      })
      .on("click", (_event, e) =>
        onSelect({ id: `${e.source.id}->${e.target.id}`, target: e.data.kind, code: e.target.code ?? null, label: e.target.label, drug: e.source.label })
      )

    const nodeLayer = root.append("g")
    const node = nodeLayer
      .selectAll<SVGCircleElement, Placed>("circle")
      .data(items)
      .join("circle")
      .attr("cx", (p) => p.x)
      .attr("cy", (p) => p.y)
      .attr("r", (p) => p.r)
      .attr("fill", (p) => nodeColor(p))
      .attr("stroke", (p) => (p.id === selected ? theme.selected : theme.ring))
      .attr("stroke-width", (p) => (p.id === selected ? 2.5 : 1.5))
      .attr("cursor", "pointer")
      .on("mouseover", (event: MouseEvent, p) => {
        highlight(p.id)
        onHover({ kind: "node", node: p, ...pointer(event) })
      })
      .on("mousemove", (event: MouseEvent, p) => onHover({ kind: "node", node: p, ...pointer(event) }))
      .on("mouseout", () => {
        highlight(null)
        onHover(null)
      })
      .on("click", (event: MouseEvent, p) => {
        if (event.defaultPrevented) return // end of a drag
        if (p.kind === "drug") {
          const role = links.find((l) => l.source === p.id)?.kind ?? "side_effect"
          onSelect({ id: p.id, target: role, code: null, label: p.label, drug: p.label })
        } else {
          onSelect({ id: p.id, target: p.kind, code: p.code ?? null, label: p.label, drug: null })
        }
      })

    const labelLayer = root.append("g").attr("pointer-events", "none")
    const label = labelLayer
      .selectAll<SVGTextElement, Placed>("text")
      .data(items)
      .join("text")
      .attr("fill", theme.text)
      .attr("font-size", (p) => (p.kind === "drug" ? 14 : 13))
      .attr("font-weight", (p) => (p.kind === "drug" ? 600 : 400))
      .attr("text-anchor", (p) => (p.side === "left" ? "end" : p.side === "right" ? "start" : "middle"))
      .attr("dominant-baseline", (p) => (p.side === "center" ? "hanging" : "middle"))
      .attr("paint-order", "stroke")
      .attr("stroke", theme.surface)
      .attr("stroke-width", 4)
      .attr("stroke-linejoin", "round")
      .text((p) => (p.kind === "drug" ? valueLabel(p.label) : p.label))
    const place = (sel: d3.Selection<SVGTextElement, Placed, SVGGElement, unknown>) =>
      sel
        .attr("x", (p) => (p.side === "left" ? p.x - p.r - 7 : p.side === "right" ? p.x + p.r + 7 : p.x))
        .attr("y", (p) => (p.side === "center" ? p.y + p.r + 5 : p.y))
    place(label)
    // One line per label; the tooltip shows names cut here in full.
    label.each(function (p) {
      const text = this as SVGTextElement
      const room = labelRoom(p)
      if (text.getComputedTextLength() <= room) return
      let value = p.kind === "drug" ? valueLabel(p.label) : p.label
      while (value.length > 4 && text.getComputedTextLength() > room) {
        value = value.slice(0, -2)
        text.textContent = `${value.trimEnd()}…`
      }
    })

    // While a node is hovered, each of its links shows its share at the far end: at the term for a drug, at the drug
    // for a term.
    const shareLayer = root.append("g").attr("pointer-events", "none").attr("font-size", 11.5).attr("font-weight", 600)
    function showShares(a: string | null) {
      const shown = a === null ? [] : edges.filter((e) => e.source.id === a || e.target.id === a)
      shareLayer
        .selectAll<SVGTextElement, Edge>("text")
        .data(shown, (e) => `${e.source.id}->${e.target.id}`)
        .join("text")
        .each(function (e) {
          const end = e.source.id === a ? e.target : e.source
          const other = end === e.target ? e.source : e.target
          const toward = other.x < end.x ? -1 : 1 // the side of the node the link arrives on
          d3.select(this)
            .attr("x", end.x + toward * (end.r + 6))
            .attr("y", end.y - 6)
            .attr("text-anchor", toward < 0 ? "end" : "start")
        })
        .attr("fill", (e) => drugColor(e.source.label))
        .attr("paint-order", "stroke")
        .attr("stroke", theme.surface)
        .attr("stroke-width", 4)
        .attr("stroke-linejoin", "round")
        .text((e) => formatPercent(e.data.percent, e.data.records))
    }

    function highlight(a: string | null, b?: string) {
      showShares(b ? null : a)
      if (a === null) {
        edge.attr("stroke", theme.link).attr("stroke-opacity", theme.linkOpacity)
        node.attr("opacity", 1)
        label.attr("opacity", 1)
        return
      }
      const linked = new Set<string>([a, ...(b ? [b] : [])])
      const active = (e: Edge) => (b ? e.source.id === a && e.target.id === b : e.source.id === a || e.target.id === a)
      for (const e of edges) if (active(e)) linked.add(e.source.id).add(e.target.id)
      edge
        .attr("stroke", (e) => (active(e) ? drugColor(e.source.label) : theme.link))
        .attr("stroke-opacity", (e) => (active(e) ? 0.9 : 0.12))
      node.attr("opacity", (p) => (linked.has(p.id) ? 1 : 0.35))
      label.attr("opacity", (p) => (linked.has(p.id) ? 1 : 0.35))
    }

    // Drag a node to move it; it stays where it is dropped.
    node.call(
      d3
        .drag<SVGCircleElement, Placed>()
        .clickDistance(4)
        .on("drag", function (event, p) {
          p.x = Math.max(p.r, Math.min(width - p.r, event.x))
          p.y = Math.max(p.r, Math.min(height - p.r, event.y))
          d3.select(this).attr("cx", p.x).attr("cy", p.y)
          edge.attr("d", path)
          place(label.filter((q) => q.id === p.id))
        })
    )
  }, [nodes, links, sizes, mode, width, height, theme, selected, drugTitle, onZoom, onHover, onSelect])

  return (
    <div ref={containerRef} className="absolute inset-0">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        role="img"
        aria-label={`Knowledge graph with ${nodes.length} nodes and ${links.length} links`}
        style={{ background: "transparent" }}
      />
    </div>
  )
}
