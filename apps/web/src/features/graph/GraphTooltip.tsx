import { formatNumber, formatPercent, ROLE_LABEL, valueLabel } from "@/lib/format"

import type { Hover } from "./GraphCanvas"

const WIDTH = 280
const capital = (text: string) => text.charAt(0).toUpperCase() + text.slice(1)

function Line({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right tabular-nums">{value}</span>
    </div>
  )
}

export function GraphTooltip({ hover, unit }: { hover: Hover; unit: string }) {
  // Keep the tooltip inside the canvas: flip left near the right edge, upward in the lower part.
  // Open away from the pointer's half of the canvas, so the hovered node, its label and its links stay visible.
  const flipX = hover.x > hover.width / 2
  const flipY = hover.y > hover.height * 0.55
  const style = {
    width: WIDTH,
    left: flipX ? hover.x - WIDTH - 28 : hover.x + 28,
    ...(flipY ? { bottom: hover.height - hover.y + 12 } : { top: hover.y + 12 }),
  }
  return (
    <div role="tooltip" className="pointer-events-none absolute z-20 grid gap-1 rounded-[10px] border border-border bg-card px-3 py-2.5 text-[12px]" style={style}>
      {hover.kind === "node" ? (
        <>
          <div className="text-[13px] font-medium text-ink">{hover.node.kind === "drug" ? valueLabel(hover.node.label) : hover.node.label}</div>
          <div className="text-muted-foreground">
            {hover.node.kind === "drug"
              ? hover.node.ingredient !== hover.node.label
                ? `Brand of ${valueLabel(hover.node.ingredient)}`
                : "Ingredient"
              : `${ROLE_LABEL[hover.node.kind]} (MedDRA Preferred Term)`}
            {hover.node.code && <> · MedDRA PT {hover.node.code}</>}
          </div>
          {hover.node.soc && <div className="text-muted-foreground">{hover.node.soc}</div>}
          <div className="mt-1 grid gap-0.5">
            {Object.entries(hover.node.records).map(([role, records]) => (
              <Line
                key={role}
                label={hover.node.kind === "drug" ? `${capital(unit)} with a ${ROLE_LABEL[role].toLowerCase()}` : `${capital(unit)} reporting it`}
                value={formatNumber(records)}
              />
            ))}
          </div>
          <div className="mt-1 text-meta text-muted-foreground">Click to read example statements.</div>
        </>
      ) : (
        <>
          <div className="text-[13px] font-medium text-ink">
            {valueLabel(hover.source.label)} → {hover.target.label}
          </div>
          <div className="text-muted-foreground">{ROLE_LABEL[hover.link.kind]} statements</div>
          <div className="mt-1 grid gap-0.5">
            <Line label={`${capital(unit)} reporting it`} value={formatNumber(hover.link.records)} />
            <Line label={`Out of ${valueLabel(hover.source.label)} ${unit} with a ${ROLE_LABEL[hover.link.kind].toLowerCase()}`} value={formatNumber(hover.link.denominator)} />
            <Line label="Share" value={formatPercent(hover.link.percent)} />
          </div>
          {hover.link.sparse ? (
            <div className="text-meta text-muted-foreground">Fewer than 10 of this drug's {unit} report any {ROLE_LABEL[hover.link.kind].toLowerCase()}: too few to compare.</div>
          ) : (
            hover.link.small && (
              <div className="text-meta text-muted-foreground">Fewer than 30 of this drug's {unit} report any {ROLE_LABEL[hover.link.kind].toLowerCase()}: a small group, so the share is unstable.</div>
            )
          )}
          <div className="mt-1 text-meta text-muted-foreground">Click to read example statements.</div>
        </>
      )}
    </div>
  )
}
