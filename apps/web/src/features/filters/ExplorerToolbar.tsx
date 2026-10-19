import { RotateCcw } from "lucide-react"
import { useState } from "react"

import type { Overview } from "@/api/client"
import { Segmented } from "@/components/segmented"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { valueLabel } from "@/lib/format"
import type { Explorer } from "@/state"

import { MultiSelect } from "./MultiSelect"

const ALL = "__all__"

/** "4 purposes and side effects", "1 side effect": what the slider's number counts. */
function roles(show: Explorer["show"], n: number) {
  const one = n === 1
  if (show === "side_effect") return one ? "side effect" : "side effects"
  if (show === "purpose") return one ? "purpose" : "purposes"
  return one ? "purpose and side effect" : "purposes and side effects"
}

type Props = {
  overview: Overview | undefined
  state: Explorer
  onChange: (update: Partial<Explorer>) => void
  onReset: () => void
}

/** What the graph shows: platform, role, drugs, MedDRA SOC, writer population and how many PTs per drug. */
export function ExplorerToolbar({ overview, state, onChange, onReset }: Props) {
  // The slider shows its value while dragging and asks for a new graph only on release.
  const [draftTop, setDraftTop] = useState<number | null>(null)
  const topN = draftTop ?? state.topN
  const drugOptions = state.level === "brand" ? (overview?.brands.map((b) => b.brand) ?? []) : (overview?.ingredients ?? [])
  const socs = overview?.socs ?? []

  return (
    <div role="group" aria-label="Graph filters" className="grid gap-2 border-b border-border bg-info px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          label="Platform"
          value={state.platform}
          onChange={(platform) => onChange({ platform })}
          options={[
            { value: "reddit", label: "Reddit" },
            { value: "webmd", label: "WebMD" },
          ]}
        />
        <Segmented
          label="Statements"
          value={state.show}
          onChange={(show) => onChange({ show })}
          options={[
            { value: "both", label: "Purposes + side effects" },
            { value: "side_effect", label: "Side effects" },
            { value: "purpose", label: "Purposes" },
          ]}
        />
        <Segmented
          label="Drug level"
          value={state.level}
          onChange={(level) => onChange({ level, drugs: null })}
          options={[
            { value: "brand", label: "By brand", title: "Only brands named in the text or the WebMD profile" },
            { value: "ingredient", label: "By ingredient", title: "Includes unnamed references assigned from a single-ingredient community" },
          ]}
        />
        <Button variant="ghost" size="sm" className="ml-auto h-[34px] px-2" onClick={onReset}>
          <RotateCcw className="size-3" />
          Reset
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <MultiSelect
          id="filter-drugs"
          label={state.level === "brand" ? "Brands" : "Ingredients"}
          noun={state.level}
          plural={`${state.level}s`}
          options={drugOptions}
          format={valueLabel}
          selection={state.drugs}
          onChange={(drugs) => onChange({ drugs })}
        />
        <Select value={state.soc ?? ALL} onValueChange={(soc) => onChange({ soc: soc === ALL ? null : soc })}>
          <SelectTrigger aria-label="MedDRA system organ class" className="h-[34px] max-w-72 gap-1.5 pr-[9px] pl-[11px]">
            <span className="text-meta text-muted-foreground" title="MedDRA system organ class">
              SOC
            </span>
            <SelectValue className="truncate font-medium text-ink">
              {state.soc ? (socs.find((s) => s.code === state.soc)?.name ?? state.soc) : "All"}
            </SelectValue>
          </SelectTrigger>
          <SelectContent className="max-h-80">
            <SelectItem value={ALL}>All system organ classes</SelectItem>
            {socs.map((soc) => (
              <SelectItem key={soc.code} value={soc.code}>
                {soc.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <MultiSelect
          id="filter-age"
          label="Age"
          noun="age group"
          plural="age groups"
          options={[...(overview?.age_groups ?? []), "not stated"]}
          format={valueLabel}
          selection={state.ageGroups}
          onChange={(ageGroups) => onChange({ ageGroups })}
        />
        <MultiSelect
          id="filter-sex"
          label="Sex"
          noun="value"
          plural="values"
          options={[...(overview?.sexes ?? []), "not stated"]}
          format={valueLabel}
          selection={state.sexes}
          onChange={(sexes) => onChange({ sexes })}
        />
        <div className="ml-auto flex h-[34px] items-center gap-2.5">
          <span aria-hidden className="text-meta text-muted-foreground">
            Each {state.level}'s top
          </span>
          <Slider
            id="filter-top"
            thumbLabel={`Terms shown for each ${state.level}`}
            valueText={`Top ${topN} ${roles(state.show, topN)}`}
            className="w-24"
            min={1}
            max={10}
            step={1}
            value={[topN]}
            onValueChange={([value]) => setDraftTop(value)}
            onValueCommit={([value]) => {
              setDraftTop(null)
              onChange({ topN: value })
            }}
          />
          <span aria-hidden className="text-meta text-muted-foreground">
            <span className="inline-block w-4 text-right font-mono text-[12.5px] text-ink">{topN}</span> {roles(state.show, topN)}
          </span>
        </div>
      </div>
    </div>
  )
}
