import { Check, ChevronDown } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"

/** null means every option; an empty list means none. */
export type Choice = string[] | null

type Props = {
  id: string
  label: string
  noun: string
  plural: string
  options: string[]
  selection: Choice
  onChange: (selection: Choice) => void
  /** Display text for an option value. */
  format?: (value: string) => string
}

export function summarize(selection: Choice, options: string[], noun: string, plural: string) {
  if (selection === null) return `All ${plural}`
  const count = selection.filter((value) => options.includes(value)).length
  if (count === 0) return `No ${plural}`
  if (count === options.length) return `All ${plural}`
  return `${count} ${count === 1 ? noun : plural} selected`
}

/** Compact value for the filter bar: "All", "None", up to two values by name, or "3 of 7". */
function shortSummary(selection: Choice, options: string[], format: (value: string) => string) {
  if (selection === null) return "All"
  const chosen = options.filter((value) => selection.includes(value))
  if (chosen.length === 0) return "None"
  if (chosen.length === options.length) return "All"
  if (chosen.length <= 2) return chosen.map(format).join(", ")
  return `${chosen.length} of ${options.length}`
}

// Plain substring search, as in the former page; cmdk's default fuzzy match surfaces unrelated terms.
const substring = (value: string, search: string) => (value.toLowerCase().includes(search.trim().toLowerCase()) ? 1 : 0)

export function MultiSelect({ id, label, noun, plural, options, selection, onChange, format = (value) => value }: Props) {
  const isChecked = (value: string) => selection === null || selection.includes(value)

  function toggle(value: string) {
    const current = selection ?? options
    const next = current.includes(value) ? current.filter((v) => v !== value) : [...current, value]
    onChange(options.every((option) => next.includes(option)) ? null : next)
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          id={id}
          variant="outline"
          role="combobox"
          aria-label={`${label}: ${summarize(selection, options, noun, plural)}`}
          className="h-[34px] gap-1.5 pr-[9px] pl-[11px] hover:bg-background aria-expanded:bg-accent"
        >
          <span className="text-meta text-muted-foreground">{label}</span>
          <span className="max-w-40 truncate font-medium text-ink">{shortSummary(selection, options, format)}</span>
          <ChevronDown className="size-4 text-muted-foreground" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 p-0" align="start">
        <Command filter={substring}>
          {options.length > 8 && <CommandInput placeholder={`Search ${plural}…`} />}
          <div className="flex gap-1 border-b p-1">
            <Button variant="ghost" size="xs" onClick={() => onChange(null)}>
              Select all
            </Button>
            <Button variant="ghost" size="xs" onClick={() => onChange([])}>
              Clear
            </Button>
            <span className="ml-auto self-center pr-2 font-mono text-meta text-subtle">{options.length}</span>
          </div>
          <CommandList>
            <CommandEmpty>No {plural} found.</CommandEmpty>
            <CommandGroup>
              {options.map((option) => (
                <CommandItem key={option} value={format(option)} onSelect={() => toggle(option)}>
                  <span
                    className={cn(
                      "flex size-4 items-center justify-center rounded-[4px] border",
                      isChecked(option) ? "border-primary bg-primary text-primary-foreground" : "border-input"
                    )}
                  >
                    {isChecked(option) && <Check className="size-3 text-current" />}
                  </span>
                  <span className="truncate">{format(option)}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}
