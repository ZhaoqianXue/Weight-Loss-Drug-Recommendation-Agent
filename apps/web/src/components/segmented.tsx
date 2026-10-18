import { ToggleGroup } from "radix-ui"

import { cn } from "@/lib/utils"

type Option<T extends string> = { value: T; label: React.ReactNode; title?: string }

/** One-of-several choice drawn as joined buttons; the selected one is navy. */
export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  className,
}: {
  label: string
  value: T
  options: Option<T>[]
  onChange: (value: T) => void
  className?: string
}) {
  return (
    <ToggleGroup.Root
      type="single"
      aria-label={label}
      value={value}
      onValueChange={(next) => next && onChange(next as T)}
      className={cn("inline-flex h-[34px] items-center gap-0.5 rounded-[9px] border border-input bg-card p-0.5", className)}
    >
      {options.map((option) => (
        <ToggleGroup.Item
          key={option.value}
          value={option.value}
          title={option.title}
          className="inline-flex h-full items-center rounded-[7px] px-2.5 text-[12px] text-prose transition-colors outline-none hover:text-ink focus-visible:ring-3 focus-visible:ring-ring/50 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
        >
          {option.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  )
}
