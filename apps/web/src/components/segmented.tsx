import { ToggleGroup } from "radix-ui"

import { cn } from "@/lib/utils"

type Option<T extends string> = { value: T; label: React.ReactNode; title?: string }

/** One-of-several choice drawn as joined buttons; the selected one is tinted, so solid navy stays for primary actions. */
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
          className="inline-flex h-full items-center rounded-[7px] px-2.5 text-[12.5px] text-prose transition-colors outline-none hover:text-ink focus-visible:ring-3 focus-visible:ring-ring/50 data-[state=on]:bg-selected data-[state=on]:font-medium data-[state=on]:text-selected-fg data-[state=on]:shadow-[inset_0_0_0_1px_var(--selected-ring)]"
        >
          {option.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  )
}
