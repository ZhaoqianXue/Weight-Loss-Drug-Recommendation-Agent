import type * as React from "react"

import { cn } from "@/lib/utils"

const VARIANTS = {
  /** Cream card separated by a hairline: the default container. */
  card: "rounded-xl border border-border bg-card",
} as const

type PanelProps = React.ComponentProps<"div"> & { variant?: keyof typeof VARIANTS }

/** Flat container: separation comes from 1px borders, never from shadows or tinted fills. */
export function Panel({ variant = "card", className, ...props }: PanelProps) {
  return <div className={cn(VARIANTS[variant], className)} {...props} />
}

/** 10px grey capitals naming a section, as on ReviewPilot cards. */
export function SectionLabel({ className, ...props }: React.ComponentProps<"p">) {
  return <p className={cn("label-caps", className)} {...props} />
}

type SectionHeadingProps = Omit<React.ComponentProps<"div">, "title"> & { title: React.ReactNode; meta?: React.ReactNode }

/** Newsreader section title with an optional one-line note under it; it sits above the section's cards. */
export function SectionHeading({ title, meta, className, ...props }: SectionHeadingProps) {
  return (
    <div className={cn("grid gap-0.5", className)} {...props}>
      <h2 className="flex items-center gap-2 font-display text-[23px] leading-[1.2] font-normal text-ink">{title}</h2>
      {meta && <p className="text-[12.5px] text-muted-foreground">{meta}</p>}
    </div>
  )
}
