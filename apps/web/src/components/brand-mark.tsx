import { Waypoints } from "lucide-react"

import { cn } from "@/lib/utils"

/** The app's mark: a graph glyph in a rounded square. Used in the masthead and as the assistant's avatar. */
export function BrandMark({ className, onMasthead = false }: { className?: string; onMasthead?: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-flex shrink-0 items-center justify-center rounded-[30%]",
        onMasthead ? "bg-masthead-fg text-masthead" : "bg-primary text-primary-foreground",
        className
      )}
    >
      <Waypoints className="size-[58%]" strokeWidth={2.1} />
    </span>
  )
}
