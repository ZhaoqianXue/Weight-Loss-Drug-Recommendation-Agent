import { useEffect, useState } from "react"

import { useTheme } from "@/components/theme-provider"

const QUERY = "(prefers-color-scheme: dark)"

/** The theme actually shown: the user's choice, or the system setting when they chose "system". */
export function useResolvedTheme(): "light" | "dark" {
  const { theme } = useTheme()
  const [system, setSystem] = useState<"light" | "dark">(() => (window.matchMedia(QUERY).matches ? "dark" : "light"))
  useEffect(() => {
    const media = window.matchMedia(QUERY)
    const update = () => setSystem(media.matches ? "dark" : "light")
    media.addEventListener("change", update)
    return () => media.removeEventListener("change", update)
  }, [])
  return theme === "system" ? system : theme
}
