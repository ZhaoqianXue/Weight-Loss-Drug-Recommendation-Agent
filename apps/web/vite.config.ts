import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The FastAPI backend (weightloss serve) owns /api; Vite only serves the frontend during development.
const apiTarget = process.env.WEIGHTLOSS_API ?? "http://127.0.0.1:5001"

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    proxy: { "/api": apiTarget },
  },
  // One local page; the bundle (about 170 kB gzipped) is not worth splitting.
  build: { chunkSizeWarningLimit: 700 },
})
