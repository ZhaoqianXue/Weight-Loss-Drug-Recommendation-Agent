const SVG_NS = "http://www.w3.org/2000/svg"

/** Save the graph as a PNG at 3x resolution, on the given background color. */
export async function exportSvgAsPng(svg: SVGSVGElement, background: string, filename: string) {
  const { width, height } = svg.getBoundingClientRect()
  const clone = svg.cloneNode(true) as SVGSVGElement
  clone.setAttribute("xmlns", SVG_NS)
  clone.setAttribute("width", String(width))
  clone.setAttribute("height", String(height))
  const backdrop = document.createElementNS(SVG_NS, "rect")
  backdrop.setAttribute("width", "100%")
  backdrop.setAttribute("height", "100%")
  backdrop.setAttribute("fill", background)
  clone.insertBefore(backdrop, clone.firstChild)

  const source = new XMLSerializer().serializeToString(clone)
  const url = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }))
  try {
    const image = new Image()
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error("Could not render the graph image"))
      image.src = url
    })
    const scale = 3
    const canvas = document.createElement("canvas")
    canvas.width = Math.ceil(width * scale)
    canvas.height = Math.ceil(height * scale)
    const context = canvas.getContext("2d")
    if (!context) throw new Error("Canvas is not available")
    context.scale(scale, scale)
    context.drawImage(image, 0, 0, width, height)
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"))
    if (!blob) throw new Error("Could not encode the PNG")
    const link = document.createElement("a")
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.click()
    URL.revokeObjectURL(link.href)
  } finally {
    URL.revokeObjectURL(url)
  }
}
