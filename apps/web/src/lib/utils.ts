import { createCn } from "cn/config"

/** Class merging that knows the custom type scale, so `text-ui` is a size and does not drop `text-ink`. */
export const cn = createCn({
  extend: {
    classGroups: {
      "font-size": [{ text: ["label", "meta", "ui", "title"] }],
    },
  },
})
