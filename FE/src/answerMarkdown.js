const HEADING = /^(#{1,3})\s+(.+)$/;
const UNORDERED_ITEM = /^\s*[-*]\s+(.+)$/;
const ORDERED_ITEM = /^\s*\d+[.)]\s+(.+)$/;

function blockType(line) {
  if (HEADING.test(line)) return "heading";
  if (UNORDERED_ITEM.test(line)) return "unordered-list";
  if (ORDERED_ITEM.test(line)) return "ordered-list";
  return "paragraph";
}

/** Parse common answer Markdown into safe text blocks, without injecting HTML. */
export function parseAnswerBlocks(content) {
  const lines = String(content ?? "").replace(/\r/g, "").split("\n");
  const blocks = [];

  for (let index = 0; index < lines.length;) {
    const line = lines[index].trim();
    if (!line) {
      index += 1;
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
      index += 1;
      continue;
    }

    const unordered = UNORDERED_ITEM.exec(lines[index]);
    const ordered = ORDERED_ITEM.exec(lines[index]);
    if (unordered || ordered) {
      const type = unordered ? "unordered-list" : "ordered-list";
      const matcher = unordered ? UNORDERED_ITEM : ORDERED_ITEM;
      const items = [];
      while (index < lines.length) {
        const match = matcher.exec(lines[index]);
        if (!match) break;
        items.push(match[1].trim());
        index += 1;
      }
      blocks.push({ type, items });
      continue;
    }

    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && blockType(lines[index].trim()) === "paragraph") {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", text: paragraph.join(" ") });
  }

  return blocks;
}

export function parseInlineMarkdown(content) {
  const parts = String(content ?? "").split(/(\*\*[^*]+\*\*)/g);
  return parts.filter(Boolean).map((part) => (
    part.startsWith("**") && part.endsWith("**")
      ? { type: "strong", text: part.slice(2, -2) }
      : { type: "text", text: part }
  ));
}
