import test from "node:test";
import assert from "node:assert/strict";
import { parseAnswerBlocks, parseInlineMarkdown } from "./answerMarkdown.js";

test("answer Markdown becomes readable headings, paragraphs, and lists", () => {
  assert.deepEqual(parseAnswerBlocks(
    "## Thành phần hồ sơ\nCần chuẩn bị các giấy tờ sau:\n- Giấy đề nghị [Evidence ID: catalog:1]\n- Điều lệ công ty\n\n### Thời hạn\nTheo nguồn.",
  ), [
    { type: "heading", level: 2, text: "Thành phần hồ sơ" },
    { type: "paragraph", text: "Cần chuẩn bị các giấy tờ sau:" },
    { type: "unordered-list", items: ["Giấy đề nghị [Evidence ID: catalog:1]", "Điều lệ công ty"] },
    { type: "heading", level: 3, text: "Thời hạn" },
    { type: "paragraph", text: "Theo nguồn." },
  ]);
});

test("ordered lists and untrusted HTML remain plain text blocks", () => {
  assert.deepEqual(parseAnswerBlocks("1. Bước một\n2) Bước hai\n\n<script>alert(1)</script>"), [
    { type: "ordered-list", items: ["Bước một", "Bước hai"] },
    { type: "paragraph", text: "<script>alert(1)</script>" },
  ]);
});

test("inline bold is parsed into text tokens without interpreting HTML", () => {
  assert.deepEqual(parseInlineMarkdown("**Lưu ý:** <script>không chạy</script>"), [
    { type: "strong", text: "Lưu ý:" },
    { type: "text", text: " <script>không chạy</script>" },
  ]);
});
