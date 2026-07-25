import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const allowed = new Set([
  "README.md",
  "iteration.md",
  "state.json",
  "results.tsv",
  "notes.md",
  "condition-memory.md",
  "batch-strategy.md",
  "estimator.py",
  "proposal.json",
]);

function checkedPath(cwd: string, name: string): string {
  if (!allowed.has(name)) throw new Error(`Path is not allowed: ${name}`);
  return resolve(cwd, name);
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "read_research_file",
    label: "Read research file",
    description: "Read one allowed research file by its base name.",
    parameters: Type.Object({ name: Type.String() }),
    async execute(_id, { name }, _signal, _update, ctx) {
      const text = await readFile(checkedPath(ctx.cwd, name), "utf8");
      return { content: [{ type: "text", text }], details: {} };
    },
  });

  pi.registerTool({
    name: "replace_research_text",
    label: "Replace research text",
    description:
      "Replace one unique exact string in estimator.py or proposal.json.",
    parameters: Type.Object({
      name: Type.String(),
      oldText: Type.String(),
      newText: Type.String(),
    }),
    async execute(_id, { name, oldText, newText }, _signal, _update, ctx) {
      if (!new Set(["estimator.py", "proposal.json"]).has(name)) {
        throw new Error(`File is read-only: ${name}`);
      }
      const path = checkedPath(ctx.cwd, name);
      const text = await readFile(path, "utf8");
      const first = text.indexOf(oldText);
      if (first < 0 || text.indexOf(oldText, first + oldText.length) >= 0) {
        throw new Error("oldText must match exactly once");
      }
      await writeFile(path, text.replace(oldText, newText));
      return { content: [{ type: "text", text: `Updated ${name}` }], details: {} };
    },
  });
}
