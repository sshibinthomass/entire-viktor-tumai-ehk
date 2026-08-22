#!/usr/bin/env node
/** Map route model IDs to their display names in the LiveBench snapshot. */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaults = [
  path.join(root, "benchmarks/route-models.json"),
  path.join(root, "benchmarks/livebench-leaderboard.csv"),
  path.join(root, "benchmarks/route-model-livebench-map.json"),
];
const ignoredTokens = new Set(["thinking", "effort", "max", "xhigh", "high", "medium", "low"]);

function matchKey(model) {
  return (model.toLowerCase().match(/[a-z]+|\d+/g) ?? [])
    .filter((token) => !ignoredTokens.has(token))
    .sort()
    .join(":");
}

function parseCsvRow(line) {
  const fields = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && quoted && line[index + 1] === '"') {
      field += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      fields.push(field);
      field = "";
    } else {
      field += character;
    }
  }
  fields.push(field);
  return fields;
}

async function main() {
  const [routeModelsArg, leaderboardArg, outputArg] = process.argv.slice(2);
  const routeModelsPath = routeModelsArg ?? defaults[0];
  const leaderboardPath = leaderboardArg ?? defaults[1];
  const outputPath = outputArg ?? defaults[2];
  const routeModels = JSON.parse(await readFile(routeModelsPath, "utf8"));
  if (!Array.isArray(routeModels) || !routeModels.every((model) => typeof model === "string")) {
    throw new Error("route-models.json must contain an array of strings");
  }

  const lines = (await readFile(leaderboardPath, "utf8")).trim().split(/\r?\n/);
  const headers = parseCsvRow(lines[0]);
  const modelColumn = headers.indexOf("Model");
  if (modelColumn === -1) throw new Error("leaderboard CSV has no Model column");
  const liveBenchModels = lines.slice(1).map((line) => parseCsvRow(line)[modelColumn]);

  const modelsByKey = new Map();
  for (const model of liveBenchModels) {
    const key = matchKey(model);
    modelsByKey.set(key, [...(modelsByKey.get(key) ?? []), model]);
  }

  const mapping = {};
  for (const routeModel of routeModels) {
    const matches = modelsByKey.get(matchKey(routeModel)) ?? [];
    if (matches.length !== 1) {
      throw new Error(`${routeModel}: expected one LiveBench match, found ${matches.length}`);
    }
    mapping[routeModel] = matches[0];
  }

  await mkdir(path.dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(mapping, null, 2)}\n`);
  console.log(`wrote ${Object.keys(mapping).length} mappings to ${outputPath}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
