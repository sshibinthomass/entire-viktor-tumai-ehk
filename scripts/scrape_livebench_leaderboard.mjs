#!/usr/bin/env node
/** Scrape the first four columns of LiveBench's default leaderboard view. */

import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const REPOSITORY = "LiveBench/new-livebench";
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_OUTPUT = path.join(REPO_ROOT, "benchmarks/livebench-leaderboard.csv");

function parseArguments(argv) {
  const options = { output: DEFAULT_OUTPUT, release: null, sourceRoot: null };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--output") options.output = argv[++index];
    else if (argument === "--release") options.release = argv[++index];
    else if (argument === "--source-root") options.sourceRoot = argv[++index];
    else throw new Error(`unknown argument: ${argument}`);
  }
  return options;
}

async function fetchText(url) {
  const response = await fetch(url, { headers: { "User-Agent": "livebench-scraper" } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.text();
}

async function githubRevision() {
  const response = await fetch(`https://api.github.com/repos/${REPOSITORY}/commits/main`, {
    headers: { Accept: "application/vnd.github+json", "User-Agent": "livebench-scraper" },
  });
  if (!response.ok) throw new Error(`unable to resolve LiveBench revision: ${response.status}`);
  return (await response.json()).sha;
}

function parseCsv(text) {
  const records = [];
  let record = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      record.push(field);
      field = "";
    } else if (character === "\n" && !quoted) {
      record.push(field.replace(/\r$/, ""));
      if (record.some((value) => value !== "")) records.push(record);
      record = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field || record.length) {
    record.push(field);
    records.push(record);
  }

  const [headers, ...rows] = records;
  return rows.map((values) => Object.fromEntries(headers.map((header, index) => {
    const value = values[index] ?? "";
    const number = Number(value);
    return [header, value !== "" && Number.isFinite(number) ? number : value];
  })));
}

function average(row, columns) {
  const values = columns.map((column) => Number(row[column])).filter(Number.isFinite);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : NaN;
}

function globalAverage(row, categories) {
  const values = Object.values(categories).map((columns) => average(row, columns));
  const valid = values.filter(Number.isFinite);
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : NaN;
}

function escapeCsv(value) {
  const string = String(value);
  return /[",\n]/.test(string) ? `"${string.replaceAll('"', '""')}"` : string;
}

function toCsv(headers, rows) {
  return [headers, ...rows]
    .map((row) => row.map(escapeCsv).join(","))
    .join("\n") + "\n";
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const revision = options.sourceRoot
    ? execFileSync("git", ["-C", options.sourceRoot, "rev-parse", "HEAD"], { encoding: "utf8" }).trim()
    : await githubRevision();
  const readSource = options.sourceRoot
    ? (relativePath) => readFile(path.join(options.sourceRoot, relativePath), "utf8")
    : (relativePath) => fetchText(
        `https://raw.githubusercontent.com/${REPOSITORY}/${revision}/${relativePath}`,
      );

  const constants = await readSource("src/lib/constants.js");
  const releasesBlock = constants.match(/RELEASES\s*=\s*\[([\s\S]*?)\]/)?.[1] ?? "";
  const releases = [...releasesBlock.matchAll(/"(\d{4}-\d{2}-\d{2})"/g)].map((match) => match[1]);
  const release = options.release ?? releases.at(-1);
  if (!release || !releases.includes(release)) throw new Error(`unknown release: ${release}`);
  const releaseKey = release.replaceAll("-", "_");

  const [tableText, categoriesText, modelLinksSource] = await Promise.all([
    readSource(`public/table_${releaseKey}.csv`),
    readSource(`public/categories_${releaseKey}.json`),
    readSource("src/Table/modelLinks.js"),
  ]);
  const categories = JSON.parse(categoriesText);
  const rows = parseCsv(tableText);
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(modelLinksSource).toString("base64")}`;
  const { getModelInfo, getVariantGroup } = await import(moduleUrl);

  const displayNameCounts = new Map();
  for (const row of rows) {
    const name = getModelInfo(row.model)?.displayName;
    if (name) displayNameCounts.set(name, (displayNameCounts.get(name) ?? 0) + 1);
  }

  const sortedRows = rows
    .filter((row) => getModelInfo(row.model))
    .sort((left, right) => {
      const scoreDifference = globalAverage(right, categories) - globalAverage(left, categories);
      return scoreDifference || left.model.localeCompare(right.model);
    });

  // The live page hides effort variants by default and keeps the best variant.
  const bestByGroup = new Map();
  for (const row of sortedRows) {
    const groupKey = getVariantGroup(row.model)?.baseName ?? row.model;
    const score = globalAverage(row, categories);
    const current = bestByGroup.get(groupKey);
    if (!current || score > current.score) bestByGroup.set(groupKey, { row, score });
  }

  const firstCategory = Object.keys(categories)[0];
  const outputRows = [...bestByGroup.values()].map(({ row }) => {
    const info = getModelInfo(row.model);
    let model = info.displayName ?? row.model;
    if ((displayNameCounts.get(model) ?? 0) > 1 && info.version !== undefined) {
      model = `${model} (${info.version})`;
    }
    return [
      model,
      info.organization ?? "",
      globalAverage(row, categories).toFixed(2),
      average(row, categories[firstCategory]).toFixed(2),
    ];
  });

  const output = path.resolve(options.output);
  const metadataOutput = output.replace(/\.csv$/i, ".metadata.json");
  await writeFile(
    output,
    toCsv(["Model", "Organization", "Global Average", `${firstCategory} Average`], outputRows),
  );
  await writeFile(metadataOutput, JSON.stringify({
    source: "https://livebench.ai/#/",
    source_repository: `https://github.com/${REPOSITORY}`,
    source_revision: revision,
    release,
    retrieved_at: new Date().toISOString(),
    view: "default filters; organization shown; effort variants collapsed",
    row_count: outputRows.length,
  }, null, 2) + "\n");
  console.log(`wrote ${outputRows.length} models to ${output}`);
  console.log(`wrote provenance to ${metadataOutput}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
