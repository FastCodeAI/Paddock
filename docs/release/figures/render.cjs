#!/usr/bin/env node
// Render each figure source in src/ to a PNG beside this script.
//
// A source is an HTML page holding one `.figure` element. The PNG is a screenshot
// of that element at 2x, so it stays sharp when a viewer scales it.
//
// Puppeteer is not a paddock dependency. Install it anywhere outside the repo and
// point NODE_PATH at it:
//
//   npm install --prefix <scratch> puppeteer@19
//   NODE_PATH=<scratch>/node_modules node docs/release/figures/render.cjs [name ...]
//
// Set CHROME to use a Chrome or Chromium other than the one puppeteer downloaded.
//
// It fails, and writes nothing for that figure, when the figure breaks its budget:
// - not 1280 px wide, or taller than 860 px;
// - any text overflowing its box, which is how a figure usually breaks after an
//   edit.
// Set FIGURE_DRAFT=1 to write the PNG anyway, to look at what broke.

const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const HERE = __dirname;
const SRC = path.join(HERE, "src");
const WIDTH = 1280;
const MAX_HEIGHT = 860;

async function render(browser, name) {
  const page = await browser.newPage();
  await page.setViewport({ width: WIDTH, height: 1000, deviceScaleFactor: 2 });
  await page.goto("file://" + path.join(SRC, name + ".html"), { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);

  const fig = await page.$(".figure");
  if (!fig) return [`${name}: no .figure element`];
  const box = await fig.boundingBox();
  const problems = [];
  if (Math.round(box.width) !== WIDTH) problems.push(`width ${box.width}, not ${WIDTH}`);
  if (box.height > MAX_HEIGHT) problems.push(`height ${Math.round(box.height)} > ${MAX_HEIGHT}`);

  const overflowing = await page.evaluate(() =>
    [...document.querySelectorAll(".figure *")]
      .filter((el) => [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim()))
      .filter((el) => el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 1)
      .map((el) => `text overflows: "${el.textContent.trim().slice(0, 50)}"`),
  );
  problems.push(...overflowing);

  if (problems.length === 0 || process.env.FIGURE_DRAFT) {
    await fig.screenshot({ path: path.join(HERE, name + ".png") });
  }
  await page.close();
  return problems.map((p) => `${name}: ${p}`);
}

(async () => {
  const wanted = process.argv.slice(2);
  const names = fs
    .readdirSync(SRC)
    .filter((f) => f.endsWith(".html"))
    .map((f) => f.slice(0, -".html".length))
    .filter((n) => wanted.length === 0 || wanted.includes(n))
    .sort();
  const browser = await puppeteer.launch({
    headless: "new",
    executablePath: process.env.CHROME || undefined,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  let failed = 0;
  for (const name of names) {
    const problems = await render(browser, name);
    if (problems.length) {
      failed += 1;
      problems.forEach((p) => console.log("FAIL " + p));
    } else {
      console.log("ok   " + name);
    }
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
