// Print an assembled user guide to PDF with headless Chrome.
//
//   npm install puppeteer-core
//   node make_pdf.js user_guide_en.html ../USER_GUIDE_EN.pdf "MPSim v5 · User Guide"
//
// CHROME may name the browser executable (default: the usual Windows path).
// Every figure is kept whole on one page: a figure taller than the printable
// area (a long panel capture, a stack of result boxes) is narrowed, keeping
// its proportions, until it fits - the browser would otherwise cut it at the
// page end and carry the rest to the next page.
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');

const src = path.resolve(process.argv[2] || 'user_guide.html');
const out = path.resolve(process.argv[3] || 'user_guide.pdf');
const label = process.argv[4] || '';
const chrome = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const MM = 96 / 25.4;              // CSS px per mm
const PAGE_W = 210, PAGE_H = 297;  // A4
const MARGIN = { top: 12, bottom: 14, left: 10, right: 10 };
const FIG_MAX_H = 228;             // mm of image height that leaves room for a caption

(async () => {
  const browser = await puppeteer.launch({ executablePath: chrome, headless: true,
    args: ['--disable-gpu', '--font-render-hinting=none'] });
  const page = await browser.newPage();
  // lay the page out at the printable width, so that what is measured below
  // is what is printed
  await page.setViewport({ width: Math.round((PAGE_W - MARGIN.left - MARGIN.right) * MM), height: 1200 });
  await page.goto(pathToFileURL(src).href, { waitUntil: 'networkidle0', timeout: 300000 });
  await page.evaluate(() => document.querySelectorAll('img[loading="lazy"]').forEach((i) => i.removeAttribute('loading')));
  await page.evaluate(async () => {
    await Promise.all([...document.images].filter((i) => !i.complete).map(
      (i) => new Promise((res) => { i.onload = i.onerror = res; })));
  });
  await page.addStyleTag({ content: `
    @page { size: A4; margin: ${MARGIN.top}mm ${MARGIN.right}mm ${MARGIN.bottom}mm ${MARGIN.left}mm; }
    body { background: #fff; padding: 0; margin: 0; }
    figure, figure img, figure .g2 > div, figure .g3 > div { break-inside: avoid; page-break-inside: avoid; }
    figcaption { break-before: avoid; page-break-before: avoid; }
    tr, .note, .warn, .code, pre { break-inside: avoid; page-break-inside: avoid; }
    h2, h3, h4 { break-after: avoid; page-break-after: avoid; }
    .g { grid-template-columns: 1fr; }
  ` });
  await page.emulateMediaType('print');
  await sleep(1500);
  const narrowed = await page.evaluate((maxH) => {
    let n = 0;
    document.querySelectorAll('figure img').forEach((im) => {
      const r = im.getBoundingClientRect();
      if (r.height > maxH) {
        const w = Math.floor(r.width * maxH / r.height);
        im.style.width = w + 'px';
        im.style.maxWidth = w + 'px';
        im.style.marginLeft = 'auto';
        im.style.marginRight = 'auto';
        n += 1;
      }
    });
    return n;
  }, FIG_MAX_H * MM);
  await sleep(800);
  await page.pdf({
    path: out, format: 'A4', printBackground: true, timeout: 600000,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate:
      `<div style="width:100%;font-size:8px;color:#57606a;padding:0 ${MARGIN.left + 2}mm;
        font-family:'Segoe UI','Malgun Gothic',sans-serif;display:flex;justify-content:space-between">
        <span>${label.replace(/[<>&]/g, '')}</span>
        <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
      </div>`,
    margin: { top: MARGIN.top + 'mm', bottom: MARGIN.bottom + 'mm', left: MARGIN.left + 'mm', right: MARGIN.right + 'mm' },
  });
  console.log(`wrote ${out} (${Math.round(fs.statSync(out).size / 1024)} KB, ${narrowed} tall figure(s) fitted to the page)`);
  await browser.close();
})().catch((e) => { console.error('FATAL ' + e.message); process.exit(1); });
