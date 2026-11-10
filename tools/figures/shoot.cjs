// Screenshot each HTML file's #fig element to PNG at 2x.  node shoot.cjs HTML_DIR OUT_DIR
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const fs = require('fs');
const path = require('path');
const [src, out] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch(process.env.FIG_CHROME ? { channel: process.env.FIG_CHROME } : {});
  const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
  for (const f of fs.readdirSync(src).filter(f => f.endsWith('.html')).sort()) {
    await p.goto('file://' + path.resolve(src, f), { waitUntil: 'networkidle' });
    await p.evaluate(() => document.fonts.ready);
    await p.waitForTimeout(150);
    const dest = path.join(out, f.replace('.html', '.png'));
    await (await p.$('#fig')).screenshot({ path: dest });
    console.log(dest, (fs.statSync(dest).size / 1024).toFixed(0) + 'KB');
  }
  await b.close();
})();
