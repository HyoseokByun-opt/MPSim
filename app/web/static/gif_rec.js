/* Animated GIF of a result: frames from a canvas (or an image URL), each
   reduced to its own 256-colour palette and written with gifenc (MIT,
   vendor/gifenc.esm.js, loaded when first needed), then saved as a file.
   Used by the 3D view (animated field lines, routes, or a turn around the
   RVE), the particle-dynamics player and the underfill flow front. */
(function () {
  "use strict";
  let lib = null;
  async function gifenc() {
    if (!lib) lib = await import("/static/vendor/gifenc.esm.js");
    return lib;
  }
  async function toCanvas(src, maxWidth, top) {
    let drawable = src, w, h;
    if (typeof src === "string") {
      const img = new Image();
      img.src = src;
      await img.decode();
      drawable = img;
      w = img.naturalWidth; h = img.naturalHeight;
    } else { w = src.width; h = src.height; }
    const s = Math.min(1, maxWidth / Math.max(w, 1));
    const c = document.createElement("canvas");
    c.width = Math.max(2, Math.round(w * s));
    c.height = Math.max(2, Math.round(h * s)) + top;
    const g = c.getContext("2d", { willReadFrequently: true });
    g.fillStyle = "#ffffff";
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(drawable, 0, top, c.width, c.height - top);
    return { c, g };
  }
  // a vertical colour bar drawn into a frame (the page's own bar is HTML and
  // not part of the picture): stops [[t, r, g, b] 0..1] or a function t -> [r, g, b]
  function colourBar(g, x, y, h, title, lo, hi, colour, fmt) {
    const w = 14;
    for (let i = 0; i < h; i++) {
      const t = 1 - i / (h - 1);
      const [r, gg, b] = colour(t);
      g.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(gg * 255)},${Math.round(b * 255)})`;
      g.fillRect(x, y + i, w, 1);
    }
    g.strokeStyle = "#8c959f"; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    g.fillStyle = "#1f2328"; g.font = "12px system-ui, sans-serif"; g.textAlign = "left";
    g.fillText(fmt(hi), x + w + 5, y + 9);
    g.fillText(fmt(lo), x + w + 5, y + h);
    // a long title moves left so that it stays in the picture
    g.font = "bold 12px system-ui, sans-serif";
    g.fillText(title, Math.max(4, Math.min(x - 2, g.canvas.width - 6 - g.measureText(title).width)), y - 8);
  }
  function save(blob, filename) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 3000);
  }
  window.GifRec = {
    colourBar,
    /* frame(k) -> canvas or image URL; overlay(g, w, h, k) draws onto the
       scaled frame (legend, title), top px of white above the picture for
       a caption; delay in ms between frames */
    async record({ count, frame, overlay, delay = 80, filename = "animation.gif", maxWidth = 800, top = 0, onProgress }) {
      const { GIFEncoder, quantize, applyPalette } = await gifenc();
      const enc = GIFEncoder();
      for (let k = 0; k < count; k++) {
        const { c, g } = await toCanvas(await frame(k), maxWidth, top);
        if (overlay) overlay(g, c.width, c.height, k);
        const rgba = g.getImageData(0, 0, c.width, c.height).data;
        const palette = quantize(rgba, 256);
        enc.writeFrame(applyPalette(rgba, palette), c.width, c.height, { palette, delay });
        if (onProgress) onProgress(k + 1, count);
        if (k % 4 === 3) await new Promise((r) => setTimeout(r, 0));      // let the page breathe
      }
      enc.finish();
      const blob = new Blob([enc.bytes()], { type: "image/gif" });
      save(blob, filename);
      return blob.size;
    },
  };
})();
