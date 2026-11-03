/* SmartBeam Structural Playground — SVG beam editor (plain JS, Streamlit component protocol v1).
 * All model numbers are SI (m, N, N/m, N·m) exactly as in the BeamModel JSON schema; units are display-only.
 * Sign conventions follow docs/ARCHITECTURE.md: point-load angle CCW from +x (−90° = down), moment CCW +.
 */
"use strict";

// ------------------------------------------------------------------ Streamlit protocol
const post = (type, data) => window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data), "*");
const FRAME_H = 762;

// ------------------------------------------------------------------ state
const S = {
  cfg: null, rev: null, version: 0, synced: true,
  selected: null, undo: [], redo: [],
  snap: true, grid: 0.05, live: false,
  view: null, mode: "config",
  results: null, resultsCurrent: false, error: null,
  presets: {}, materials: {},
  drag: null, hoverX: null, syncTimer: null, sendCount: 0,
};
const UNIT = { N: 1, kN: 1e3, m: 1, mm: 1e-3, "N/m": 1, "kN/m": 1e3, "N·m": 1, "kN·m": 1e3 };
const C = { cyan: "#2D6CDF", amber: "#DCC9A1", green: "#5FBF8F", red: "#E06A5A", violet: "#F7F3E8",
  text: "#F7F3E8", muted: "#B8AE95", border: "#1F4A80", grid: "#0E2847", bg: "#0A1B33", blue: "#7EA2E3" };
const $ = (id) => document.getElementById(id);
const svg = $("cv"), stage = $("stage"), props = $("props");
const clone = (o) => JSON.parse(JSON.stringify(o));
const r6 = (v) => Math.round(v * 1e6) / 1e6;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------------ number formatting
function num(v, sig = 4) {
  if (!isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) return v.toExponential(sig - 1).replace(/\.?0+e/, "e");
  return String(parseFloat(v.toPrecision(sig)));
}
const units = () => S.cfg.units;
const fF = (n) => `${num(n / UNIT[units().force])} ${units().force}`;
const fL = (m) => `${num(m / UNIT[units().length])} ${units().length}`;
const fW = (w) => `${num(w / UNIT[units().dist])} ${units().dist}`;
const fM = (m) => `${num(m / UNIT[units().moment])} ${units().moment}`;

// ------------------------------------------------------------------ geometry helpers
const W = () => svg.clientWidth || 800;
const H = () => svg.clientHeight || 680;
const sx = (x) => S.view.ox + x * S.view.scale;
const wx = (px) => (px - S.view.ox) / S.view.scale;
const BY = () => S.view.oy;           // beam axis (screen y)
const BH = 12;                        // drawn beam thickness, px (not to scale)

function fit() {
  const L = S.cfg.L, w = W(), m = Math.max(70, w * 0.09);
  S.view = { scale: (w - 2 * m) / L, ox: m, oy: Math.round(H() * 0.31) };
  S.userView = false;
}
function snapX(x) {
  const v = S.snap ? Math.round(x / S.grid) * S.grid : Math.round(x * 1e4) / 1e4;
  return r6(v);
}
function svgPoint(e) {
  const b = svg.getBoundingClientRect();
  return { x: e.clientX - b.left, y: e.clientY - b.top };
}

// ------------------------------------------------------------------ config helpers
let idSeq = 1;
function newId(prefix) {
  const used = new Set([...S.cfg.supports, ...S.cfg.loads].map((o) => o.id));
  let id;
  do { id = prefix + idSeq++; } while (used.has(id));
  return id;
}
function ensureIds(cfg) {
  const seen = new Set();
  for (const [arr, p] of [[cfg.supports, "s"], [cfg.loads, "l"]]) {
    for (const o of arr) {
      if (!o.id || seen.has(o.id)) { o.id = p + "x" + (idSeq++); }
      seen.add(o.id);
    }
  }
  cfg.units = Object.assign({ force: "kN", length: "m", dist: "kN/m", moment: "kN·m" }, cfg.units || {});
  return cfg;
}
function find(id) {
  return S.cfg.supports.find((o) => o.id === id) || S.cfg.loads.find((o) => o.id === id) || null;
}
function isSupport(o) { return !!o && !o.kind; }
function modelOnly(cfg) {
  const d = clone(cfg);
  d.supports.forEach((s) => delete s.id);
  d.loads.forEach((l) => delete l.id);
  d.playground = { units: d.units };
  delete d.units; delete d.material_key;
  return d;
}
function supportAt(x, exceptId) {
  return S.cfg.supports.some((s) => s.id !== exceptId && Math.abs(s.x - x) < 1e-9);
}
function freeSupportX(x, exceptId) {
  // nearest grid position to x not occupied by another support
  const step = S.snap ? S.grid : S.cfg.L / 100;
  for (let k = 0; k < 400; k++) {
    for (const c of [x + k * step, x - k * step]) {
      const v = r6(clamp(c, 0, S.cfg.L));
      if (!supportAt(v, exceptId)) return v;
    }
  }
  return null;
}
function pointKind(p) {
  const a = ((p.angle_deg % 360) + 360) % 360;
  if (Math.abs(a - 270) < 1e-9) return "down";
  if (Math.abs(a - 90) < 1e-9) return "up";
  if (a < 1e-9) return "axial_r";
  if (Math.abs(a - 180) < 1e-9) return "axial_l";
  return "inclined";
}
function distKind(d, L) {
  const eq = Math.abs(d.w1 - d.w2) <= 1e-9 * Math.max(1, Math.abs(d.w1), Math.abs(d.w2));
  if (eq && d.x1 <= 1e-9 && Math.abs(d.x2 - L) <= 1e-9) return "udl";
  if (eq) return "partial";
  if (d.w1 === 0 || d.w2 === 0) return "triangular";
  return "linear";
}
const DIST_LABEL = { udl: "Full UDL", partial: "Partial UDL", linear: "Linearly varying", triangular: "Triangular" };
const POINT_LABEL = { down: "Vertical ↓", up: "Vertical ↑", axial_r: "Axial →", axial_l: "Axial ←", inclined: "Inclined" };
const SUP_LABEL = { fixed: "Fixed", pin: "Pinned", roller: "Roller" };
const DOFS = { fixed: ["u", "v", "theta"], pin: ["u", "v"], roller: ["v"] };

function itemLabel(o) {
  if (isSupport(o)) return `${SUP_LABEL[o.type]} support`;
  if (o.kind === "point") return `Point load · ${POINT_LABEL[pointKind(o)]}`;
  if (o.kind === "distributed") return DIST_LABEL[distKind(o, S.cfg.L)];
  return `Moment · ${o.M < 0 ? "CW" : "CCW"}`;
}

// ------------------------------------------------------------------ history + sync
function snapshot() { return JSON.stringify(S.cfg); }
function pushUndo(snap) {
  S.undo.push(snap);
  if (S.undo.length > 200) S.undo.shift();
  S.redo = [];
}
function changed() {
  S.version += 1;
  S.synced = false;
  S.resultsCurrent = false;
  S.error = null;
  scheduleSync();
  renderAll();
}
function commit(mutate) {
  const before = snapshot();
  mutate();
  if (snapshot() === before) { renderAll(); return; }
  pushUndo(before);
  changed();
}
function undo() {
  if (!S.undo.length) return;
  S.redo.push(snapshot());
  S.cfg = JSON.parse(S.undo.pop());
  if (S.selected && S.selected !== "beam" && !find(S.selected)) S.selected = null;
  changed();
}
function redo() {
  if (!S.redo.length) return;
  S.undo.push(snapshot());
  S.cfg = JSON.parse(S.redo.pop());
  if (S.selected && S.selected !== "beam" && !find(S.selected)) S.selected = null;
  changed();
}
function sendValue(action) {
  clearTimeout(S.syncTimer);
  S.sendCount += 1;
  post("streamlit:setComponentValue", {
    value: { config: S.cfg, action, nonce: `${Date.now()}-${S.sendCount}`, version: S.version },
    dataType: "json",
  });
  S.synced = true;
  renderStatus(action === "edit" ? "" : "Analysing…");
}
function scheduleSync() {
  clearTimeout(S.syncTimer);
  S.syncTimer = setTimeout(() => sendValue(S.live ? "live" : "edit"), S.live ? 400 : 700);
}

// ------------------------------------------------------------------ toast
let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
}

// ------------------------------------------------------------------ adding items
function addTool(tool, x) {
  const L = S.cfg.L;
  if (tool === "beam") { S.selected = "beam"; renderAll(); return; }
  if (x == null) x = L / 2;
  x = snapX(clamp(x, 0, L));
  let item = null;
  if (tool === "fixed" || tool === "pin" || tool === "roller") {
    if (x == null || supportAt(x)) {
      const free = freeSupportX(x);
      if (free == null) { toast("No free position for another support."); return; }
      if (supportAt(x)) toast("A support already sits there — placed at the nearest free grid point.");
      x = free;
    }
    item = { id: newId("s"), type: tool, x };
    commit(() => S.cfg.supports.push(item));
  } else {
    const span = (f) => {
      const half = Math.max(S.snap ? S.grid : 0, (L * f) / 2);
      let a = snapX(clamp(x - half, 0, L)), b = snapX(clamp(x + half, 0, L));
      if (b - a < 1e-9) { a = 0; b = L; }
      return [a, b];
    };
    const P = 1000;
    if (tool === "point_down") item = { kind: "point", x, magnitude: P, angle_deg: -90 };
    if (tool === "point_up") item = { kind: "point", x, magnitude: P, angle_deg: 90 };
    if (tool === "axial") item = { kind: "point", x, magnitude: P, angle_deg: 0 };
    if (tool === "inclined") item = { kind: "point", x, magnitude: P, angle_deg: -60 };
    if (tool === "udl") item = { kind: "distributed", x1: 0, x2: L, w1: P, w2: P, direction: "down" };
    if (tool === "partial_udl") { const [a, b] = span(0.3); item = { kind: "distributed", x1: a, x2: b, w1: P, w2: P, direction: "down" }; }
    if (tool === "linear") { const [a, b] = span(0.5); item = { kind: "distributed", x1: a, x2: b, w1: P / 2, w2: 1.5 * P, direction: "down" }; }
    if (tool === "triangular") { const [a, b] = span(0.5); item = { kind: "distributed", x1: a, x2: b, w1: 0, w2: P, direction: "down" }; }
    if (tool === "moment_cw") item = { kind: "moment", x, M: -P };
    if (tool === "moment_ccw") item = { kind: "moment", x, M: P };
    if (!item) return;
    item = Object.assign({ id: newId("l") }, item);
    commit(() => S.cfg.loads.push(item));
  }
  S.selected = item.id;
  renderAll();
}

function deleteSelected() {
  const id = S.selected;
  if (!id) return;
  if (id === "beam") { toast("The beam cannot be deleted — use Clear to remove supports and loads."); return; }
  commit(() => {
    S.cfg.supports = S.cfg.supports.filter((o) => o.id !== id);
    S.cfg.loads = S.cfg.loads.filter((o) => o.id !== id);
  });
  S.selected = null;
  renderAll();
}

function duplicateSelected() {
  const o = find(S.selected);
  if (!o) { toast("Select a support or load to duplicate."); return; }
  const L = S.cfg.L, step = S.snap ? S.grid : L / 20;
  const c = clone(o);
  if (isSupport(o)) {
    const x = freeSupportX(r6(clamp(o.x + step, 0, L)), null);
    if (x == null) { toast("No free position for another support."); return; }
    c.id = newId("s"); c.x = x;
    commit(() => S.cfg.supports.push(c));
  } else {
    c.id = newId("l");
    if (c.kind === "distributed") {
      const len = c.x2 - c.x1;
      const shift = c.x2 + len <= L + 1e-9 ? len : Math.min(step, L - c.x2);
      c.x1 = r6(c.x1 + shift); c.x2 = r6(c.x2 + shift);
    } else {
      c.x = r6(c.x + step <= L + 1e-9 ? c.x + step : c.x - step);
    }
    commit(() => S.cfg.loads.push(c));
  }
  S.selected = c.id;
  renderAll();
}

function setLength(cfg, newL) {
  // Move items that sat on the right end with it; clamp everything else onto the new span.
  const oldL = cfg.L, eps = 1e-9;
  newL = r6(newL);
  const mv = (x) => (Math.abs(x - oldL) < eps ? newL : Math.min(x, newL));
  cfg.L = newL;
  for (const s of cfg.supports) s.x = r6(mv(s.x));
  // supports pushed onto the same point: keep the first
  const seen = [];
  cfg.supports = cfg.supports.filter((s) => (seen.some((x) => Math.abs(x - s.x) < eps) ? false : seen.push(s.x)));
  for (const l of cfg.loads) {
    if (l.kind === "distributed") {
      l.x1 = r6(mv(l.x1)); l.x2 = r6(mv(l.x2));
      if (l.x2 - l.x1 < 1e-6) l.x1 = r6(Math.max(0, l.x2 - Math.min(newL / 10, S.grid)));
    } else l.x = r6(mv(l.x));
  }
}

// ------------------------------------------------------------------ SVG drawing primitives
function arrow(x1, y1, x2, y2, color, w = 2, head = 9) {
  const a = Math.atan2(y2 - y1, x2 - x1), hw = head * 0.55;
  const bx = x2 - head * Math.cos(a), by = y2 - head * Math.sin(a);
  const p1 = `${bx + hw * Math.sin(a)},${by - hw * Math.cos(a)}`, p2 = `${bx - hw * Math.sin(a)},${by + hw * Math.cos(a)}`;
  return `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" stroke="${color}" stroke-width="${w}" stroke-linecap="round"/>` +
    `<polygon points="${x2},${y2} ${p1} ${p2}" fill="${color}"/>`;
}
function momentArc(cx, cy, r, ccw, color, w = 2) {
  // 260° arc; ccw = counter-clockwise as seen on screen (y up)
  const t0 = (-40 * Math.PI) / 180, t1 = (220 * Math.PI) / 180, n = 28, pts = [];
  for (let i = 0; i <= n; i++) {
    const t = ccw ? t0 + ((t1 - t0) * i) / n : t1 - ((t1 - t0) * i) / n;
    pts.push([cx + r * Math.cos(t), cy - r * Math.sin(t)]);
  }
  const [ex, ey] = pts[n], [px, py] = pts[n - 2];
  return `<polyline points="${pts.map((p) => p.join(",")).join(" ")}" fill="none" stroke="${color}" stroke-width="${w}" stroke-linecap="round"/>` +
    arrow(px, py, ex, ey, color, w, 9).replace(/<line[^>]*>/, "");
}
function text(x, y, s, opts = {}) {
  const anchor = opts.anchor || "middle", fill = opts.fill || C.text, size = opts.size || 11;
  const weight = opts.bold ? ' font-weight="700"' : "";
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${fill}" font-size="${size}"${weight}` +
    ` paint-order="stroke" stroke="${C.bg}" stroke-width="3" stroke-linejoin="round">${esc(s)}</text>`;
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
}

// ------------------------------------------------------------------ glyphs
function drawSupport(s, sel) {
  const X = sx(s.x), y0 = BY() + BH / 2, L = S.cfg.L, col = sel ? C.cyan : C.text;
  let g = "";
  if (s.type === "fixed") {
    const left = s.x < 1e-9 ? true : Math.abs(s.x - L) < 1e-9 ? false : null;
    const h = 34, dir = left === false ? 1 : -1;
    g += `<line x1="${X}" y1="${BY() - h}" x2="${X}" y2="${BY() + h}" stroke="${col}" stroke-width="3"/>`;
    const sides = left === null ? [-1, 1] : [dir];
    for (const d of sides) for (let k = -h + 4; k <= h; k += 8)
      g += `<line x1="${X}" y1="${BY() + k}" x2="${X + 9 * d}" y2="${BY() + k + 8}" stroke="${C.muted}" stroke-width="1.2"/>`;
  } else {
    const th = s.type === "pin" ? 22 : 16, half = 13;
    g += `<polygon points="${X},${y0} ${X - half},${y0 + th} ${X + half},${y0 + th}" fill="${C.bg}" stroke="${col}" stroke-width="2"/>`;
    g += `<circle cx="${X}" cy="${y0}" r="3" fill="${C.bg}" stroke="${col}" stroke-width="1.6"/>`;
    let gy = y0 + th;
    if (s.type === "roller") {
      g += `<circle cx="${X - 7}" cy="${gy + 4.5}" r="4" fill="none" stroke="${col}" stroke-width="1.6"/>` +
        `<circle cx="${X + 7}" cy="${gy + 4.5}" r="4" fill="none" stroke="${col}" stroke-width="1.6"/>`;
      gy += 9;
    }
    g += `<line x1="${X - 20}" y1="${gy}" x2="${X + 20}" y2="${gy}" stroke="${col}" stroke-width="2"/>`;
    for (let k = -18; k <= 14; k += 7) g += `<line x1="${X + k}" y1="${gy + 7}" x2="${X + k + 6}" y2="${gy}" stroke="${C.muted}" stroke-width="1.1"/>`;
  }
  const hit = `<rect x="${X - 22}" y="${BY() - 36}" width="44" height="78" fill="transparent"/>`;
  return `<g class="item" data-id="${s.id}">${hit}${g}${sel ? selRing(X - 24, BY() - 38, 48, 82) : ""}</g>`;
}
function selRing(x, y, w, h) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7" fill="none" stroke="${C.cyan}" stroke-dasharray="4 3" stroke-width="1.2"/>`;
}
function drawPoint(p, sel) {
  const X = sx(p.x), a = (p.angle_deg * Math.PI) / 180, len = 58;
  const dx = Math.cos(a), dyUp = Math.sin(a); // direction in y-up world
  // the head touches the beam surface the force pushes on
  const hy = dyUp < -1e-9 ? BY() - BH / 2 : dyUp > 1e-9 ? BY() + BH / 2 : BY();
  const hx = X;
  const tx = hx - dx * len, ty = hy + dyUp * len; // screen y is flipped
  const col = sel ? C.cyan : C.amber;
  const kind = pointKind(p);
  const lbl = fF(p.magnitude) + (kind === "inclined" ? ` ∠${num(p.angle_deg)}°` : "");
  const ly = ty + (dyUp < 0 ? -8 : 16);
  const ax = Math.abs(dyUp) <= 1e-9;
  const hit = `<circle cx="${(hx + tx) / 2}" cy="${(hy + ty) / 2}" r="${len / 2 + 6}" fill="transparent"/>`;
  return `<g class="item" data-id="${p.id}">${hit}${arrow(tx, ty, hx, hy, col, 2.4, 11)}` +
    text(tx, ax ? hy - 10 : ly, lbl, { fill: col }) + `</g>`;
}
function drawDistributed(d, wmax, sel) {
  const X1 = sx(d.x1), X2 = sx(d.x2), down = d.direction !== "up", col = sel ? C.cyan : C.amber;
  const hOf = (w) => 12 + 52 * (Math.abs(w) / wmax);
  const base = down ? BY() - BH / 2 : BY() + BH / 2, sgn = down ? -1 : 1;
  const n = Math.max(2, Math.min(40, Math.round((X2 - X1) / 26)));
  let g = "";
  const top = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n, X = X1 + (X2 - X1) * t, w = d.w1 + (d.w2 - d.w1) * t, h = hOf(w);
    const yT = base + sgn * h;
    top.push(`${X},${yT}`);
    if (Math.abs(w) > 1e-12 && h > 13) g += arrow(X, yT, X, base, col, 1.5, 7);
  }
  const poly = `<polygon points="${X1},${base} ${top.join(" ")} ${X2},${base}" fill="${col}" fill-opacity="0.07"/>`;
  g = poly + g + `<polyline points="${top.join(" ")}" fill="none" stroke="${col}" stroke-width="2"/>`;
  const ty = (w) => base + sgn * (hOf(w) + 9) + (down ? 0 : 4);
  const same = Math.abs(d.w1 - d.w2) < 1e-12;
  if (same) g += text((X1 + X2) / 2, ty(d.w1), fW(d.w1), { fill: col });
  else {
    g += text(X1, ty(d.w1), fW(d.w1), { fill: col, anchor: "start" });
    g += text(X2, ty(d.w2), fW(d.w2), { fill: col, anchor: "end" });
  }
  const yTop = Math.min(base, base + sgn * hOf(Math.max(d.w1, d.w2))) - 4;
  const hit = `<rect x="${X1}" y="${down ? yTop : base}" width="${Math.max(6, X2 - X1)}" height="${hOf(Math.max(d.w1, d.w2)) + 6}" fill="transparent"/>`;
  let handles = "";
  if (sel) {
    for (const [k, X, w] of [["x1", X1, d.w1], ["x2", X2, d.w2]]) {
      const y = base + sgn * hOf(w);
      handles += `<rect class="handle" data-handle="${k}" data-id="${d.id}" x="${X - 6}" y="${y - 6}" width="12" height="12" rx="3" fill="${C.bg}" stroke="${C.cyan}" stroke-width="2"/>`;
    }
  }
  return `<g class="item" data-id="${d.id}">${hit}${g}</g>${handles}`;
}
function drawMoment(m, sel) {
  const X = sx(m.x), col = sel ? C.cyan : C.violet, ccw = m.M > 0;
  return `<g class="item" data-id="${m.id}"><circle cx="${X}" cy="${BY()}" r="30" fill="transparent"/>` +
    momentArc(X, BY(), 24, ccw, col, 2.2) + `<circle cx="${X}" cy="${BY()}" r="3" fill="${col}"/>` +
    text(X, BY() - 34, `${fM(Math.abs(m.M))} ${ccw ? "↺" : "↻"}`, { fill: col }) + `</g>`;
}

// ------------------------------------------------------------------ result overlays
function res() { return S.results && S.resultsCurrent ? S.results : null; }

function drawReactions(r) {
  let g = "";
  for (const re of r.reactions) {
    const X = sx(re.x), y0 = BY() + BH / 2 + 44, y1 = y0 + 34;
    if (Math.abs(re.Fy) > 1e-9) {
      g += re.Fy > 0 ? arrow(X, y1, X, y0, C.green, 2.4, 10) : arrow(X, y0, X, y1, C.green, 2.4, 10);
      g += text(X, y1 + 14, `R${re.type === "fixed" ? "y" : ""} = ${fF(Math.abs(re.Fy))}`, { fill: C.green, bold: true });
    }
    if (Math.abs(re.Fx) > 1e-9) {
      const yA = BY() + 4, a = X - (re.Fx > 0 ? 64 : -64), b = X - (re.Fx > 0 ? 18 : -18);
      g += arrow(a, yA, b, yA, C.green, 2.4, 10) + text((a + b) / 2, yA + 16, `Rx = ${fF(Math.abs(re.Fx))}`, { fill: C.green });
    }
    if (Math.abs(re.M) > 1e-9) {
      g += momentArc(X, BY(), 40, re.M > 0, C.green, 2) + text(X, BY() - 48, `M = ${fM(Math.abs(re.M))} ${re.M > 0 ? "↺" : "↻"}`, { fill: C.green, bold: true });
    }
  }
  return g;
}

function band(xs, ys, color, title, fmt, opts = {}) {
  const base = BY() + 236, amp = 78;
  let m = opts.scaleMax || 0;
  if (!m) for (const y of ys) m = Math.max(m, Math.abs(y));
  const k = m > 0 ? amp / m : 0;
  const pts = xs.map((x, i) => `${sx(x).toFixed(1)},${(base - ys[i] * k).toFixed(1)}`);
  let g = `<line x1="${sx(0)}" y1="${base}" x2="${sx(S.cfg.L)}" y2="${base}" stroke="${C.muted}" stroke-width="1"/>`;
  if (!opts.noFill) g += `<polygon points="${sx(xs[0])},${base} ${pts.join(" ")} ${sx(xs[xs.length - 1])},${base}" fill="${color}" fill-opacity="0.16"/>`;
  g += `<polyline points="${pts.join(" ")}" fill="none" stroke="${color}" stroke-width="2"/>`;
  if (title) g += text(Math.max(8, sx(0) - 8), base - amp - 30, title, { anchor: "start", fill: C.muted });
  if (m > 0) {
    let iMax = 0, iMin = 0;
    ys.forEach((y, i) => { if (y > ys[iMax]) iMax = i; if (y < ys[iMin]) iMin = i; });
    for (const i of new Set([iMax, iMin])) {
      if (Math.abs(ys[i]) < 1e-9 * m) continue;
      const X = sx(xs[i]), Y = base - ys[i] * k;
      g += `<circle cx="${X}" cy="${Y}" r="3.5" fill="${color}"/>` +
        text(X, ys[i] >= 0 ? Y - 9 : Y + 17, `${fmt(ys[i])} @ ${fL(xs[i])}`, { fill: color, bold: true });
    }
  }
  return g;
}

function deflected(r) {
  let m = 0;
  for (const v of r.v) m = Math.max(m, Math.abs(v));
  if (m === 0) return text(sx(S.cfg.L / 2), BY() - 70, "No deflection", { fill: C.muted });
  const k = 70 / m, scale = (k / S.view.scale);
  const pts = r.x.map((x, i) => `${sx(x).toFixed(1)},${(BY() - r.v[i] * k).toFixed(1)}`).join(" ");
  const e = r.max.v_max;
  const X = sx(e.x), Y = BY() - e.value * k;
  return `<polyline points="${pts}" fill="none" stroke="${C.cyan}" stroke-width="3" stroke-linejoin="round"/>` +
    `<circle cx="${X}" cy="${Y}" r="4" fill="${C.cyan}"/>` +
    text(Math.max(8, sx(0)), BY() - 112, `Deformed shape (exaggerated ×${num(scale, 3)})`, { anchor: "start", fill: C.muted });
}

function stressGradient(r) {
  let m = 0;
  for (const s of r.sigma_bottom) m = Math.max(m, Math.abs(s));
  for (const s of r.sigma_top) m = Math.max(m, Math.abs(s));
  const L = S.cfg.L, n = 40, stops = [];
  for (let i = 0; i <= n; i++) {
    const x = (L * i) / n;
    let j = r.x.findIndex((xx) => xx >= x);
    if (j < 0) j = r.x.length - 1;
    const s = Math.abs(r.sigma_bottom[j]) >= Math.abs(r.sigma_top[j]) ? r.sigma_bottom[j] : r.sigma_top[j];
    const t = m > 0 ? s / m : 0; // −1..1  (compression blue, tension red)
    const c = t >= 0 ? mix([230, 237, 247], [239, 68, 68], t) : mix([230, 237, 247], [59, 130, 246], -t);
    stops.push(`<stop offset="${(i / n) * 100}%" stop-color="rgb(${c.join(",")})"/>`);
  }
  return `<linearGradient id="sgrad" x1="0" x2="1" y1="0" y2="0">${stops.join("")}</linearGradient>`;
}
function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }

// ------------------------------------------------------------------ main render
function renderCanvas() {
  if (!S.cfg || !S.view) return;
  const cfg = S.cfg, L = cfg.L, w = W(), h = H(), r = res();
  let g = "<defs>";
  if (r && S.mode === "stress") g += stressGradient(r);
  g += `<linearGradient id="bgrad" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#2D6CDF"/><stop offset="1" stop-color="#12355B"/></linearGradient></defs>`;

  // grid
  const x0 = wx(0), x1 = wx(w);
  const major = niceStep(90 / S.view.scale);
  if (S.snap && S.grid * S.view.scale >= 9) {
    for (let x = Math.ceil(x0 / S.grid) * S.grid; x <= x1; x += S.grid)
      g += `<line x1="${sx(x)}" y1="0" x2="${sx(x)}" y2="${h - 40}" stroke="${C.grid}" stroke-width="1"/>`;
  }
  for (let x = Math.ceil(x0 / major) * major; x <= x1; x += major)
    g += `<line x1="${sx(x)}" y1="0" x2="${sx(x)}" y2="${h - 40}" stroke="#0E2847" stroke-width="1"/>`;

  // ruler
  const ry = h - 34;
  g += `<rect x="0" y="${ry}" width="${w}" height="34" fill="#07142A"/><line x1="0" y1="${ry}" x2="${w}" y2="${ry}" stroke="${C.border}"/>`;
  const minor = major / 5;
  for (let x = Math.ceil(x0 / minor) * minor; x <= x1; x += minor) {
    const isMajor = Math.abs(x / major - Math.round(x / major)) < 1e-6;
    g += `<line x1="${sx(x)}" y1="${ry}" x2="${sx(x)}" y2="${ry + (isMajor ? 10 : 5)}" stroke="${C.muted}"/>`;
    if (isMajor) g += text(sx(x), ry + 23, fL(r6(x)), { fill: C.muted, size: 10 });
  }

  // beam
  const fill = r && S.mode === "stress" ? "url(#sgrad)" : "url(#bgrad)";
  const bsel = S.selected === "beam";
  if (r && S.mode === "defl") g += `<rect x="${sx(0)}" y="${BY() - BH / 2}" width="${L * S.view.scale}" height="${BH}" fill="none" stroke="${C.muted}" stroke-dasharray="5 4"/>`;
  else g += `<g class="item" data-id="beam"><rect x="${sx(0)}" y="${BY() - BH / 2}" width="${L * S.view.scale}" height="${BH}" rx="2" fill="${fill}" stroke="${bsel ? "#fff" : C.cyan}" stroke-width="${bsel ? 2.4 : 1.5}"/></g>`;

  // dimension line
  const dy = BY() + 112;
  g += `<line x1="${sx(0)}" y1="${BY() + 40}" x2="${sx(0)}" y2="${dy + 6}" stroke="${C.muted}" stroke-dasharray="2 3"/>` +
    `<line x1="${sx(L)}" y1="${BY() + 40}" x2="${sx(L)}" y2="${dy + 6}" stroke="${C.muted}" stroke-dasharray="2 3"/>` +
    arrow((sx(0) + sx(L)) / 2, dy, sx(0), dy, C.muted, 1, 8) + arrow((sx(0) + sx(L)) / 2, dy, sx(L), dy, C.muted, 1, 8) +
    text((sx(0) + sx(L)) / 2, dy - 6, `L = ${fL(L)}`, { fill: C.text, bold: true });

  // supports and loads
  for (const s of cfg.supports) g += drawSupport(s, S.selected === s.id);
  const wmax = Math.max(1e-12, ...cfg.loads.filter((l) => l.kind === "distributed").map((l) => Math.max(Math.abs(l.w1), Math.abs(l.w2))));
  for (const l of cfg.loads.filter((l) => l.kind === "distributed")) g += drawDistributed(l, wmax, S.selected === l.id);
  for (const l of cfg.loads) {
    if (l.kind === "point") g += drawPoint(l, S.selected === l.id);
    if (l.kind === "moment") g += drawMoment(l, S.selected === l.id);
  }

  // beam end handle (drag to change length)
  g += `<rect class="handle" data-handle="L" x="${sx(L) + 4}" y="${BY() - 9}" width="9" height="18" rx="3" fill="${C.cyan}" opacity="0.85"><title>Drag to change the beam length</title></rect>`;

  // results
  if (r) {
    g += drawReactions(r);
    if (S.mode === "sfd") g += band(r.x, r.V, C.cyan, `SFD · V (${units().force})`, fF);
    if (S.mode === "bmd") g += band(r.x, r.M, C.amber, `BMD · M (${units().moment}), sagging +`, fM);
    if (S.mode === "defl") g += deflected(r) + band(r.x, r.v.map((v) => v * 1e3), C.cyan, "Deflection v (mm), up +", (v) => `${num(v)} mm`);
    if (S.mode === "stress") {
      const sb = r.sigma_bottom.map((s) => s / 1e6), st = r.sigma_top.map((s) => s / 1e6);
      const scaleMax = Math.max(...sb.map(Math.abs), ...st.map(Math.abs));
      g += band(r.x, sb, C.cyan, "σ bottom (cyan) · σ top (amber) — MPa, tension +", (s) => `${num(s)} MPa`, { noFill: true, scaleMax });
      g += band(r.x, st, C.amber, "", (s) => `${num(s)} MPa`, { noFill: true, scaleMax });
    }
  } else if (S.mode !== "config") {
    g += text(sx(L / 2), BY() + 236, S.results ? "Results are out of date — press Analyse Structure" : "Press Analyse Structure to see results", { fill: C.muted });
  }

  if (S.error) {
    const msg = S.error.length > 150 ? S.error.slice(0, 147) + "…" : S.error;
    g += `<rect x="12" y="10" width="${w - 24}" height="34" rx="8" fill="#412E3D" stroke="${C.red}" pointer-events="none"/>` +
      text(24, 31, "⚠ " + msg, { anchor: "start", fill: C.red });
  }
  // crosshair
  if (S.hoverX != null && S.hoverX >= 0 && S.hoverX <= L)
    g += `<line x1="${sx(S.hoverX)}" y1="${BY() - 120}" x2="${sx(S.hoverX)}" y2="${ry}" stroke="${C.cyan}" stroke-opacity="0.25" stroke-dasharray="3 4" pointer-events="none"/>`;
  svg.innerHTML = g;
}

// ------------------------------------------------------------------ properties panel
function field(label, name, value, unit, extra = "") {
  return `<div class="row"><label for="f-${name}">${label}</label><div class="val">` +
    `<input type="number" id="f-${name}" data-field="${name}" value="${value}" step="any" ${extra}>` +
    (unit ? `<span class="unit">${esc(unit)}</span>` : "") + `</div></div>`;
}
function selectRow(label, name, opts, value) {
  const o = opts.map(([v, t]) => `<option value="${esc(v)}" ${v === value ? "selected" : ""}>${esc(t)}</option>`).join("");
  return `<div class="row"><label for="f-${name}">${label}</label><div class="val"><select id="f-${name}" data-field="${name}">${o}</select></div></div>`;
}
const vu = (vSI, u) => r6(vSI / UNIT[u]);

function renderProps() {
  const cfg = S.cfg, u = cfg.units, o = find(S.selected);
  let h = "";
  if (o && isSupport(o)) {
    h += `<h3>${SUP_LABEL[o.type]} support <span class="chip">${o.id}</span></h3>`;
    h += selectRow("Type", "s_type", [["fixed", "Fixed"], ["pin", "Pinned"], ["roller", "Roller"]], o.type);
    h += field("Position x", "x", vu(o.x, u.length), u.length, 'min="0"');
    const dofs = DOFS[o.type];
    h += `<div class="row"><label>Restrained</label><div class="val dofs">` +
      [["u", "u (axial)"], ["v", "v"], ["theta", "θ"]].map(([d, t]) =>
        `<label title="${t}"><input type="checkbox" data-field="dof" value="${d}" ${dofs.includes(d) ? "checked" : ""}>${d === "theta" ? "θ" : d}</label>`).join("") +
      `</div></div><p class="note">Fixed = u, v, θ · Pinned = u, v · Roller = v (resists only force normal to the beam).</p>`;
  } else if (o && o.kind === "point") {
    const k = pointKind(o);
    h += `<h3>Point load <span class="chip">${o.id}</span></h3>`;
    h += selectRow("Direction", "p_dir", Object.entries(POINT_LABEL), k);
    h += field("Magnitude", "magnitude", vu(o.magnitude, u.force), u.force, 'min="0"');
    h += field("Position x", "x", vu(o.x, u.length), u.length, 'min="0"');
    h += field("Angle", "angle_deg", r6(o.angle_deg), "° CCW +x");
    const fx = o.magnitude * Math.cos((o.angle_deg * Math.PI) / 180), fy = o.magnitude * Math.sin((o.angle_deg * Math.PI) / 180);
    h += `<p class="ro">Fx = ${fF(Math.abs(fx) < 1e-9 * o.magnitude ? 0 : fx)} · Fy = ${fF(Math.abs(fy) < 1e-9 * o.magnitude ? 0 : fy)}</p>`;
    if (k !== "down" && k !== "up") h += `<p class="note">Axial component present → solved with planar frame elements (u, v, θ per node).</p>`;
  } else if (o && o.kind === "distributed") {
    const k = distKind(o, cfg.L);
    h += `<h3>Distributed load <span class="chip">${o.id}</span></h3>`;
    h += selectRow("Type", "d_type", Object.entries(DIST_LABEL), k);
    h += field("Start x₁", "x1", vu(o.x1, u.length), u.length, 'min="0"');
    h += field("End x₂", "x2", vu(o.x2, u.length), u.length, 'min="0"');
    h += field("w at x₁", "w1", vu(o.w1, u.dist), u.dist, 'min="0"');
    h += field("w at x₂", "w2", vu(o.w2, u.dist), u.dist, 'min="0"');
    h += selectRow("Direction", "direction", [["down", "Down ↓"], ["up", "Up ↑"]], o.direction);
    const R = 0.5 * (o.w1 + o.w2) * (o.x2 - o.x1);
    const xc = o.w1 + o.w2 > 0 ? o.x1 + ((o.x2 - o.x1) * (o.w1 + 2 * o.w2)) / (3 * (o.w1 + o.w2)) : (o.x1 + o.x2) / 2;
    h += `<p class="ro">Resultant = ${fF(R)} at x = ${fL(xc)}</p>`;
  } else if (o && o.kind === "moment") {
    h += `<h3>Applied moment <span class="chip">${o.id}</span></h3>`;
    h += field("Magnitude", "Mmag", vu(Math.abs(o.M), u.moment), u.moment, 'min="0"');
    h += field("Position x", "x", vu(o.x, u.length), u.length, 'min="0"');
    h += selectRow("Sense", "sense", [["ccw", "Counter-clockwise ↺ (+)"], ["cw", "Clockwise ↻ (−)"]], o.M < 0 ? "cw" : "ccw");
  } else {
    const s = cfg.section, mats = S.materials, mk = cfg.material_key;
    h += `<h3>Beam ${S.selected === "beam" ? '<span class="chip">selected</span>' : ""}</h3>`;
    h += `<div class="row"><label for="f-name">Name</label><div class="val"><input type="text" id="f-name" data-field="name" value="${esc(cfg.name || "")}"></div></div>`;
    h += field("Length L", "L", vu(cfg.L, u.length), u.length, 'min="0"');
    h += selectRow("Section", "shape", [["rect", "Rectangle"], ["solid_circ", "Solid circle"], ["hollow_circ", "Hollow circle"]], s.shape);
    const dims = { rect: [["b", "Width b"], ["h", "Depth h"]], solid_circ: [["d", "Diameter d"]], hollow_circ: [["D", "Outer D"], ["d", "Inner d"]] }[s.shape];
    for (const [k, t] of dims) h += field(t, "sec_" + k, s[k] == null ? "" : r6(s[k] * 1e3), "mm", 'min="0"');
    const mopts = Object.entries(mats).map(([k, m]) => [k, m.name]);
    if (!mk) mopts.unshift(["", `Custom: ${cfg.material.name}`]);
    h += selectRow("Material", "material", mopts, mk || "");
    h += `<p class="ro">E = ${num(cfg.material.E / 1e9)} GPa · σy = ${num(cfg.material.yield_strength / 1e6)} MPa</p>`;
    h += `<h4>Components</h4><div class="list">`;
    for (const it of [...cfg.supports, ...cfg.loads]) {
      const pos = it.kind === "distributed" ? `${fL(it.x1)}–${fL(it.x2)}` : fL(it.x);
      h += `<button data-pick="${it.id}">${esc(itemLabel(it))}<span>${pos}</span></button>`;
    }
    if (!cfg.supports.length && !cfg.loads.length) h += `<p class="note">Empty — drag supports and loads from the toolbox.</p>`;
    h += `</div>`;
  }
  h += `<h4>Display units</h4>`;
  h += selectRow("Force", "u_force", [["N", "N"], ["kN", "kN"]], u.force);
  h += selectRow("Length", "u_length", [["m", "m"], ["mm", "mm"]], u.length);
  h += selectRow("Distributed", "u_dist", [["N/m", "N/m"], ["kN/m", "kN/m"]], u.dist);
  h += selectRow("Moment", "u_moment", [["N·m", "N·m"], ["kN·m", "kN·m"]], u.moment);
  if (o) h += `<p class="note">Drag on the canvas to move · Del deletes · Ctrl+D duplicates · Esc deselects.</p>`;
  props.innerHTML = h;
}

function onField(e) {
  const el = e.target, f = el.dataset.field;
  if (!f) return;
  const cfg = S.cfg, u = cfg.units, o = find(S.selected), L = cfg.L;
  const raw = el.value, v = parseFloat(raw);
  const needNum = !["name", "shape", "material", "s_type", "p_dir", "d_type", "direction", "sense", "dof"].includes(f) && !f.startsWith("u_");
  if (needNum && !isFinite(v)) { toast("Enter a number."); renderProps(); return; }
  if (f.startsWith("u_")) { commit(() => { cfg.units[f.slice(2)] = raw; }); return; }

  if (!o) { // beam
    if (f === "name") return commit(() => { cfg.name = raw; });
    if (f === "L") {
      const nL = v * UNIT[u.length];
      if (!(nL > 0)) { toast("Length must be positive."); renderProps(); return; }
      return commit(() => setLength(cfg, nL));
    }
    if (f === "shape") return commit(() => {
      const d = cfg.section.shape === "rect" ? cfg.section.h : cfg.section.shape === "solid_circ" ? cfg.section.d : cfg.section.D;
      cfg.section = raw === "rect" ? { shape: "rect", b: r6(d / 2), h: d, d: null, D: null }
        : raw === "solid_circ" ? { shape: "solid_circ", b: null, h: null, d, D: null }
          : { shape: "hollow_circ", b: null, h: null, d: r6(d * 0.8), D: d };
    });
    if (f.startsWith("sec_")) {
      const k = f.slice(4), val = r6(v / 1e3);
      if (!(val > 0)) { toast("Section dimensions must be positive."); renderProps(); return; }
      if (cfg.section.shape === "hollow_circ" && ((k === "d" && val >= cfg.section.D) || (k === "D" && val <= cfg.section.d))) {
        toast("Inner diameter must be smaller than the outer diameter."); renderProps(); return;
      }
      return commit(() => { cfg.section[k] = val; });
    }
    if (f === "material" && S.materials[raw]) return commit(() => { cfg.material = clone(S.materials[raw]); cfg.material_key = raw; });
    return;
  }

  if (isSupport(o)) {
    if (f === "s_type") return commit(() => { o.type = raw; });
    if (f === "dof") {
      const set = [...props.querySelectorAll('input[data-field="dof"]')].filter((c) => c.checked).map((c) => c.value).sort().join(",");
      const map = { "theta,u,v": "fixed", "u,v": "pin", v: "roller" };
      if (!map[set]) { toast("Only fixed (u,v,θ), pinned (u,v) or roller (v) restraint sets are supported."); renderProps(); return; }
      return commit(() => { o.type = map[set]; });
    }
    if (f === "x") {
      const x = r6(clamp(v * UNIT[u.length], 0, L));
      if (supportAt(x, o.id)) { toast("Another support already sits at that position."); renderProps(); return; }
      return commit(() => { o.x = x; });
    }
  }
  if (o.kind === "point") {
    if (f === "magnitude") return commit(() => { o.magnitude = Math.abs(v) * UNIT[u.force]; });
    if (f === "x") return commit(() => { o.x = r6(clamp(v * UNIT[u.length], 0, L)); });
    if (f === "angle_deg") return commit(() => { o.angle_deg = r6(((v + 180) % 360 + 360) % 360 - 180); });
    if (f === "p_dir") return commit(() => { o.angle_deg = { down: -90, up: 90, axial_r: 0, axial_l: 180, inclined: -60 }[raw]; });
  }
  if (o.kind === "distributed") {
    if (f === "x1" || f === "x2") {
      const x = r6(clamp(v * UNIT[u.length], 0, L));
      const a = f === "x1" ? x : o.x1, b = f === "x2" ? x : o.x2;
      if (!(b > a)) { toast("Start must be left of the end (x₁ < x₂)."); renderProps(); return; }
      return commit(() => { o[f] = x; });
    }
    if (f === "w1" || f === "w2") return commit(() => { o[f] = Math.abs(v) * UNIT[u.dist]; });
    if (f === "direction") return commit(() => { o.direction = raw; });
    if (f === "d_type") return commit(() => {
      const w = Math.max(o.w1, o.w2) || 1000;
      if (raw === "udl") { o.x1 = 0; o.x2 = L; o.w1 = o.w2 = w; }
      if (raw === "partial") {
        o.w1 = o.w2 = w;
        if (o.x1 <= 1e-9 && Math.abs(o.x2 - L) <= 1e-9) { o.x1 = snapX(L / 4); o.x2 = snapX((3 * L) / 4); }
      }
      if (raw === "linear") { o.w1 = w / 2; o.w2 = w; }
      if (raw === "triangular") { o.w1 = 0; o.w2 = w; }
    });
  }
  if (o.kind === "moment") {
    if (f === "Mmag") return commit(() => { o.M = (o.M < 0 ? -1 : 1) * Math.abs(v) * UNIT[u.moment]; });
    if (f === "x") return commit(() => { o.x = r6(clamp(v * UNIT[u.length], 0, L)); });
    if (f === "sense") return commit(() => { o.M = (raw === "cw" ? -1 : 1) * Math.abs(o.M); });
  }
}

// ------------------------------------------------------------------ status, toolbar state
function renderStatus(extra) {
  const r = res();
  let info = "";
  if (extra) info = extra;
  else if (S.error) info = "Analysis rejected — see message";
  else if (r) info = `${r.method} · δmax ${num(r.max.v_max.value * 1e3)} mm · |M|max ${fM(Math.abs(r.max.M_max.value))} · |σ|max ${num(Math.abs(r.max.sigma_max.value) / 1e6)} MPa · ${r.stability}`;
  else if (S.results) info = "Model changed since the last analysis";
  else info = `${S.cfg.supports.length} support(s) · ${S.cfg.loads.length} load(s)`;
  $("info").textContent = info;
  const sy = $("sync");
  sy.textContent = S.synced ? (S.live ? "● live · synced" : "● synced") : "● editing…";
  sy.className = S.synced ? "ok" : "dirty";
}
function renderToolbar() {
  const q = (c) => document.querySelector(`[data-cmd="${c}"]`);
  q("undo").disabled = !S.undo.length;
  q("redo").disabled = !S.redo.length;
  q("delete").disabled = !S.selected || S.selected === "beam";
  q("duplicate").disabled = !find(S.selected);
  const has = !!res();
  document.querySelectorAll("#modes button").forEach((b) => {
    b.classList.toggle("on", b.dataset.mode === S.mode);
    b.disabled = b.dataset.mode !== "config" && !has;
  });
}
function renderAll() {
  if (!S.cfg) return;
  renderCanvas();
  renderProps();
  renderStatus();
  renderToolbar();
}

// ------------------------------------------------------------------ toolbar commands
const CMDS = {
  undo, redo,
  delete: deleteSelected,
  duplicate: duplicateSelected,
  clear() { commit(() => { S.cfg.supports = []; S.cfg.loads = []; }); S.selected = null; renderAll(); },
  reset() {
    const p = S.presets.cantilever_end;
    if (!p) return;
    commit(() => { S.cfg = ensureIds(clone(p.config)); });
    S.selected = null; S.mode = "config"; S.snap = true; S.grid = 0.05;
    $("snapChk").checked = true; $("gridSel").value = "0.05";
    fit(); renderAll();
  },
  presets(e) {
    const m = $("presetMenu");
    if (!m.hidden) { m.hidden = true; return; }
    m.innerHTML = Object.entries(S.presets).map(([k, p]) => `<button data-preset="${k}">${esc(p.label)}</button>`).join("");
    const b = e.currentTarget.getBoundingClientRect();
    m.style.left = `${b.left}px`; m.style.top = `${b.bottom + 4}px`;
    m.hidden = false;
  },
  zoomIn() { zoomAt(1.25, W() / 2); },
  zoomOut() { zoomAt(0.8, W() / 2); },
  fit() { fit(); renderCanvas(); },
  analyse() { S.error = null; sendValue("analyse"); },
  save() {
    const blob = new Blob([JSON.stringify(modelOnly(S.cfg), null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${(S.cfg.name || "beam").replace(/[^\w-]+/g, "_")}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  },
  load() { $("fileIn").click(); },
};
function zoomAt(f, cx) {
  const x = wx(cx);
  S.view.scale = clamp(S.view.scale * f, 5, 1e6);
  S.userView = true;
  S.view.ox = cx - x * S.view.scale;
  renderCanvas();
}

document.querySelectorAll("[data-cmd]").forEach((b) => b.addEventListener("click", (e) => { if (S.cfg) CMDS[b.dataset.cmd](e); }));
$("presetMenu").addEventListener("click", (e) => {
  const k = e.target.closest("[data-preset]")?.dataset.preset;
  if (!k) return;
  $("presetMenu").hidden = true;
  commit(() => { S.cfg = ensureIds(clone(S.presets[k].config)); S.cfg.units = Object.assign({}, S.cfg.units); });
  S.selected = null; S.mode = "config";
  fit(); renderAll();
  toast(`Loaded preset: ${S.presets[k].label}`);
});
document.addEventListener("pointerdown", (e) => {
  if (!e.target.closest(".menu-wrap")) $("presetMenu").hidden = true;
});
$("snapChk").addEventListener("change", (e) => { S.snap = e.target.checked; renderCanvas(); });
$("gridSel").addEventListener("change", (e) => { S.grid = parseFloat(e.target.value); renderCanvas(); });
$("liveChk").addEventListener("change", (e) => {
  S.live = e.target.checked;
  if (S.live) sendValue("live");
  renderStatus();
});
$("fileIn").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  f.text().then((t) => {
    let d;
    try { d = JSON.parse(t); } catch { toast("Not a valid JSON file."); return; }
    if (!d || typeof d.L !== "number" || !d.section || !d.material || !Array.isArray(d.supports) || !Array.isArray(d.loads)) {
      toast("JSON must be a SmartBeam configuration (L, section, material, supports, loads)."); return;
    }
    const cfg = { version: 1, name: d.name || f.name.replace(/\.json$/i, ""), L: d.L, section: d.section, material: d.material,
      supports: d.supports, loads: d.loads, units: (d.playground && d.playground.units) || S.cfg.units,
      material_key: Object.keys(S.materials).find((k) => JSON.stringify(S.materials[k]) === JSON.stringify(d.material)) || null };
    commit(() => { S.cfg = ensureIds(cfg); });
    S.selected = null; fit(); renderAll();
    toast(`Loaded ${f.name}`);
  });
});
document.querySelectorAll("#modes button").forEach((b) => b.addEventListener("click", () => { S.mode = b.dataset.mode; renderCanvas(); renderToolbar(); }));
props.addEventListener("change", onField);
props.addEventListener("click", (e) => {
  const id = e.target.closest("[data-pick]")?.dataset.pick;
  if (id) { S.selected = id; renderAll(); }
});

// ------------------------------------------------------------------ toolbox: HTML5 drag & drop + click-to-add
document.querySelectorAll(".tool").forEach((t) => {
  t.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/plain", t.dataset.tool); e.dataTransfer.effectAllowed = "copy"; });
  t.addEventListener("click", () => {
    const tool = t.dataset.tool, L = S.cfg.L;
    let x = L / 2;
    if (["fixed", "pin", "roller"].includes(tool)) x = !supportAt(0) ? 0 : !supportAt(L) ? L : L / 2;
    if (tool === "point_down" || tool === "point_up") x = L;
    addTool(tool, x);
  });
});
stage.addEventListener("dragover", (e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; stage.classList.add("drop"); S.hoverX = snapX(clamp(wx(svgPoint(e).x), 0, S.cfg.L)); renderCanvas(); renderCoord(); });
stage.addEventListener("dragleave", () => stage.classList.remove("drop"));
stage.addEventListener("drop", (e) => {
  e.preventDefault();
  stage.classList.remove("drop");
  const tool = e.dataTransfer.getData("text/plain");
  if (!tool || !S.cfg) return;
  const x = wx(svgPoint(e).x);
  if (tool !== "beam" && (x < -0.15 * S.cfg.L || x > 1.15 * S.cfg.L)) { toast("Drop the component onto the beam."); return; }
  addTool(tool, x);
});

// ------------------------------------------------------------------ canvas pointer interaction
function renderCoord() {
  const x = S.hoverX;
  $("coord").textContent = x == null ? "x = —" : `x = ${fL(x)}${S.snap ? " (snap " + num(S.grid) + " m)" : ""}`;
}
svg.addEventListener("pointerdown", (e) => {
  if (!S.cfg || e.button !== 0) return;
  const p = svgPoint(e);
  const hEl = e.target.closest("[data-handle]"), iEl = e.target.closest("[data-id]");
  svg.setPointerCapture(e.pointerId);
  if (hEl) {
    S.drag = { type: "handle", handle: hEl.dataset.handle, id: hEl.dataset.id, before: snapshot(), moved: false, p0: p };
  } else if (iEl && iEl.dataset.id !== "beam") {
    const o = find(iEl.dataset.id);
    S.selected = o.id;
    const ref = o.kind === "distributed" ? o.x1 : o.x;
    S.drag = { type: "item", id: o.id, off: wx(p.x) - ref, before: snapshot(), moved: false, p0: p };
    renderAll();
  } else {
    S.drag = { type: "pan", p0: p, ox: S.view.ox, oy: S.view.oy, moved: false, beam: !!iEl };
    svg.classList.add("panning");
  }
});
svg.addEventListener("pointermove", (e) => {
  if (!S.cfg) return;
  const p = svgPoint(e), d = S.drag;
  S.hoverX = snapX(wx(p.x));
  renderCoord();
  if (!d) { renderCanvas(); return; }
  if (!d.moved && Math.hypot(p.x - d.p0.x, p.y - d.p0.y) < 3) return;
  d.moved = true;
  const L = S.cfg.L;
  if (d.type === "pan") {
    S.view.ox = d.ox + (p.x - d.p0.x);
    S.view.oy = d.oy + (p.y - d.p0.y);
    S.userView = true;
  } else if (d.type === "item") {
    const o = find(d.id), x = snapX(wx(p.x) - d.off);
    if (o.kind === "distributed") {
      const len = o.x2 - o.x1, a = r6(clamp(x, 0, L - len));
      o.x1 = a; o.x2 = r6(a + len);
    } else {
      const nx = r6(clamp(x, 0, L));
      if (!(isSupport(o) && supportAt(nx, o.id))) o.x = nx;
    }
  } else if (d.type === "handle") {
    if (d.handle === "L") {
      S.cfg = JSON.parse(d.before);
      const minL = Math.max(S.snap ? S.grid : 0.01, 0.01);
      setLength(S.cfg, Math.max(minL, snapX(wx(p.x))));
    } else {
      const o = find(d.id), x = r6(clamp(snapX(wx(p.x)), 0, L)), minW = S.snap ? S.grid : L / 200;
      if (d.handle === "x1") o.x1 = r6(Math.min(x, o.x2 - minW));
      else o.x2 = r6(Math.max(x, o.x1 + minW));
      o.x1 = Math.max(0, o.x1); o.x2 = Math.min(L, o.x2);
    }
  }
  renderCanvas();
  if (d.type !== "pan") { renderProps(); }
});
function endDrag() {
  const d = S.drag;
  S.drag = null;
  svg.classList.remove("panning");
  if (!d) return;
  if (d.type === "pan") {
    if (!d.moved) { S.selected = d.beam ? "beam" : null; renderAll(); }
    return;
  }
  if (d.moved && snapshot() !== d.before) { pushUndo(d.before); changed(); }
  else renderAll();
}
svg.addEventListener("pointerup", endDrag);
svg.addEventListener("pointercancel", endDrag);
svg.addEventListener("pointerleave", () => { if (!S.drag) { S.hoverX = null; renderCoord(); renderCanvas(); } });
svg.addEventListener("wheel", (e) => {
  if (!S.cfg) return;
  // Plain vertical wheel scrolls the Streamlit page; Ctrl/Cmd + wheel (or trackpad pinch) zooms, Shift pans.
  const zoom = e.ctrlKey || e.metaKey, pan = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
  if (!zoom && !pan) return;
  e.preventDefault();
  if (zoom) zoomAt(Math.exp(-e.deltaY * 0.0015), svgPoint(e).x);
  else { S.view.ox -= e.deltaX || e.deltaY; S.userView = true; renderCanvas(); }
}, { passive: false });

document.addEventListener("keydown", (e) => {
  if (!S.cfg || (e.target.closest && e.target.closest("input, select, textarea"))) return;
  const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
  if (mod && k === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
  else if (mod && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); redo(); }
  else if (mod && k === "d") { e.preventDefault(); duplicateSelected(); }
  else if (k === "delete" || k === "backspace") { e.preventDefault(); deleteSelected(); }
  else if (k === "escape") { S.selected = null; renderAll(); }
  else if ((k === "arrowleft" || k === "arrowright") && find(S.selected)) {
    e.preventDefault();
    const o = find(S.selected), dx = (k === "arrowleft" ? -1 : 1) * (S.snap ? S.grid : S.cfg.L / 100), L = S.cfg.L;
    commit(() => {
      if (o.kind === "distributed") {
        const len = o.x2 - o.x1, a = r6(clamp(o.x1 + dx, 0, L - len));
        o.x1 = a; o.x2 = r6(a + len);
      } else {
        const nx = r6(clamp(o.x + dx, 0, L));
        if (!(isSupport(o) && supportAt(nx, o.id))) o.x = nx;
      }
    });
  }
});

new ResizeObserver(() => { if (S.cfg) { if (!S.view || !S.userView) fit(); renderCanvas(); } }).observe(stage);

// ------------------------------------------------------------------ Streamlit render
window.addEventListener("message", (e) => {
  const m = e.data;
  if (!m || m.type !== "streamlit:render") return;
  const a = m.args || {};
  S.presets = a.presets || {};
  S.materials = a.materials || {};
  if (a.config && a.config_rev !== S.rev) {
    S.cfg = ensureIds(clone(a.config));
    S.rev = a.config_rev;
    S.version = a.seen_version || 0;
    S.undo = []; S.redo = []; S.selected = null; S.synced = true;
  }
  if (!S.view || !S.userView) fit();
  // results only count when Python has seen the canvas' latest edit
  const fresh = a.seen_version === S.version;
  S.results = a.results || null;
  S.resultsCurrent = !!(a.results && a.results_current && fresh);
  S.error = fresh ? a.error || null : null;
  if (S.resultsCurrent && S.mode === "config" && S.justAnalysed) S.mode = "bmd";
  S.justAnalysed = false;
  renderAll();
  post("streamlit:setFrameHeight", { height: FRAME_H });
});
// the first analysis switches the overlay to the BMD so the result is visible immediately
document.querySelector('[data-cmd="analyse"]').addEventListener("click", () => { S.justAnalysed = true; });
post("streamlit:componentReady", { apiVersion: 1 });
post("streamlit:setFrameHeight", { height: FRAME_H });
