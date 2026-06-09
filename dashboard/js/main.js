// Low-severity fire dashboard.
//
// Two views, each with its own fire list:
//   scrubber — Rx treatments accumulating inside a recognizable megafire.
//              Fires: Creek, August Complex, Dixie, Camp, Caldor, North Complex.
//   change   — total simulated severity reduction for the year of the fire.
//              Fires: the six with largest absolute mean ΔNBR drop (Dixie,
//              Caldor, KNP Complex, Sugar, Windy, Dolan).
//
// Slider range is per fire: from 2009 (empty) to fire.year.
// Changing fire resets the slider, stops any animation, re-zooms the map.
// Histogram in the change view is a KDE density curve drawn as a small
// overlay inside the map (no binning).

const DATA = "data/";

// Severity classes — the paper's 5-class boundaries collapsed to 3 to
// exaggerate cross-boundary movement in the change-view swipe. Boundaries
// at 270 and 440 (the paper's severe / very-severe cutoffs); the bottom
// class buckets everything else as "unburned" for visual contrast.
// Gray for the bottom, muted red for severe, saturated red for very severe.
const SEV_CLASSES = [
  { key: "unburned",     name: "Unburned",     min: -Infinity, max: 270,      color: "#4a4a52" },
  { key: "severe",       name: "Severe",       min: 270,        max: 440,      color: "#8a3d33" },
  { key: "very_severe",  name: "Very severe",  min: 440,        max: Infinity, color: "#ff1a05" },
];
function classOf(v) {
  for (let i = SEV_CLASSES.length - 1; i >= 0; i--) {
    if (v >= SEV_CLASSES[i].min) return SEV_CLASSES[i];
  }
  return SEV_CLASSES[0];
}

const state = {
  view: "scrubber",
  fireSlug: null,
  year: 2009,
  fires: [],
  fireData: {},
  caOutline: null,
  stateMock: null,
  map: null,
  playing: false,
  playTimer: null,
  smoke: null,
  mapL: null,           // change-view observed map  (lazy)
  mapR: null,           // change-view counterfactual map (lazy)
  compare: null,        // maplibregl.Compare instance (lazy)
  compareLoaded: false, // have we wired sources into mapL/mapR?
  compareFireSlug: null,// which fire is currently loaded in the compare maps
};

// ---------- data ----------
async function loadIndex() {
  state.fires     = await d3.json(DATA + "fires_index.json");
  state.caOutline = await d3.json(DATA + "ca_outline.json");
  state.stateMock = await d3.json(DATA + "state_mock_treatments.json");
  state.smoke     = await d3.json(DATA + "smoke_coef.json");
}
async function ensureFire(slug) {
  if (state.fireData[slug]) return state.fireData[slug];
  state.fireData[slug] = await d3.json(DATA + "fire_" + slug + ".json");
  return state.fireData[slug];
}

function firesForView(view) {
  return state.fires.filter(f => f.set === view || f.set === "both");
}

// ---------- sidebar ----------
function setupSidebar() {
  document.querySelectorAll("#view-list li").forEach(li => {
    li.classList.toggle("active", li.dataset.view === state.view);
    li.addEventListener("click", () => {
      if (state.view === li.dataset.view) return;
      document.querySelectorAll("#view-list li").forEach(x => x.classList.remove("active"));
      li.classList.add("active");
      state.view = li.dataset.view;
      onViewChange();
    });
  });

  const sel = document.getElementById("fire-select");
  sel.addEventListener("change", async () => {
    state.fireSlug = sel.value;
    resetForFireChange();
    if (state.view === "scrubber") {
      await loadFireIntoMap();
    } else {
      await loadFireIntoCompare();
    }
    refresh();
  });

  const sl = document.getElementById("year-slider");
  sl.addEventListener("input", () => {
    state.year = +sl.value;
    document.getElementById("year-readout").textContent = state.year;
    refresh();
  });
  document.getElementById("play-btn").addEventListener("click", togglePlay);

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (state.view === "scrubber") {
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "ArrowRight") setYear(+sl.value + 1);
      if (e.key === "ArrowLeft")  setYear(+sl.value - 1);
    }
  });
}

function populateFireDropdown() {
  const sel = document.getElementById("fire-select");
  const opts = firesForView(state.view);
  sel.innerHTML = opts.map(f =>
    `<option value="${f.slug}">${f.name} (${f.year}) — ${(f.acres/1000).toFixed(0)}k acres</option>`
  ).join("");
  // pick first one for this view if current selection isn't in the list
  if (!opts.some(f => f.slug === state.fireSlug)) state.fireSlug = opts[0].slug;
  sel.value = state.fireSlug;
}

function fireMeta() { return state.fires.find(f => f.slug === state.fireSlug); }

function setupSliderForFire() {
  const f = fireMeta();
  const sl = document.getElementById("year-slider");
  sl.min = f.min_year;   // 2009
  sl.max = f.max_year;   // fire.year
  sl.step = 1;
  sl.value = f.min_year; // start empty
  state.year = f.min_year;
  document.getElementById("year-readout").textContent = state.year;
  const tk = document.querySelectorAll(".tick-labels span");
  if (tk.length === 4) {
    const yrs = [f.min_year, Math.round((f.min_year*2 + f.max_year)/3), Math.round((f.min_year + f.max_year*2)/3), f.max_year];
    tk.forEach((t, i) => t.textContent = String(yrs[i]));
  }
}

function setYear(v) {
  const sl = document.getElementById("year-slider");
  v = Math.max(+sl.min, Math.min(+sl.max, v));
  sl.value = v; state.year = v;
  document.getElementById("year-readout").textContent = state.year;
  refresh();
}
function togglePlay() {
  if (state.playing) return stopPlay();
  state.playing = true;
  document.getElementById("play-btn").classList.add("playing");
  document.getElementById("play-btn").textContent = "❚❚ Pause";
  const sl = document.getElementById("year-slider");
  let v = +sl.min;
  setYear(v);
  state.playTimer = setInterval(() => {
    v++;
    if (v > +sl.max) return stopPlay();
    setYear(v);
  }, 700);
}
function stopPlay() {
  state.playing = false;
  if (state.playTimer) clearInterval(state.playTimer);
  document.getElementById("play-btn").classList.remove("playing");
  document.getElementById("play-btn").textContent = "▶ Play animation";
}
function resetForFireChange() {
  stopPlay();
  setupSliderForFire();
}

// ---------- view switching ----------
async function onViewChange() {
  populateFireDropdown();
  setupSliderForFire();
  stopPlay();

  const inset       = document.getElementById("inset");
  const scrubCtrls  = document.getElementById("scrubber-controls");
  const changeCtrls = document.getElementById("change-controls");
  const mapEl       = document.getElementById("map");
  const cmpEl       = document.getElementById("compare");

  const help = document.querySelector(".help");

  if (state.view === "scrubber") {
    document.getElementById("kicker").textContent = "Fire treatments";
    document.getElementById("title").innerHTML = "A decade of treatments inside one fire";
    document.getElementById("dek").innerHTML =
      'Each square is a 1&nbsp;km² pixel that the simulated Rx policy treated at some point. Drag the slider to watch coverage grow.';
    inset.style.display = "";
    scrubCtrls.style.display = "";
    changeCtrls.style.display = "none";
    mapEl.style.display = "";
    cmpEl.style.display = "none";
    if (help) help.style.display = "";
    setLegendForScrubber();
    await loadFireIntoMap();
    refresh();
    state.map.resize();
  } else {
    document.getElementById("kicker").textContent = "Severity change";
    document.getElementById("title").innerHTML = "What the same fire would have looked like";
    document.getElementById("dek").innerHTML =
      'Each pixel is shaded by its <em class="accent">severity class</em>. Drag the divider on the map to reveal the counterfactual where prescribed fire had been running through the fire&apos;s year.';
    inset.style.display = "none";
    scrubCtrls.style.display = "none";
    changeCtrls.style.display = "";
    mapEl.style.display = "none";
    cmpEl.style.display = "";
    if (help) help.style.display = "none";
    setLegendForChange();
    await initCompareMaps();
    await loadFireIntoCompare();
    refresh();
    state.mapL.resize(); state.mapR.resize();
  }
}

// ---- Compare-view map init ----
async function initCompareMaps() {
  if (state.mapL && state.mapR) return;
  const style = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
  state.mapL = new maplibregl.Map({ container: "map-left",  style, center: [-119.5,37.5], zoom: 6, attributionControl: false });
  state.mapR = new maplibregl.Map({ container: "map-right", style, center: [-119.5,37.5], zoom: 6, attributionControl: false });
  state.mapR.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
  await Promise.all([
    new Promise(res => state.mapL.on("load", res)),
    new Promise(res => state.mapR.on("load", res)),
  ]);
  state.compare = new maplibregl.Compare(state.mapL, state.mapR, "#compare", {
    orientation: "vertical",
  });
}

async function loadFireIntoCompare() {
  const fd = await ensureFire(state.fireSlug);
  const f  = fireMeta();

  for (const [m, mode] of [[state.mapL, "obs"], [state.mapR, "cf"]]) {
    if (!m.getSource("pixels")) m.addSource("pixels", { type: "geojson", data: fd });
    else                          m.getSource("pixels").setData(fd);

    const perimFC = { type: "FeatureCollection",
      features: [{ type: "Feature", geometry: fd.perimeter, properties: {} }] };
    if (!m.getSource("perimeter")) m.addSource("perimeter", { type: "geojson", data: perimFC });
    else                            m.getSource("perimeter").setData(perimFC);

    const valueExpr = mode === "obs"
      ? ["get", "obs"]
      : ["get", String(fd.year), ["get", "sim"]];
    const colorExpr = ["step", valueExpr,
      SEV_CLASSES[0].color,
      ...SEV_CLASSES.slice(1).flatMap(c => [c.min, c.color]),
    ];

    if (!m.getLayer("pixel-fill")) {
      m.addLayer({ id: "pixel-fill", type: "fill", source: "pixels",
        paint: { "fill-color": colorExpr, "fill-opacity": 0.88, "fill-antialias": true } });
    } else {
      m.setPaintProperty("pixel-fill", "fill-color", colorExpr);
      m.setPaintProperty("pixel-fill", "fill-opacity", 0.88);
    }
    if (!m.getLayer("perimeter-line")) {
      m.addLayer({ id: "perimeter-line", type: "line", source: "perimeter",
        paint: { "line-color": "#f8f8f2", "line-width": 1.3, "line-opacity": 0.55 } });
    }
  }
  state.compareFireSlug = state.fireSlug;
  // Both maps stay in lock-step; just fly the left one and Compare sync handles the rest.
  state.mapL.fitBounds([[f.bbox[0],f.bbox[1]],[f.bbox[2],f.bbox[3]]], { padding: 70, duration: 600 });
}

function setLegendForScrubber() {
  document.getElementById("legend-section").innerHTML = `
    <div class="label">Treatments by year</div>
    <div class="legend-bar treat"></div>
    <div class="legend-ticks"><span>—</span><span>—</span><span>—</span></div>
    <div class="legend-cap"><span>untreated</span><span>treated</span></div>`;
  const f = fireMeta();
  const tk = document.querySelectorAll("#legend-section .legend-ticks span");
  if (f && tk.length === 3) {
    tk[0].textContent = f.min_year + 1;
    tk[1].textContent = Math.round((f.min_year + 1 + f.max_year) / 2);
    tk[2].textContent = f.max_year;
  }
}
function setLegendForChange() {
  const cols = `repeat(${SEV_CLASSES.length}, 1fr)`;
  const swatches = SEV_CLASSES.map(c =>
    `<div style="background:${c.color}"></div>`).join("");
  const names = SEV_CLASSES.map(c => `<span>${c.name}</span>`).join("");
  document.getElementById("legend-section").innerHTML = `
    <div class="label">Severity class</div>
    <div class="class-legend">
      <div class="bar" style="grid-template-columns:${cols}">${swatches}</div>
      <div class="names" style="grid-template-columns:${cols}">${names}</div>
      <div class="cap"><span>unburned</span><span>more severe →</span></div>
    </div>`;
}

// ---------- map ----------
async function initMap() {
  state.map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: [-119.5, 37.5],
    zoom: 6,
    attributionControl: false,
  });
  state.map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
  await new Promise(res => state.map.on("load", res));
}

async function loadFireIntoMap() {
  const fd = await ensureFire(state.fireSlug);
  const m  = state.map;

  if (!m.getSource("pixels")) m.addSource("pixels", { type: "geojson", data: fd });
  else                          m.getSource("pixels").setData(fd);

  const perimFC = { type: "FeatureCollection",
    features: [{ type: "Feature", geometry: fd.perimeter, properties: {} }] };
  if (!m.getSource("perimeter")) m.addSource("perimeter", { type: "geojson", data: perimFC });
  else                            m.getSource("perimeter").setData(perimFC);

  if (!m.getLayer("pixel-fill")) {
    m.addLayer({
      id: "pixel-fill", type: "fill", source: "pixels",
      paint: { "fill-color": "#ffb86c", "fill-opacity": 0, "fill-antialias": true },
    });
  }
  if (!m.getLayer("perimeter-line")) {
    m.addLayer({
      id: "perimeter-line", type: "line", source: "perimeter",
      paint: { "line-color": "#f8f8f2", "line-width": 1.4, "line-opacity": 0.6 },
    });
  }

  const f = fireMeta();
  m.fitBounds([[f.bbox[0], f.bbox[1]], [f.bbox[2], f.bbox[3]]],
              { padding: 70, duration: 700 });
}

// ---------- refresh ----------
async function refresh() {
  const fd = await ensureFire(state.fireSlug);
  if (state.view === "scrubber") refreshScrubber(fd);
  else                            refreshChange(fd);
  if (state.view === "scrubber") drawStateInset(fd);
  refreshStats(fd);
}

function refreshScrubber(fd) {
  const m = state.map;
  m.setPaintProperty("pixel-fill", "fill-color", "#ffb86c");
  m.setPaintProperty("pixel-fill", "fill-opacity",
    ["case",
      ["all", [">", ["get", "ft"], 2000], ["<=", ["get", "ft"], state.year]],
      0.6, 0
    ]);
}

function refreshChange(fd) {
  // Paint expressions are set in loadFireIntoCompare(); nothing to refresh
  // per-tick because the change view has no slider any more.
}

function refreshStats(fd) {
  const block      = document.getElementById("stat-block");
  const smokeBlock = document.getElementById("smoke-block");

  // ---- Top block (view-specific) ----
  let rows;
  if (state.view === "scrubber") {
    const treated = fd.features.filter(f => f.properties.ft > 2000 && f.properties.ft <= state.year).length;
    const total   = fd.features.length;
    rows = [
      ["Fire", `${fd.name} ${fd.year}`],
      ["Pixels treated", d3.format(",")(treated) + " / " + d3.format(",")(total)],
      ["Coverage", (100*treated/total).toFixed(1) + "%"],
    ];
  } else {
    // Counterfactual evaluated at the fire's year (Rx programme through fire).
    // Bins follow SEV_CLASSES so the stats track the legend automatically.
    const Y = fd.year;
    const tally = (vals) => {
      const c = SEV_CLASSES.map(() => 0);
      for (const v of vals) {
        for (let i = SEV_CLASSES.length - 1; i >= 0; i--) {
          if (v >= SEV_CLASSES[i].min) { c[i]++; break; }
        }
      }
      return c;
    };
    const n   = fd.features.length;
    const o   = tally(fd.features.map(f => f.properties.obs));
    const s   = tally(fd.features.map(f => f.properties.sim[Y]));
    const pair = (a, b) =>
      `${a.toFixed(1)}% <span class="arrow">→</span> <span class="averted">${b.toFixed(1)}%</span>`;
    rows = [
      ["Fire", `${fd.name} ${fd.year}`],
      ...SEV_CLASSES.map((cls, i) => [cls.name, pair(100*o[i]/n, 100*s[i]/n)]),
    ];
  }
  block.innerHTML = rows.map(r =>
    `<div class="stat-row"><span class="stat-label">${r[0]}</span><span class="stat-value">${r[1]}</span></div>`).join("");

  // ---- Smoke block: single % row ----
  const Y      = state.view === "change" ? fd.year : (state.year < 2010 ? null : state.year);
  const sumObs = d3.sum(fd.features, f => f.properties.obs);
  const sumSim = Y == null ? sumObs : d3.sum(fd.features, f => f.properties.sim[Y]);
  const pctAvert = sumObs > 0 ? 100 * (1 - sumSim / sumObs) : 0;
  smokeBlock.innerHTML = `
    <div class="stat-row"><span class="stat-label">Smoke PM₂.₅ averted</span>
      <span class="stat-value averted">${pctAvert.toFixed(1)}%</span></div>`;
}

// ---------- inset ----------
function drawStateInset(fd) {
  const cv = document.getElementById("state-inset");
  const rect = cv.getBoundingClientRect();
  if (rect.width < 10) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = rect.width * dpr; cv.height = rect.height * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  const proj = d3.geoAlbers().parallels([34, 40.5]).rotate([120, 0])
    .fitExtent([[10, 10], [W-10, H-10]], state.caOutline);
  const path = d3.geoPath(proj, ctx);

  ctx.beginPath(); path(state.caOutline);
  ctx.fillStyle = "#21222c"; ctx.fill();
  ctx.strokeStyle = "#44475a"; ctx.lineWidth = 0.9; ctx.stroke();

  const pts = state.stateMock.filter(p => p.y <= state.year);
  for (const p of pts) {
    const xy = proj([p.lon, p.lat]);
    if (!xy) continue;
    ctx.fillStyle = "rgba(255,184,108,0.35)";
    ctx.fillRect(xy[0] - 0.7, xy[1] - 0.7, 1.4, 1.4);
  }

  ctx.beginPath(); path(fd.perimeter);
  ctx.fillStyle = "rgba(255,121,198,0.85)";
  ctx.fill();
  ctx.strokeStyle = "#f1fa8c";
  ctx.lineWidth = 1.3;
  ctx.stroke();

  document.getElementById("inset-foot").textContent =
    `${fd.name} · ${fd.year} · ${(fd.acres/1000).toFixed(0)}k acres · ${d3.format(",")(pts.length)} treatments placed statewide by ${state.year}`;
}

// (histogram removed — severity-class map carries the comparison now)

// ---------- Boot ----------
(async function init() {
  await loadIndex();
  state.fireSlug = firesForView("scrubber")[0].slug;
  setupSidebar();
  await initMap();
  await onViewChange();

  let rs;
  window.addEventListener("resize", () => {
    clearTimeout(rs);
    rs = setTimeout(() => {
      if (state.map)  state.map.resize();
      if (state.mapL) state.mapL.resize();
      if (state.mapR) state.mapR.resize();
      refresh();
    }, 200);
  });
})();
