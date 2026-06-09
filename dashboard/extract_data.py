"""Extract per-fire visualization data.

Two fire lists, since the two dashboard views want different fires:

- SCRUBBER_FIRES: recognizable megafires for the treatment-buildup view.
- CHANGE_FIRES:   fires with the largest simulated severity reduction
  (most negative mean ΔNBR drop in the year-of-fire counterfactual).

For every fire we write a GeoJSON FeatureCollection of 1 km² pixel polygons
**clipped to the fire perimeter**.  Each feature carries:

  obs        observed ΔNBR severity
  ft         representative simulation year_treat for that grid (random
             draw matching the notebook's `groupby.year_treat.first()`)
  sim        {<year>: mean simulated ΔNBR if the random Rx policy had
                      run through <year>}     year ∈ [2010 .. fire_year]
  delta      {<year>: sim[<year>] - obs}      (negative = severity averted)

The intensity-year axis is now *dense* (every year, not every-other) so the
slider scrubs smoothly.

Outputs:
  fire_<slug>.json
  fires_index.json                (with `set` ∈ {"scrubber","change","both"})
  ca_outline.json
  state_mock_treatments.json      (used by the inset)
"""
import json
import os

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box, mapping
from shapely.ops import transform

SIM_PATH = "/mnt/sherlock/oak/prescribed_data/results/simulations/policy_no_spill_large_new_4000"
DNBR_PARQUET = "/home/topcat/projects/extract/data/dnbr.parquet"
MTBS_SHP = "/mnt/sherlock/oak/prescribed_data/geoms/mtbs_perims_DD_updated/mtbs_perims_DD.shp"
CA_SHP_DIR = "/mnt/sherlock/oak/prescribed_data/geoms/california_geom"
OUT = "/home/topcat/projects/extract/dashboard/data"
os.makedirs(OUT, exist_ok=True)

SCRUBBER_FIRES = [
    ("CA3720111927220200905", "Creek", 2020),
    ("CA3966012280920200817", "August Complex", 2020),
    ("CA3987612137920210714", "Dixie", 2021),
    ("CA3982012144020181108", "Camp", 2018),
    ("CA3858612053820210815", "Caldor", 2021),
    ("CA4009112093120200817", "North Complex", 2020),
]
CHANGE_FIRES = [
    # Ranked by mean per-pixel ΔNBR reduction under the simulated Rx programme,
    # with famous/large fires preferred at the margins.  See analysis run
    # 2026-05-14 (mean & max per-pixel ranking).
    ("CA4156412340420210801", "McCash", 2021),              # mean -48, max -56
    ("CA3858612053820210815", "Caldor", 2021),              # mean -46, max -60 — Tahoe
    ("CA4114212301620210731", "Haypress", 2021),            # mean -44, max -58
    ("CA4118512343320200728", "Red Salmon Complex", 2020),  # mean -43, max -54
    ("CA4005312066920190904", "Walker", 2019),              # mean -43, max -50 — 2019
    ("CA4009112093120200817", "North Complex", 2020),       # mean -42, max -56
    ("CA4075212333720210731", "Monument", 2021),            # mean -41, max -57
    ("CA4185812335420200908", "Slater", 2020),              # mean -41, max -55
    ("CA3726212222320200816", "CZU Aug Lightning", 2020),   # mean -40, max -53 — Santa Cruz
    ("CA3987612137920210714", "Dixie", 2021),               # mean -37, max -59 — largest ever
    ("CA3983912034520210702", "Sugar", 2021),               # mean -33, max -55
    ("CA3612312160220200819", "Dolan", 2020),               # mean -31, max -53 — Big Sur
    ("CA3658211879520210912", "KNP Complex", 2021),         # mean -27, max -55 — Sequoia
]

TO_LL_TR = pyproj.Transformer.from_crs("EPSG:3310", "EPSG:4326", always_xy=True)
TO_LL = TO_LL_TR.transform
rng = np.random.default_rng(42)
# Half a kilometre in the projected (Albers) coordinate system, plus a small
# overlap pad so adjacent pixels share an edge in the rendered output instead
# of leaving a hairline gap from float rounding / reprojection seams.
HALF_KM_PROJ = 505.0  # 5 m overlap on each side


def round_coords(geom, nd=4):
    return transform(lambda x, y, z=None: (round(x, nd), round(y, nd)), geom)


def pixel_square_albers(x_alb, y_alb):
    """1 km square built in EPSG:3310 (so it tiles exactly), then reprojected
    to WGS84.  This avoids the gaps you get when constructing the box in
    degrees where dlon varies with latitude.
    """
    sq = box(x_alb - HALF_KM_PROJ, y_alb - HALF_KM_PROJ,
             x_alb + HALF_KM_PROJ, y_alb + HALF_KM_PROJ)
    return transform(lambda x, y, z=None: TO_LL(x, y), sq)


# ---------- CA outline ----------
print("Loading MTBS perimeters…")
mtbs = gpd.read_file(MTBS_SHP).to_crs("EPSG:4326")
mtbs["year"] = pd.to_datetime(mtbs["Ig_Date"]).dt.year

print("Extracting CA outline…")
for fn in os.listdir(CA_SHP_DIR):
    if fn.endswith(".shp"):
        ca = gpd.read_file(os.path.join(CA_SHP_DIR, fn)).to_crs("EPSG:4326")
        g = ca.geometry.union_all().simplify(0.02, preserve_topology=True)
        with open(f"{OUT}/ca_outline.json", "w") as f:
            json.dump(mapping(round_coords(g, 3)), f)
        break

# ---------- Per-fire extraction ----------
all_specs = {}  # evid -> (name, yr, set-tag list)
for ev, n, y in SCRUBBER_FIRES: all_specs.setdefault(ev, (n, y, []))[2].append("scrubber")
for ev, n, y in CHANGE_FIRES:   all_specs.setdefault(ev, (n, y, []))[2].append("change")

fire_index = []
for evid, (name, yr, tags) in all_specs.items():
    print(f"\n=== {name} {yr} ({evid}) — sets: {tags} ===")
    row = mtbs[mtbs.Event_ID == evid]
    if row.empty:
        print("  not found"); continue

    perim_poly_full = row.geometry.iloc[0]
    perim_poly = perim_poly_full.simplify(0.0005, preserve_topology=True)
    perim_geo = mapping(round_coords(perim_poly, 5))
    acres = float(row.BurnBndAc.iloc[0])

    pix = duckdb.query(f"""
        SELECT grid_id, lat, lon, dnbr FROM '{DNBR_PARQUET}'
        WHERE event_id = '{evid}'
    """).to_df()
    if pix.empty:
        print("  no pixels"); continue
    # The dnbr.parquet `lon`/`lat` are actually EPSG:3310 (x/y) — the native
    # Albers grid the rest of the pipeline uses.  Keep them as Albers and
    # transform pixel boxes once at construction time.
    print(f"  pixels: {len(pix)}, mean obs dnbr: {pix.dnbr.mean():.1f}")

    pix_str = ",".join(str(g) for g in pix.grid_id.tolist())
    # Matches the notebook's total-benefit SQL (simulation_replication.ipynb
    # ~line 1485): sum across all land_types (main + spillover rows are
    # stored separately and both contribute), and clamp positive coeffs to 0
    # so model-noise positives don't inflate the counterfactual.  Without the
    # clamp, fires like Dixie / North Complex flip to "worse under spillover".
    sim = duckdb.query(f"""
        SELECT grid_id, sim, year_treat,
               CASE WHEN coeff > 0 THEN 0 ELSE coeff END AS coeff
        FROM '{SIM_PATH}/*.parquet'
        WHERE grid_id IN ({pix_str})
          AND year = {yr}
          AND sim IS NOT NULL
    """).to_df()
    print(f"  sim events: {len(sim):,}, unique sims: {sim.sim.nunique()}")

    rep_year = {}
    for g, sub in sim.groupby("grid_id"):
        rep_year[g] = int(rng.choice(sub["year_treat"].to_numpy()))

    # Per-year cumulative counterfactual: for each Y in 2010..yr,
    # cf[Y][g] = obs + (sum_over_sims sum_coeff_for_year_treat<=Y) / N_SIMS.
    N_SIMS = 1000
    obs_d = dict(zip(pix.grid_id, pix.dnbr.astype(float)))
    INTENSITY_YEARS = list(range(2010, yr + 1))   # dense, year-by-year

    cf = {Y: {} for Y in INTENSITY_YEARS}
    for Y in INTENSITY_YEARS:
        sub = sim[sim.year_treat <= Y]
        sum_per_grid = sub.groupby("grid_id")["coeff"].sum() / N_SIMS
        for g, d in sum_per_grid.items():
            # No max(., 0): negative dnbr (regrowth/water) is meaningful; the
            # dashboard's class bins put anything <270 in the Unburned bucket
            # anyway, so the clamp only distorted the side-panel delta metric.
            cf[Y][g] = obs_d[g] + d

    feats, n_clipped = [], 0
    for _, r in pix.iterrows():
        sq = pixel_square_albers(r.lon, r.lat)
        inter = sq.intersection(perim_poly_full)
        if inter.is_empty: continue
        if inter.geom_type not in ("Polygon", "MultiPolygon"): continue
        geom_json = mapping(round_coords(inter, 5))
        n_clipped += 0 if (inter.geom_type == "Polygon" and inter.equals(sq)) else 1

        sim_d, delta_d = {}, {}
        for Y in INTENSITY_YEARS:
            v = cf[Y].get(r.grid_id, obs_d[r.grid_id])
            sim_d[str(Y)] = round(float(v), 1)
            delta_d[str(Y)] = round(float(v - obs_d[r.grid_id]), 1)

        feats.append({
            "type": "Feature",
            "geometry": geom_json,
            "properties": {
                "obs": round(float(r.dnbr), 1),
                "ft": int(rep_year.get(r.grid_id, 0)),
                "sim": sim_d,
                "delta": delta_d,
            },
        })
    print(f"  polygons: {len(feats)} (boundary-clipped: {n_clipped})")

    set_tag = "both" if len(tags) > 1 else tags[0]
    fc = {
        "type": "FeatureCollection",
        "name": name, "year": yr, "id": evid,
        "acres": int(acres),
        "intensity_years": INTENSITY_YEARS,
        "min_year": 2009,
        "max_year": yr,
        "perimeter": perim_geo,
        "features": feats,
    }
    slug = name.lower().replace(" ", "_")
    with open(f"{OUT}/fire_{slug}.json", "w") as f:
        json.dump(fc, f, separators=(",", ":"))
    print(f"  wrote fire_{slug}.json")

    fire_index.append({
        "id": evid, "name": name, "slug": slug, "year": yr,
        "acres": int(acres), "n_pixels": len(feats),
        "dnbr_obs_mean": round(float(pix.dnbr.mean()), 1),
        "centroid": [round(c, 4) for c in perim_poly.centroid.coords[0]],
        "bbox": [round(c, 4) for c in perim_poly.bounds],
        "set": set_tag,
        "min_year": 2009,
        "max_year": yr,
    })

with open(f"{OUT}/fires_index.json", "w") as f:
    json.dump(fire_index, f, indent=2)
print(f"\nwrote fires_index.json")

# ---------- State-mock treatments (kept for inset) ----------
print("\nState-mock treatments (one sim)…")
state = duckdb.query(f"""
    SELECT grid_id, ANY_VALUE(lat) lat, ANY_VALUE(lon) lon, MIN(year_treat) year_treat
    FROM '{SIM_PATH}/sim_1_10.parquet' WHERE sim = 1
    GROUP BY grid_id
""").to_df()
lons, lats = TO_LL_TR.transform(state["lon"].to_numpy(), state["lat"].to_numpy())
state_out = [{"lon": round(float(lo), 3), "lat": round(float(la), 3), "y": int(y)}
             for lo, la, y in zip(lons, lats, state["year_treat"])]
with open(f"{OUT}/state_mock_treatments.json", "w") as f:
    json.dump(state_out, f, separators=(",", ":"))
print(f"  wrote {len(state_out)} pts")
print("\nDone.")
