
# irctc_rirm_super_app.py
# IRCTC IRM — Unified Super App (v9)
# --------------------------------------------------------------------------------------------------
# Combines: colorful UI, ML forecast, EMSR-b, multi-leg optimizer, Monte Carlo coach revenue using
# PRS fare rules, true berth seatmaps + swap/squeeze, family seating with constraints & severity
# heatmaps, YAML rules engine, DB persistence (rules/seatmaps/allocations), REST hooks, and exports.
#
# Install:
#   pip install streamlit pandas numpy altair pyyaml requests sqlalchemy psycopg2-binary reportlab scikit-learn xgboost scipy
#
# Run:
#   streamlit run irctc_rirm_super_app.py
#
import os, io, json, math, zipfile, datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# Optional deps
try:
    import yaml
except Exception:
    yaml = None

# Optional DB
USE_DB = os.getenv("DATABASE_URL") is not None
if USE_DB:
    try:
        from irm_db import init_engine, Base, RulesVersion, SeatmapVersion, SeatRow, FamilyAllocation, \
                           save_seatmap_version, save_family_allocations, save_rules_version, load_latest_rules_json
        engine = init_engine(os.getenv("DATABASE_URL"))
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        USE_DB = False
        print("DB disabled:", e)

st.set_page_config(page_title="IRCTC IRM — Unified Super App", layout="wide")

# --------------------------
# Theme
# --------------------------
PRIMARY = "#635bff"; ACCENT  = "#22c55e"; WARN = "#f59e0b"; DANGER = "#ef4444"; MUTED = "#64748b"
st.markdown(f"""
<style>
  .pill {{ display:inline-block; padding:4px 10px; border-radius:999px; background:{PRIMARY}15; color:{PRIMARY}; font-weight:600; margin-right:6px; }}
  .pill-green {{ background:{ACCENT}20; color:{ACCENT}; }}
  .pill-amber {{ background:{WARN}26; color:{WARN}; }}
  .pill-red {{ background:{DANGER}26; color:{DANGER}; }}
  .section-title {{ font-size:1.15rem; font-weight:800; margin-top:6px; color:{PRIMARY}; }}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Constants
# --------------------------
DEFAULT_STATIONS = ["NDLS","CNB","LKO","ALD","MGS","BSB","PNBE","HWH"]
DEFAULT_CLASSES  = ["SL","3A","2A","1A","CC"]
SEATS_PER_COACH = {"SL":72, "3A":64, "2A":48, "1A":24, "CC":72}
BASE_FARES = {"SL": 500, "3A": 1500, "2A": 2400, "1A": 4000, "CC": 900}
FARE_BUCKETS = {
    "SL":[0.8,0.9,1.0,1.1,1.2],
    "3A":[0.7,0.85,1.0,1.2,1.35],
    "2A":[0.75,0.9,1.0,1.15,1.3],
    "1A":[0.85,1.0,1.15,1.3,1.5],
    "CC":[0.8,0.95,1.0,1.1,1.25]
}

# Default YAML rules (editable)
DEFAULT_RULES = {
  "pricing": {
    "pace_uplift": [
      {"delta_min": 0.10, "multiplier": 1.12},
      {"delta_min": 0.03, "multiplier": 1.06},
      {"delta_min": -0.03, "multiplier": 1.00},
      {"delta_min": -0.10, "multiplier": 0.96},
      {"delta_min": -1.00, "multiplier": 0.90}
    ],
    "seasonality": {"Low": 0.95, "Normal": 1.00, "Festive/Peak": 1.15},
    "class_multipliers": {"SL": 1.00, "3A": 1.00, "2A": 1.00, "1A": 1.05, "CC": 1.00}
  },
  "tatkal": {"open_hours":24, "premium_open_hours":24, "premium_uplift_pct":35},
  "overbooking": {"max_percent":15, "noshow_fallback_percent":8},
  "upgrade_policy": {"enable_bid_for_upgrade": True, "min_bid": 150, "max_bid": 1500},
  "allocation": {"priority_rule": "Longest first", "rac_buffer_per_coach": 4}
}

# PRS fare rules (can be fetched from PRS; else defaults)
DEFAULT_FARE_RULES = {
  "distance_slabs_km": [0, 51, 101, 151, 201, 301, 401, 501, 751, 1001],
  "base_fare_per_km": {"SL": 0.9, "3A": 2.5, "2A": 3.8, "1A": 7.0, "CC": 1.8},
  "slab_multipliers": [1.00, 0.98, 0.96, 0.95, 0.94, 0.93, 0.92, 0.90, 0.88, 0.86],
  "superfast_surcharge": {"threshold_km": 100, "amount": {"SL":45,"3A":75,"2A":90,"1A":120,"CC":60}},
  "reservation_fee": {"SL":20,"3A":40,"2A":50,"1A":60,"CC":30},
  "gst_pct": {"AC": 5.0, "NONAC": 0.0},
  "tatkal": {"window_hours":24, "pct_uplift": {"SL":10,"3A":30,"2A":30,"1A":30,"CC":20}},
  "premium_tatkal": {"window_hours":24, "pct_uplift": {"SL":30,"3A":40,"2A":40,"1A":40,"CC":30}},
  "dynamic_surge_steps": [0.0, 0.05, 0.1, 0.15],
  "dynamic_occupancy_bands": [0.7, 0.85, 0.95, 1.05]
}

# --------------------------
# Sidebar: Global controls & rules loading
# --------------------------
with st.sidebar:
    st.title("🎛️ Global Controls")
    train_number = st.text_input("Train Number", "12301")
    run_date = st.date_input("Departure Date", value=dt.date.today() + dt.timedelta(days=21))
    classes_selected = st.multiselect("Classes", DEFAULT_CLASSES, default=["SL","3A","2A"])

    st.markdown("#### Coaches per class")
    coach_counts = {}
    for cls in DEFAULT_CLASSES:
        default = (8 if cls=="SL" else 4 if cls in ["3A","CC"] else 2 if cls=="2A" else 1)
        coach_counts[cls] = st.number_input(f"{cls} coaches", 0, 30, default, 1)

    st.markdown("#### Demand & Competition")
    base_demand_index = st.slider("Demand index (0.5x–2.0x)", 50, 200, 110, 5) / 100.0
    seasonality = st.select_slider("Seasonality", ["Low", "Normal", "Festive/Peak"], value="Normal")
    pace_bias = st.slider("Booking pace bias (−20..+20)", -20, 20, 0, 5) / 100.0
    competitor_pressure = st.slider("Competitor pressure %", 0, 50, 15, 5)

    st.markdown("#### Capacity & Risk")
    tatkal_share = st.slider("Tatkal quota % (per class)", 0, 40, 20, 5)
    overbooking_pct = st.slider("Overbooking allowance %", 0, 30, 10, 5)
    no_show_rate = st.slider("No-show rate %", 0, 25, 8)

    st.markdown("#### PRS/UTS & Optimizer")
    prs_base = st.text_input("PRS API base URL", "")
    uts_base = st.text_input("UTS API base URL", "")
    api_key  = st.text_input("Auth token / API key", type="password")
    optimizer_base = st.text_input("Optimizer API base", "http://localhost:8010")

    st.markdown("---")
    st.markdown("### Rules Engine (YAML)")
    rules_source = st.radio("Rules source", ["Default", "Upload YAML", "Edit inline"], index=0)
    uploaded_yaml = st.file_uploader("Upload rules.yaml", type=["yaml","yml"]) if rules_source=="Upload YAML" else None
    inline_yaml = st.text_area("Edit YAML", height=260) if rules_source=="Edit inline" else None

# YAML helpers
def safe_yaml_load(txt: str) -> dict:
    if yaml is None: raise RuntimeError("PyYAML missing. pip install pyyaml")
    return yaml.safe_load(txt)
def safe_yaml_dump(obj: dict) -> str:
    if yaml is None: raise RuntimeError("PyYAML missing. pip install pyyaml")
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)

# Resolve rules
if rules_source == "Default":
    rules = DEFAULT_RULES
elif rules_source == "Upload YAML" and uploaded_yaml:
    try:
        rules = safe_yaml_load(uploaded_yaml.read().decode("utf-8"))
        st.sidebar.success("Rules loaded from YAML.")
    except Exception as e:
        st.sidebar.error(f"YAML error: {e}"); rules = DEFAULT_RULES
elif rules_source == "Edit inline" and inline_yaml:
    try:
        rules = safe_yaml_load(inline_yaml)
        st.sidebar.success("Rules loaded from editor.")
    except Exception as e:
        st.sidebar.error(f"YAML error: {e}"); rules = DEFAULT_RULES
else:
    rules = DEFAULT_RULES
st.sidebar.download_button("⬇️ Download current rules.yaml", data=safe_yaml_dump(rules).encode("utf-8"),
                           file_name="rules.yaml", mime="text/yaml")

# PRS fare rules
def load_prs_fare_rules(prs_base, api_key):
    if not prs_base: return DEFAULT_FARE_RULES
    try:
        import requests
        r = requests.get(f"{prs_base}/fare_rules", headers={"Authorization": f"Bearer {api_key}"}, timeout=6)
        if r.status_code==200: return r.json()
    except Exception:
        pass
    return DEFAULT_FARE_RULES
fare_rules = load_prs_fare_rules(prs_base, api_key)

# --------------------------
# Utilities (shared)
# --------------------------
def sim_build_od_table(stations):
    rows = []
    for i in range(len(stations)-1):
        for j in range(i+1, len(stations)):
            origin, dest = stations[i], stations[j]
            legs = j - i
            dist = 150 * legs
            rows.append({"OD": f"{origin}-{dest}", "O": origin, "D": dest, "legs": legs, "distance": dist})
    return pd.DataFrame(rows)
od_df = sim_build_od_table(DEFAULT_STATIONS)

def season_multiplier(tag, rules):
    return rules["pricing"]["seasonality"].get(tag, 1.0)

def _norm_ppf(p):
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        x = p - 0.5
        return np.sqrt(2) * np.sign(x) * np.sqrt(np.log(1/(1 - 2*abs(x))))

def emsr_b(fares_desc, means, sds, cap):
    fares = np.array(sorted(fares_desc, reverse=True), dtype=float)
    means = np.array(means, dtype=float)
    sds = np.array(sds, dtype=float)
    cum_mean = np.cumsum(means)
    cum_sd = np.sqrt(np.cumsum(sds**2))
    prot = np.zeros_like(fares)
    for k in range(len(fares)-1):
        fk, fk1 = fares[k], fares[k+1]
        crit = fk1 / fk
        z = _norm_ppf(1 - crit)
        y = cum_mean[k] + cum_sd[k]*z
        prot[k] = max(0.0, min(cap, y))
    cum_prot = np.maximum.accumulate(prot)
    booking_limits = np.diff(np.append(cum_prot, cap))
    return prot, booking_limits

def capacity_for_class(cls, tatkal_share_pct, coaches):
    total = SEATS_PER_COACH.get(cls, 50) * coaches
    tatkal = math.floor(total * tatkal_share_pct / 100.0)
    gen = total - tatkal
    return {"total": total, "general": gen, "tatkal": tatkal}

def baseline_forecast(od_df, cls, base_idx, season_tag, competitor_pct, rules):
    s_mult = season_multiplier(season_tag, rules)
    base = BASE_FARES[cls]
    price_expect = base * np.mean(FARE_BUCKETS[cls]) * rules["pricing"]["class_multipliers"].get(cls, 1.0)
    mean = (120 / (price_expect / 100.0)) * base_idx * s_mult
    recs = []
    for _, r in od_df.iterrows():
        dist_factor = max(0.6, 1.2 - (r["legs"] - 1) * 0.1)
        comp_factor = 1.0 - (competitor_pct / 100.0) * (0.2 + 0.1 * (r["legs"] - 1))
        mu = max(2.0, mean * dist_factor * comp_factor)
        sd = max(1.5, mu * 0.35)
        recs.append({"OD": r["OD"], "mu": mu, "sd": sd})
    return pd.DataFrame(recs)

def simulate_booking_curve(days_to_dep=30, bias=0.0):
    t = np.arange(days_to_dep + 1)
    p, q = 0.03 + bias*0.1, 0.4 + max(-0.2, bias)
    adoption = (np.exp((p+q)*t) - 1)/(np.exp((p+q)*t) + (q/p))
    adoption = (adoption - adoption.min()) / (adoption.max() - adoption.min() + 1e-9)
    adoption = np.clip(adoption + np.random.normal(0, 0.02, size=adoption.size), 0, 1)
    return t, adoption

# Seatmaps — true berth layouts
BERTH_PATTERNS = {
    "SL": ["LB","MB","UB","LB*","UB*","SL","SU","MB*"],
    "3A": ["LB","MB","UB","LB*","UB*","SL","SU","MB*"],
    "2A": ["LB","UB","LB*","UB*","SL","SU"],
    "1A": ["L","U","L*","U*"],
    "CC": ["A","B","C","D","E"]
}
def seat_layout_grid(cls) -> Tuple[int,int]:
    if cls == "SL": return 12, 6
    if cls == "3A": return 8, 8
    if cls == "2A": return 8, 6
    if cls == "1A": return 6, 4
    if cls == "CC": return 10, 6
    return 10, 6

def coaches_to_bays(cls, coaches: int) -> int:
    seats_per_bay = len(BERTH_PATTERNS[cls])
    total_seats = SEATS_PER_COACH.get(cls, seats_per_bay*8) * max(1, coaches)
    return max(1, total_seats // seats_per_bay // max(1, coaches))

def build_true_berths(cls, coach_id: str, num_bays: int, women_only_bays: int = 0):
    pattern = BERTH_PATTERNS[cls]
    rows = []; seat_no=1
    rng = np.random.default_rng(abs(hash(coach_id)) % (2**32))
    booked_ratio = 0.52
    for bay in range(1, num_bays+1):
        is_women_only = (bay <= women_only_bays)
        for pos, bt in enumerate(pattern, start=1):
            status="FREE"; pnr=None; od=None
            if rng.random() < booked_ratio:
                status="CONF"; pnr=f"{coach_id}-{seat_no:03d}"; od="NDLS-LKO"
            rows.append({"coach": coach_id, "seat": seat_no, "bay": bay, "pos": pos,
                         "berth_type": bt, "status": status, "pnr": pnr, "od": od,
                         "women_only": is_women_only})
            seat_no += 1
    return pd.DataFrame(rows)

def generate_train_seatmaps_true(classes, coach_counts: Dict[str,int], women_ratio=0.1, prs_inventory=None):
    frames = []
    w_bays_map = {}
    if prs_inventory and "classes" in prs_inventory:
        for entry in prs_inventory["classes"]:
            if "women_only_bays_per_coach" in entry:
                w_bays_map[entry["class"]] = int(entry["women_only_bays_per_coach"])
    for cls in classes:
        coaches = coach_counts.get(cls,0)
        if coaches<=0: continue
        bays_per_coach = coaches_to_bays(cls, coaches)
        w_bays = w_bays_map.get(cls, max(0, int(round(women_ratio * bays_per_coach))))
        for k in range(coaches):
            coach_id = f"{cls}-{k+1:02d}"
            frames.append(build_true_berths(cls, coach_id, bays_per_coach, women_only_bays=w_bays))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["coach","seat","bay","pos","berth_type","status","pnr","od","women_only"]
    )

# Seat swap/squeeze (from earlier versions)
def swap_seats(seatmap: pd.DataFrame, coach_a: str, seat_a: int, coach_b: str, seat_b: int):
    a_idx = (seatmap["coach"]==coach_a) & (seatmap["seat"]==seat_a)
    b_idx = (seatmap["coach"]==coach_b) & (seatmap["seat"]==seat_b)
    if not a_idx.any() or not b_idx.any(): return False, "Seat not found."
    row_a = seatmap.loc[a_idx].iloc[0].to_dict(); row_b = seatmap.loc[b_idx].iloc[0].to_dict()
    for k in ["status","pnr","od"]:
        seatmap.loc[a_idx, k], seatmap.loc[b_idx, k] = row_b[k], row_a[k]
    return True, "Swapped."
def squeeze_confirm_from_rac(seatmap: pd.DataFrame, target_od: str):
    confirmed = 0; assigned = []
    rac_idx = seatmap.index[seatmap["status"]=="RAC"].tolist()
    free_idx = seatmap.index[seatmap["status"]=="FREE"].tolist()
    for ri in rac_idx:
        if not free_idx: break
        fi = free_idx.pop(0)
        seatmap.at[fi, "status"] = "CONF"; seatmap.at[fi, "pnr"] = f"RACCONF-{seatmap.at[ri,'coach']}-{seatmap.at[ri,'seat']}"; seatmap.at[fi, "od"] = target_od
        seatmap.at[ri, "status"] = "FREE"; seatmap.at[ri, "pnr"]=None; seatmap.at[ri, "od"]=None
        confirmed += 1; assigned.append((seatmap.at[fi,"coach"], int(seatmap.at[fi,"seat"])))
    return confirmed, assigned

# PRS connectors (stubs)
def fetch_prs_inventory(train_no, run_date, prs_base, api_key):
    if not prs_base: return None
    try:
        import requests
        r = requests.get(f"{prs_base}/inventory", params={"train":train_no, "date":str(run_date)},
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=8)
        if r.status_code==200: return r.json()
    except Exception: pass
    return None

# Fare calc
def calc_fare(distance_km, cls, occupancy, hours_to_dep, rules:dict):
    slabs = rules["distance_slabs_km"]; idx = max(i for i,s in enumerate(slabs) if distance_km >= s)
    per_km = rules["base_fare_per_km"][cls] * rules["slab_multipliers"][min(idx, len(rules["slab_multipliers"])-1)]
    fare = per_km * max(distance_km, 1)
    sfs = rules["superfast_surcharge"]
    if distance_km >= sfs["threshold_km"]: fare += sfs["amount"][cls]
    fare += rules["reservation_fee"][cls]
    bands = rules["dynamic_occupancy_bands"]; steps = rules["dynamic_surge_steps"]
    band_idx = sum(1 for b in bands if occupancy >= b) - 1; band_idx = max(-1, band_idx)
    if band_idx >= 0: fare *= (1.0 + steps[min(band_idx, len(steps)-1)])
    if hours_to_dep <= rules["premium_tatkal"]["window_hours"]:
        fare *= (1.0 + rules["premium_tatkal"]["pct_uplift"][cls]/100.0)
    elif hours_to_dep <= rules["tatkal"]["window_hours"]:
        fare *= (1.0 + rules["tatkal"]["pct_uplift"][cls]/100.0)
    is_ac = cls in ["3A","2A","1A","CC"]; gst = rules["gst_pct"]["AC" if is_ac else "NONAC"]/100.0
    fare *= (1.0 + gst)
    return round(fare, 2)

# Monte Carlo revenue
def mc_revenue_per_coach(seatmap_df: pd.DataFrame, od_df: pd.DataFrame, fare_rules: dict,
                         hours_to_dep=24, trials=1000, no_show=no_show_rate, overbook=overbooking_pct):
    if seatmap_df.empty: return pd.DataFrame(columns=["coach","rev_mean","rev_p10","rev_p90","load_mean"])
    tmp = seatmap_df.copy(); tmp["class"] = tmp["coach"].str.split("-").str[0]
    od_list = od_df.to_dict("records")
    rng = np.random.default_rng(11)
    coaches = sorted(tmp["coach"].unique().tolist()); out = []
    for coach in coaches:
        cls = coach.split("-")[0]; seats = tmp[tmp["coach"]==coach].copy(); cap = len(seats)
        draws = []; loads = []
        for t in range(trials):
            occ0 = (seats["status"]=="CONF").sum(); cap_eff = int(cap*(1+overbook/100.0))
            extra = rng.binomial(n=max(cap_eff-occ0,0), p=0.5); occ = min(cap_eff, occ0 + extra)
            show = rng.binomial(n=occ, p=max(0.0, 1 - no_show/100.0)); loads.append(show/cap)
            if show>0:
                ods = rng.choice(od_list, size=show, replace=True); dists = np.array([o["distance"] for o in ods])
                occupancy_ratio = min(1.2, show/cap)
                fares = [calc_fare(d, cls, occupancy_ratio, hours_to_dep, fare_rules) for d in dists]
                draws.append(np.sum(fares))
            else: draws.append(0.0)
        draws = np.array(draws); loads = np.array(loads)
        out.append({"coach": coach, "rev_mean": float(np.mean(draws)), "rev_p10": float(np.quantile(draws, 0.10)),
                    "rev_p90": float(np.quantile(draws, 0.90)), "load_mean": float(np.mean(loads))})
    return pd.DataFrame(out)

# Family constraints
RANK = {"LB":1,"L":1,"A":1,"B":2,"C":3,"MB":3,"UB":4,"U":4,"SL":2,"SU":5,"D":4,"E":5,"LB*":2,"UB*":3,"MB*":4,"L*":2,"U*":3}
def seat_is_upper(bt: str) -> bool: return bt in {"UB","SU","U","UB*","U*"}
def seat_is_lower(bt: str) -> bool: return bt in {"LB","SL","L","LB*","L*"}

def synthesize_families(n_families=12, max_size=5, seed=7, women_only_toggle=False):
    rng = np.random.default_rng(seed); fams = []
    for fid in range(1, n_families+1):
        size = int(rng.integers(2, max_size+1)); ages = list(rng.integers(4, 70, size=size))
        if rng.random()<0.6: ages[0] = rng.integers(2, 11)
        if rng.random()<0.3: ages[-1] = rng.integers(61, 75)
        w_only = women_only_toggle or (rng.random() < 0.1)
        genders = [int(rng.random()<0.5) for _ in range(size)]  # 1=female, 0=male
        fams.append({"family_id": fid, "size": size, "ages": ages, "women_only": w_only, "genders": genders})
    return fams

def assign_family_to_bay(bay_df: pd.DataFrame, fam: dict, flags: dict):
    free = bay_df[bay_df["status"]=="FREE"].copy()
    if free.empty or len(free) < fam["size"]: return [], {}, []
    if fam["women_only"] and not bool(bay_df["women_only"].iloc[0]): return [], {}, []
    violations = {"child_upper":0, "elder_upper":0, "women_only_violation":0, "mixed_gender":0}; viol_seats=[]
    if fam["women_only"] and flags.get("women_only_booking"):
        if any(g==0 for g in fam["genders"]): violations["mixed_gender"]=1; return [], violations, []
    children = [a for a in fam["ages"] if a <= 12]; elders = [a for a in fam["ages"] if a >= 60]
    def score(bt):
        base = RANK.get(bt, 9)
        if flags.get("protect_children") and children and seat_is_lower(bt): base -= 1.5
        if flags.get("protect_elderly") and elders and seat_is_lower(bt): base -= 1.0
        if flags.get("protect_children") and children and seat_is_upper(bt): base += 1.0
        if flags.get("protect_elderly") and elders and seat_is_upper(bt): base += 1.0
        return base
    free["score"] = free["berth_type"].apply(score)
    picks = free.sort_values("score").head(fam["size"])
    child_need = sum(1 for a in fam["ages"] if a<=12); elder_need = sum(1 for a in fam["ages"] if a>=60)
    assigned_bts = picks["berth_type"].tolist()
    for i in range(min(child_need, len(assigned_bts))):
        if seat_is_upper(assigned_bts[i]): violations["child_upper"] += 1; viol_seats.append(int(picks.iloc[i]["seat"]))
    if elder_need>0 and seat_is_upper(assigned_bts[-1]): violations["elder_upper"] += 1; viol_seats.append(int(picks.iloc[-1]["seat"]))
    return picks["seat"].tolist(), violations, viol_seats

def family_seating_optimizer(seatmap_df: pd.DataFrame, cls: str, n_fams=12, flags=None):
    if flags is None: flags = {}
    fams = synthesize_families(n_families=n_fams, women_only_toggle=flags.get("women_only_booking", False))
    df = seatmap_df.copy(); logs = []; conflicts = []; seat_level = []
    for fam in fams:
        placed = False
        for coach_id, dfc in df[df["coach"].str.startswith(cls)].groupby("coach"):
            for bay, bay_df in dfc.groupby("bay"):
                picks, viol, viol_seats = assign_family_to_bay(bay_df, fam, flags)
                if picks:
                    df.loc[df["seat"].isin(picks) & (df["coach"]==coach_id), ["status","pnr","od"]] = ["CONF", f"FAM-{fam['family_id']}", "GROUP-OD"]
                    logs.append(f"Family {fam['family_id']} (size {fam['size']}) → {coach_id} bay {bay}: seats {picks}")
                    if any(viol.values()):
                        conflicts.append({"coach": coach_id, "bay": int(bay), **viol})
                        for s in viol_seats: seat_level.append({"coach": coach_id, "bay": int(bay), "seat": int(s), **viol})
                    placed = True; break
            if placed: break
        if not placed:
            logs.append(f"Family {fam['family_id']} could not be placed contiguously.")
            conflicts.append({"coach":"NA","bay":-1,"child_upper":0,"elder_upper":0,"women_only_violation":0,"mixed_gender":0})
    conf_df = pd.DataFrame(conflicts) if conflicts else pd.DataFrame(columns=["coach","bay","child_upper","elder_upper","women_only_violation","mixed_gender"])
    seat_conf_df = pd.DataFrame(seat_level) if seat_level else pd.DataFrame(columns=["coach","bay","seat","child_upper","elder_upper","women_only_violation","mixed_gender"])
    return df, logs, conf_df, seat_conf_df

SEVERITY = {"child_upper": 2.0, "elder_upper": 1.5, "women_only_violation": 3.0, "mixed_gender": 4.0}
def severity_score(row): return sum(row.get(k,0)*w for k,w in SEVERITY.items())

# Multi-leg optimizer helpers
def build_leg_index(stations): return [(stations[i], stations[i+1]) for i in range(len(stations)-1)]
def od_to_legs_map(od_row, stations):
    O, D = od_row["O"], od_row["D"]; i, j = stations.index(O), stations.index(D)
    return [(stations[k], stations[k+1]) for k in range(i, j)]
def allocate_multileg(od_df, demand_df, seg_capacity, rule="Longest first"):
    df = od_df.merge(demand_df, on="OD").copy()
    if rule == "Longest first": df = df.sort_values(["legs", "mu"], ascending=[False, False])
    elif rule == "Shortest first": df = df.sort_values(["legs", "mu"], ascending=[True, False])
    else: df = df.sort_values("mu", ascending=False)
    df["alloc"] = 0; cap = seg_capacity.copy()
    def seg_min_cap(od_leg_list): return min(cap.get(seg, 0) for seg in od_leg_list)
    for idx, row in df.iterrows():
        od_legs = od_to_legs_map(row, DEFAULT_STATIONS); max_possible = seg_min_cap(od_legs)
        give = min(int(round(row["mu"])), max_possible)
        if give>0: df.at[idx,"alloc"]=give; [cap.__setitem__(seg, cap[seg]-give) for seg in od_legs]
    df["spill"] = (df["mu"] - df["alloc"]).clip(lower=0); return df, cap

# --------------------------
# Tabs
# --------------------------
tabs = st.tabs([
    "Dashboard","Forecast & EMSR","Multi-leg Optimizer","Coach Revenue (MC/PRS)",
    "Seatmaps (True & Tools)","Family Seating & Conflicts","Rules (YAML)","DB Sync","REST Hooks","Exports"
])

stations = DEFAULT_STATIONS
t, adoption = simulate_booking_curve(days_to_dep=30, bias=pace_bias)
pace_now, pace_expected = adoption[-1], 0.65

# 1) Dashboard
with tabs[0]:
    st.markdown(f"<div class='section-title'>🚆 IRCTC IRM — Unified Super App</div>", unsafe_allow_html=True)
    st.caption(f"<span class='pill'>Train {train_number}</span> "
               f"<span class='pill-green'>{run_date}</span> "
               f"<span class='pill-amber'>Season: {seasonality}</span> "
               f"<span class='pill-red'>Tatkal {tatkal_share}%</span>", unsafe_allow_html=True)
    pace_df = pd.DataFrame({"Day": -np.arange(len(adoption)), "Pace": adoption})
    pace_chart = alt.Chart(pace_df).mark_line(strokeWidth=3).encode(
        x=alt.X("Day:Q", title="Days to departure"),
        y=alt.Y("Pace:Q", title="Cumulative booking share"),
        color=alt.value(PRIMARY),
        tooltip=["Day", alt.Tooltip("Pace:Q", format=".0%")]
    ).properties(height=220)
    st.altair_chart(pace_chart, use_container_width=True)
    st.markdown("### Capacity Snapshot")
    snap = []
    for cls in classes_selected:
        inv = capacity_for_class(cls, tatkal_share, coach_counts.get(cls, 0))
        eff = max(1, int(inv["total"] * (1 + overbooking_pct/100.0) * (1 - no_show_rate/100.0)))
        snap.append({"Class": cls, "Coaches": coach_counts.get(cls,0), "Capacity": inv["total"], "Eff.Cap": eff, "Base fare": BASE_FARES[cls]})
    st.dataframe(pd.DataFrame(snap), use_container_width=True)

# 2) Forecast & EMSR (with optional ML)
with tabs[1]:
    st.markdown("<div class='section-title'>Forecast & EMSR</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload historical CSV (O, D, class, distance, price, dow, season, bookings)", type=["csv"], key="mlcsv")
    ml_enable = st.checkbox("Use ML forecast if CSV uploaded", value=False)
    for cls in classes_selected:
        st.markdown(f"#### Class {cls}")
        df_demand = None
        base = BASE_FARES[cls]
        price_expect = base * np.mean(FARE_BUCKETS[cls]) * rules["pricing"]["class_multipliers"].get(cls, 1.0)
        if uploaded and ml_enable:
            try:
                df = pd.read_csv(uploaded)
                req = {"O","D","class","distance","price","dow","season","bookings"}
                assert req.issubset(df.columns)
                df = df.dropna().copy()
                X = pd.get_dummies(df[["O","D","class","season","dow","distance","price"]], drop_first=True)
                y = df["bookings"].astype(float)
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42)
                except Exception:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(random_state=42)
                model.fit(X, y)
                dow = dt.date.today().weekday()
                feat = []
                for _, r in od_df.iterrows():
                    feat.append({"O": r["O"], "D": r["D"], "class": cls, "season": seasonality, "dow": dow, "distance": r["distance"], "price": price_expect})
                fdf = pd.get_dummies(pd.DataFrame(feat)[["O","D","class","season","dow","distance","price"]], drop_first=True)
                for c in X.columns:
                    if c not in fdf.columns: fdf[c]=0
                fdf = fdf[X.columns]
                preds = np.maximum(0.5, model.predict(fdf)); sds = np.maximum(1.5, preds*0.35)
                df_demand = pd.DataFrame({"OD": od_df["OD"], "mu": preds, "sd": sds})
                st.success("ML forecast applied.")
            except Exception as e:
                st.warning(f"ML forecast failed ({e}); using baseline.")
                df_demand = baseline_forecast(od_df, cls, base_demand_index, seasonality, competitor_pressure, rules)
        else:
            df_demand = baseline_forecast(od_df, cls, base_demand_index, seasonality, competitor_pressure, rules)

        hm = od_df.merge(df_demand, on="OD")
        st.altair_chart(alt.Chart(hm).mark_rect().encode(
            x=alt.X("O:N", title="Origin"), y=alt.Y("D:N", title="Destination"),
            color=alt.Color("mu:Q", title="Mean demand", scale=alt.Scale(scheme="turbo")),
            tooltip=["OD", alt.Tooltip("mu:Q", format=".1f"), alt.Tooltip("sd:Q", format=".1f")]
        ).properties(height=260), use_container_width=True)

        fare_table = sorted([round(BASE_FARES[cls]*m) for m in FARE_BUCKETS[cls]], reverse=True)
        if seasonality == "Low": shares = np.array([0.35, 0.25, 0.2, 0.12, 0.08])
        elif seasonality == "Festive/Peak": shares = np.array([0.10, 0.15, 0.25, 0.25, 0.25])
        else: shares = np.array([0.20, 0.22, 0.22, 0.20, 0.16])
        shares = shares / shares.sum()
        total_mu = df_demand["mu"].sum(); total_sd = np.sqrt((df_demand["sd"]**2).sum())
        bucket_means = total_mu * shares; bucket_sds = total_sd * shares
        inv = capacity_for_class(cls, tatkal_share, coach_counts.get(cls, 0))
        eff_cap = max(1, int(inv["total"] * (1 + overbooking_pct/100.0) * (1 - no_show_rate/100.0)))
        prot, book_limits = emsr_b(fare_table, bucket_means, bucket_sds, eff_cap)
        emsr_df = pd.DataFrame({"rank": np.arange(1, len(prot)+1), "fare": fare_table,
                                "protection_up_to": np.round(prot,1), "booking_limit": np.round(book_limits,0).astype(int)})
        c1, c2 = st.columns(2)
        with c1: st.dataframe(pd.DataFrame({"bucket": [f"B{i+1}" for i in range(len(fare_table))], "fare": fare_table}), use_container_width=True)
        with c2: st.dataframe(emsr_df, use_container_width=True)
        st.session_state.setdefault("exports", {})
        st.session_state["exports"][f"{cls}_demand.csv"] = df_demand
        st.session_state["exports"][f"{cls}_emsr.csv"] = emsr_df

# 3) Multi-leg Optimizer
with tabs[2]:
    st.markdown("<div class='section-title'>Multi-leg Optimizer (segment-feasible)</div>", unsafe_allow_html=True)
    prio_rule = st.selectbox("Priority rule", ["Longest first", "Shortest first", "Proportional to μ"])
    cls_opt = st.selectbox("Class to optimize", classes_selected)
    exp_key = f"{cls_opt}_demand.csv"
    exp_dict = st.session_state.get("exports", {})
df_demand = exp_dict.get(exp_key)
if df_demand is None or not isinstance(df_demand, pd.DataFrame) or df_demand.empty:
    df_demand = baseline_forecast(od_df, cls_opt, base_demand_index, seasonality, competitor_pressure, rules)
    inv = capacity_for_class(cls_opt, tatkal_share, coach_counts.get(cls_opt, 0))
    eff_cap = max(1, int(inv["total"] * (1 + overbooking_pct/100.0) * (1 - no_show_rate/100.0)))
    leg_index = build_leg_index(stations); seg_cap_each = max(1, eff_cap // len(leg_index))
    seg_capacity = {leg: seg_cap_each for leg in leg_index}
    if prio_rule == "Proportional to μ":
        df_tmp = od_df.merge(df_demand, on="OD"); total_mu = df_tmp["mu"].sum()
        df_tmp["alloc"] = (df_tmp["mu"]/max(total_mu,1e-6))*eff_cap
        alloc_df = df_tmp[["OD","O","D","legs","mu","alloc"]]; seg_left = seg_capacity
    else:
        alloc_df, seg_left = allocate_multileg(od_df, df_demand, seg_capacity, prio_rule)
    a1, a2, a3 = st.columns(3)
    a1.metric("Allocated seats", int(alloc_df["alloc"].sum()))
    a2.metric("Total spill", f"{alloc_df.get('spill', pd.Series([])).sum():.1f}")
    a3.metric("Segments @0 cap", sum(1 for v in seg_left.values() if v<=0))
    st.altair_chart(alt.Chart(alloc_df).mark_bar().encode(
        x=alt.X("OD:N", sort=None, title="OD"), y=alt.Y("alloc:Q", title="Allocated seats"),
        color=alt.value(ACCENT), tooltip=["OD", alt.Tooltip("alloc:Q", format=".0f"), alt.Tooltip("mu:Q", format=".1f")]
    ).properties(height=260), use_container_width=True)
    st.session_state["exports"][f"{cls_opt}_alloc_multileg.csv"] = alloc_df

# 4) Coach Revenue (Monte Carlo with PRS fare rules)
with tabs[3]:
    st.markdown("<div class='section-title'>Coach Revenue — Monte Carlo (PRS fare rules)</div>", unsafe_allow_html=True)
    inv_json = fetch_prs_inventory(train_number, run_date, prs_base, api_key)
    seatmap_true = generate_train_seatmaps_true(classes_selected, coach_counts, women_ratio=0.1, prs_inventory=inv_json)
    trials = st.slider("Trials", 100, 5000, 1000, 100)
    hours_to_dep = st.slider("Hours to departure", 6, 72, 24, 6)
    mc = mc_revenue_per_coach(seatmap_true, od_df, fare_rules, hours_to_dep=hours_to_dep, trials=trials)
    st.dataframe(mc, use_container_width=True)
    if not mc.empty:
        st.altair_chart(alt.Chart(mc).mark_bar().encode(
            x=alt.X("coach:N", sort=None), y=alt.Y("rev_mean:Q", title="₹ mean revenue"),
            tooltip=["coach","rev_mean","rev_p10","rev_p90","load_mean"]
        ).properties(height=280), use_container_width=True)
    st.session_state["exports"]["coach_mc_revenue.csv"] = mc

# 5) Seatmaps (True & Tools)
with tabs[4]:
    st.markdown("<div class='section-title'>Seatmaps — True Berths, Women-only, Swap/Squeeze</div>", unsafe_allow_html=True)
    inv_json = fetch_prs_inventory(train_number, run_date, prs_base, api_key)
    seatmap_init = generate_train_seatmaps_true(classes_selected, coach_counts, women_ratio=0.1, prs_inventory=inv_json)
    if "seatmap_work" not in st.session_state: st.session_state["seatmap_work"] = seatmap_init.copy()
    work = st.session_state["seatmap_work"]
    coaches = sorted(work["coach"].unique().tolist()) if not work.empty else []
    if not coaches: st.info("No coaches configured — increase coach counts in the sidebar.")
    else:
        sel_coach = st.selectbox("Select coach", coaches)
        view_df = work[work["coach"]==sel_coach].copy()
        bays = sorted(view_df["bay"].unique().tolist())
        for b in bays[:10]:
            bay_df = view_df[view_df["bay"]==b].sort_values("pos")
            line = " | ".join([f"{r['berth_type']}{'*' if r['status']=='CONF' else ''}{'(W)' if r['women_only'] and r['pos']==1 else ''}" for _,r in bay_df.iterrows()])
            st.write(f"Bay {b:02d}: {line}")
        st.markdown("**Actions**")
        act = st.radio("Action", ["Swap two seats", "Squeeze RAC→CONF"], horizontal=True)
        if act == "Swap two seats":
            c1,c2 = st.columns(2)
            with c1:
                coach_a = st.selectbox("Coach A", coaches, key="coach_a_v9")
                seat_a = st.number_input("Seat A #", 1, 500, 1, 1)
            with c2:
                coach_b = st.selectbox("Coach B", coaches, key="coach_b_v9")
                seat_b = st.number_input("Seat B #", 1, 500, 2, 1)
            if st.button("🔁 Swap now"):
                ok, msg = swap_seats(work, coach_a, int(seat_a), coach_b, int(seat_b))
                if ok:
                    st.success("Seats swapped.")
                else:
                    st.error(str(msg))
        else:
            target_od = st.selectbox("Confirm RAC for OD", [r["OD"] for _, r in od_df.iterrows()])
            if st.button("✅ Squeeze RAC→CONF now"):
                cnt, assigned = squeeze_confirm_from_rac(work, target_od)
                st.success(f"Confirmed {cnt} RAC passengers.")
                if cnt>0: st.write("Assigned:", ", ".join([f"{c}-{s}" for c,s in assigned]))
        # Export/import
        def export_zip(df):
            buf = io.BytesIO(); zf = zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED)
            zf.writestr("seatmap.csv", df.to_csv(index=False))
            for coach_id, dfc in df.groupby("coach"): zf.writestr(f"{coach_id}.csv", dfc.to_csv(index=False))
            zf.close(); return buf.getvalue()
        st.download_button("⬇️ Download seatmaps.zip", data=export_zip(work), file_name=f"seatmaps_{train_number}.zip", mime="application/zip")
        up_zip = st.file_uploader("Import seatmap.csv or seatmaps.zip", type=["csv","zip"], key="seatzip_v9")
        if up_zip is not None:
            try:
                if up_zip.name.endswith(".csv"): st.session_state["seatmap_work"] = pd.read_csv(up_zip)
                else:
                    zf = zipfile.ZipFile(up_zip)
                    if "seatmap.csv" in zf.namelist():
                        st.session_state["seatmap_work"] = pd.read_csv(zf.open("seatmap.csv"))
                    else:
                        frames = [pd.read_csv(zf.open(n)) for n in zf.namelist() if n.endswith(".csv")]
                        if frames: st.session_state["seatmap_work"] = pd.concat(frames, ignore_index=True)
                st.success("Seatmaps imported.")
            except Exception as e: st.error(f"Import failed: {e}")

# 6) Family Seating & Conflicts
with tabs[5]:
    st.markdown("<div class='section-title'>Family Seating — Constraints & Severity</div>", unsafe_allow_html=True)
    protect_children = st.checkbox("Protect children from UB/SU", True)
    protect_elderly  = st.checkbox("Protect elderly (>60) from UB/SU", True)
    women_only_booking = st.checkbox("Women-only booking mode (strict)", False)
    sel_class = st.selectbox("Class to allocate", classes_selected, key="fam_cls_v9")
    inv_json = fetch_prs_inventory(train_number, run_date, prs_base, api_key)
    src_map = generate_train_seatmaps_true([sel_class], {sel_class: coach_counts.get(sel_class,0)}, women_ratio=0.1, prs_inventory=inv_json)
    fam_count = st.slider("Number of families", 2, 50, 20, 1)
    if st.button("Run family allocation now"):
        flags = {"protect_children": protect_children, "protect_elderly": protect_elderly, "women_only_booking": women_only_booking}
        out_map, log, conf_df, seat_conf = family_seating_optimizer(src_map, sel_class, n_fams=fam_count, flags=flags)
        if not conf_df.empty: conf_df["severity"] = conf_df.apply(severity_score, axis=1)
        st.dataframe(out_map.head(60), use_container_width=True)
        st.text("\n".join(log[:25]) + ("\n... (truncated)" if len(log)>25 else ""))
        st.markdown("**Aggregate conflicts (coach/bay)**")
        st.dataframe(conf_df.sort_values("severity", ascending=False), use_container_width=True)
        if not conf_df.empty:
            agg = conf_df.groupby(["coach","bay"]).sum(numeric_only=True).reset_index()
            melt = agg.melt(id_vars=["coach","bay"], var_name="violation", value_name="count")
            heat = alt.Chart(melt).mark_rect().encode(
                x=alt.X("coach:N", sort=None), y=alt.Y("bay:O"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="orangered"), title="Violations"),
                tooltip=["coach","bay","violation","count"]
            ).properties(height=300)
            st.altair_chart(heat, use_container_width=True)
        st.markdown("**Seat-level conflicts (drill-down)**")
        st.dataframe(seat_conf.sort_values(["coach","bay","seat"]), use_container_width=True)
        st.session_state["latest_alloc"] = out_map
        st.session_state["seat_conf"] = seat_conf
        st.session_state["seatmap_latest"] = src_map

# 7) Rules (YAML)
with tabs[6]:
    st.markdown("<div class='section-title'>Rules (YAML) — Effective</div>", unsafe_allow_html=True)
    st.json(rules)
    st.code(safe_yaml_dump(DEFAULT_RULES), language="yaml")

# 8) DB Sync
with tabs[7]:
    st.markdown("<div class='section-title'>PostgreSQL Sync</div>", unsafe_allow_html=True)
    if not USE_DB: st.info("Set DATABASE_URL to enable DB persistence.")
    else:
        # Save rules
        if st.button("Save rules to DB"):
            try: vid = save_rules_version(engine, train_number, json.dumps(rules)); st.success(f"Rules saved (version {vid}).")
            except Exception as e: st.error(str(e))
        if st.button("Load latest rules from DB"):
            try:
                rj = load_latest_rules_json(engine, train_number)
                if rj: st.json(json.loads(rj))
                else: st.warning("No rules found.")
            except Exception as e: st.error(str(e))
        # Save seatmap version (from latest seatmap tab 5/6)
        latest_map = st.session_state.get("seatmap_work")
        if latest_map is None or (hasattr(latest_map, "empty") and latest_map.empty):
            latest_map = st.session_state.get("seatmap_latest")

        if st.button("Save seatmap version"):
            if latest_map is None or (hasattr(latest_map, "empty") and latest_map.empty):
                st.warning("No seatmap available.")
            else:
                try:
                    v_id = save_seatmap_version(engine, train_number, latest_map)
                    st.success(f"Seatmap version saved: {v_id}")
                    st.session_state["seatmap_version_id"] = v_id
                except Exception as e:
                    st.error(str(e))
        # Save family allocations linked to version
        alloc_source = st.session_state.get("latest_alloc"); seat_conf = st.session_state.get("seat_conf", pd.DataFrame())
        if st.button("Persist family allocations (link to seatmap version)"):
            if alloc_source is None or (hasattr(alloc_source, "empty") and alloc_source.empty):
                st.warning("Run family allocation first.")
            else:
                try:
                    v_id = st.session_state.get("seatmap_version_id")
                    if v_id is None: st.warning("Save a seatmap version first.")
                    else:
                        fam_rows = alloc_source[alloc_source["pnr"].fillna("").str.startswith("FAM-")]
                        rows = []
                        for _, r in fam_rows.iterrows():
                            rows.append({
                              "train_number": train_number, "run_date": str(run_date),
                              "coach": r["coach"], "bay": int(r["bay"]), "seat": int(r["seat"]),
                              "class": r["coach"].split("-")[0], "family_id": int(str(r["pnr"]).split("-")[-1])
                            })
                        alloc_df = pd.DataFrame(rows)
                        if not seat_conf.empty:
                            alloc_df = alloc_df.merge(seat_conf[["coach","bay","seat","child_upper","elder_upper","women_only_violation","mixed_gender"]], on=["coach","bay","seat"], how="left").fillna(0)
                        vid = save_family_allocations(engine, alloc_df, seatmap_version_id=v_id)
                        st.success(f"Saved {len(alloc_df)} rows under version {vid} (linked to seatmap_version_id={v_id}).")
                except Exception as e: st.error(str(e))

# 9) REST Hooks
with tabs[8]:
    st.markdown("<div class='section-title'>REST Hooks — Optimizer & PRS</div>", unsafe_allow_html=True)
    st.caption("Calls FastAPI optimizer (/emsr, /multileg) and reads PRS endpoints if available.")
    cls = st.selectbox("Class for optimizer call", classes_selected, key="opt_cls_v9")
    # Demand payload from forecast or synthetic
    exp_dict2 = st.session_state.get("exports", {})
demand_df = exp_dict2.get(f"{cls}_demand.csv")
if demand_df is None or not isinstance(demand_df, pd.DataFrame) or demand_df.empty:
    demand_df = baseline_forecast(od_df, cls, base_demand_index, seasonality, competitor_pressure, rules)
    fares = sorted([round(BASE_FARES[cls]*m) for m in FARE_BUCKETS[cls]], reverse=True)
    bucket_means = [max(1.0, demand_df["mu"].sum()/len(fares)) for _ in fares]
    bucket_sds   = [max(0.5, np.sqrt((demand_df["sd"]**2).sum())/len(fares)) for _ in fares]
    cap = int(SEATS_PER_COACH.get(cls,60)*coach_counts.get(cls,0)*(1+overbooking_pct/100.0)*(1-no_show_rate/100.0))
    payload_emsr = {"fares": fares, "bucket_means": bucket_means, "bucket_sds": bucket_sds, "cap": cap}
    payload_multi = {"od_table": od_df.to_dict("records"),
                     "demand": demand_df.rename(columns={"OD":"OD","mu":"mu","sd":"sd"}).to_dict("records"),
                     "cap": cap, "rule": "Longest first"}
    st.code(json.dumps({"emsr":payload_emsr,"multileg":payload_multi})[:800]+" ...", language="json")
    if st.button("Call optimizer now"):
        try:
            import requests
            r1 = requests.post(f"{optimizer_base}/emsr", json=payload_emsr, timeout=10); r1.raise_for_status()
            r2 = requests.post(f"{optimizer_base}/multileg", json=payload_multi, timeout=10); r2.raise_for_status()
            st.success("Optimizer responded."); st.json({"emsr": r1.json(), "multileg": r2.json()})
        except Exception as e: st.error(f"Optimizer call failed: {e}")

# 10) Exports
with tabs[9]:
    st.markdown("<div class='section-title'>Exports — CSV & PDF</div>", unsafe_allow_html=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, frame in (st.session_state.get("exports") or {}).items():
            if isinstance(frame, pd.DataFrame): zf.writestr(name, frame.to_csv(index=False))
    st.download_button("⬇️ Download ZIP (CSVs)", data=buf.getvalue(), file_name=f"irm_export_{train_number}.zip", mime="application/zip")
    st.markdown("#### Generate PDF (reportlab required)")
    want_pdf = st.checkbox("Create PDF report now", key="v9pdf")
    if want_pdf:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import cm
            pdf_path = f"irm_report_{train_number}.pdf"
            c = canvas.Canvas(pdf_path, pagesize=A4); W,H=A4
            def line(y,text,size=12,color=(0,0,0)):
                c.setFont("Helvetica-Bold", size); c.setFillColorRGB(*color); c.drawString(2*cm,y,text)
            y=H-2*cm; line(y, f"IRCTC IRM Report — Train {train_number}", 16, (0.39,0.35,1.0)); y-=1.0*cm
            line(y, f"Date: {run_date}   Season: {seasonality}", 11, (0.39,0.35,1.0)); y-=0.8*cm
            exp = st.session_state.get("exports") or {}
            for key in sorted(exp.keys()): y-=0.6*cm; line(y, key.replace(".csv",""), 11, (0.13,0.77,0.37)); 
            c.showPage(); c.save()
            with open(pdf_path, "rb") as f:
                st.download_button("📄 Download PDF report", f, file_name=pdf_path, mime="application/pdf")
            st.success("PDF generated.")
        except Exception as e: st.error(f"ReportLab not installed. Install with: pip install reportlab • Error: {e}")

st.caption("Unified Super App v9 — plug PRS/UTS & optimizer services for production. Replace stubs with real endpoints.")
