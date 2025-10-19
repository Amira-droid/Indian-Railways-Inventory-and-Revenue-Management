
# optimizer_api.py
# FastAPI microservice for optimizer endpoints
# --------------------------------------------
# Run:
#   uvicorn optimizer_api:app --reload --port 8010
#
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="IRCTC IRM Optimizer API", version="1.0")

class EMSRRequest(BaseModel):
    fares: List[float]
    bucket_means: List[float]
    bucket_sds: List[float]
    cap: int

class EMSRResponse(BaseModel):
    protection: List[float]
    booking_limits: List[float]

class ODE(BaseModel):
    OD: str
    O: str
    D: str
    legs: int
    distance: float

class DEM(BaseModel):
    OD: str
    mu: float
    sd: float

class MultiLegRequest(BaseModel):
    od_table: List[ODE]
    demand: List[DEM]
    cap: int
    rule: str = "Longest first"

class MultiLegResponse(BaseModel):
    allocations: List[Dict[str, Any]]

def _norm_ppf(p: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        x = p - 0.5
        return np.sqrt(2) * np.sign(x) * np.sqrt(np.log(1/(1 - 2*abs(x))))

@app.post("/emsr", response_model=EMSRResponse)
def emsr(req: EMSRRequest):
    fares = np.array(sorted(req.fares, reverse=True), dtype=float)
    means = np.array(req.bucket_means, dtype=float)
    sds = np.array(req.bucket_sds, dtype=float)
    cum_mean = np.cumsum(means)
    cum_sd = np.sqrt(np.cumsum(sds**2))
    prot = np.zeros_like(fares)
    for k in range(len(fares)-1):
        fk, fk1 = fares[k], fares[k+1]
        crit = fk1/fk
        z = _norm_ppf(1-crit)
        y = cum_mean[k] + cum_sd[k]*z
        prot[k] = max(0.0, min(req.cap, y))
    booking_limits = np.diff(np.append(np.maximum.accumulate(prot), req.cap))
    return {"protection": prot.tolist(), "booking_limits": booking_limits.tolist()}

@app.post("/multileg", response_model=MultiLegResponse)
def multileg(req: MultiLegRequest):
    od_df = [{"OD":o.OD, "O":o.O, "D":o.D, "legs":o.legs, "distance":o.distance} for o in req.od_table]
    dem = {d.OD: d.mu for d in req.demand}
    # naive equal per-segment cap
    stations = sorted({r["O"] for r in od_df} | {r["D"] for r in od_df}, key=lambda x: list({s:i for i,s in enumerate(sorted({r['O'] for r in od_df} | {r['D'] for r in od_df}))}).index(x))
    leg_index = [(stations[i], stations[i+1]) for i in range(len(stations)-1)]
    seg_cap_each = max(1, int(req.cap/len(leg_index)))
    cap = {seg: seg_cap_each for seg in leg_index}
    # order
    if req.rule == "Longest first":
        od_df.sort(key=lambda r: (r["legs"], dem.get(r["OD"],0)), reverse=True)
    elif req.rule == "Shortest first":
        od_df.sort(key=lambda r: (r["legs"], dem.get(r["OD"],0)))
    else:
        od_df.sort(key=lambda r: (dem.get(r["OD"],0)), reverse=True)

    def od_to_legs(o,d):
        i = stations.index(o); j = stations.index(d)
        return [(stations[k], stations[k+1]) for k in range(i,j)]

    out = []
    for r in od_df:
        legs = od_to_legs(r["O"], r["D"])
        max_possible = min(cap.get(seg,0) for seg in legs)
        give = min(int(round(dem.get(r["OD"],0))), max_possible)
        if give>0:
            for seg in legs: cap[seg] -= give
        out.append({"OD":r["OD"], "alloc":give, "demand":dem.get(r["OD"],0)})
    return {"allocations": out}

@app.get("/health")
def health():
    return {"ok": True}
