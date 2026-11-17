"""GTA VI Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Forecasts hours 2..24 of a launch day from hour 1's print, using the shape
of past big launches' release days, fitted at run time from the case's own
`comparables` (no hard-coded launch, no network, stdlib only).

Method
------
For every comparable launch with a usable release-day hour 1, take the
log-ratio of each later hour's viewers to hour 1's viewers:

    r_h = ln(v_h / v_1),  h = 2..24

Hour 1's size is the key signal. Across past launches, a huge hour-1 print
means the launch is already at full tilt (its day stays flat, peak about
1.0-1.3x hour 1: ELDEN RING 875k -> 0.95x, Cyberpunk 2077 852k -> 1.26x,
Hogwarts Legacy 387k -> 1.30x, The Last of Us Part II 365k -> 1.34x),
while a small hour-1 print means the unlock has not happened yet and the
day rises many-fold (Hollow Knight: Silksong 164 -> 2120x peak, Black
Myth: Wukong 4941 -> 92x, ARC Raiders 1410 -> 260x). So each comparable
is weighted by how close its hour-1 print is to this case's, with a
Gaussian kernel on log hour-1 viewers (bandwidth 1.0 in log units):

    w_c = exp(-(ln v_1,c - ln p1)^2 / 2)

The forecast for hour h is the weighted median m_h of r_h across the pool
and the weighted inter-quartile range, scaled to a normal sigma s_h:

    q_tau(h) = p1 * exp(m_h + z_tau * s_h)

with z_tau the normal quantile at level tau. The spread is floored at half
the unweighted pool's IQR spread, so it cannot collapse when the kernel
concentrates on a few launches (e.g. a record-breaking hour 1 larger than
any comparable). This is the unlock-schedule adjustment in data form:
rolling-midnight launches share GTA VI's schedule (New Zealand first at
11:00Z the day before, UK at 00:00Z, US East 05:00Z, US West 08:00Z), and
the kernel picks the launches whose hour 1 sits at the same point of that
wave as this case's.

Which launches are in the pool: a launch is dropped when its own notes say
it unlocked at one simultaneous global time ("simultaneous", "single
global") instead of rolling midnight local, when it is tiered "weak", or
when it is the target game's own pre-release history ("target-history").
Simultaneous launches have a pre-launch-quiet hour 1 and a spike hours
later, a different shape from a rolling-midnight launch like GTA VI; the
rest (rolling-midnight, other, and early-access-contaminated launches,
whose release-day shape is still informative) form the pool.

If an hour has fewer than 3 launches carrying it, or the case has no
usable comparables at all, that hour falls back to the house persistence
baseline: q_tau(h) = p1 * exp(z_tau * 0.15 * sqrt(h-1)).

Robustness: hour-1 prints below 1 count as 1; viewer counts below 1 count
as 1 inside logs; every quantile is clamped to [1, 1e9], forced
non-decreasing, and rounded half-up to 2 decimals. Any failure while
fitting still writes a valid baseline forecast, never a crash: an invalid
case forfeits prizes, so validity comes first.

Validation (leave-one-out over the practice comparables, each treated as
a pseudo-case with its own hour 1 known and hours 2..24 scored by the
contest pinball loss on the log scale, 2 * max(tau*d, (tau-1)*d),
d = ln y - ln q): mean loss 0.443 vs the persistence baseline's 0.930,
winning 32 of 37 pseudo-cases; on the 12 rolling-midnight launches, the
launches shaped like GTA VI, 0.256 vs the baseline's 0.357, winning 12
of 12. The unweighted median-shape variant scored 0.835 overall, so the
size weighting is what carries the edge; platform-only weighting and
shape smoothing were tested and did not help.

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, stdlib only).
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Bandwidth of the Gaussian kernel on log hour-1 viewers.
TAU = 1.0
# Scale on the IQR-derived log spread.
BETA = 1.0
# Floor on the spread, as a fraction of the unweighted pool's IQR spread.
SPREAD_FLOOR = 0.5
# The persistence baseline's log spread per square-root hour, used as the
# fallback when an hour has too few launches to fit.
SIGMA = 0.15
# Minimum pool size (launches carrying an hour) for a fitted row.
MIN_POOL = 3


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, ~1e-9)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def quantile(sorted_vals, tau: float) -> float:
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def wquantile(pairs, tau: float) -> float:
    """Weighted quantile (type-7 style): each observation sits at the
    midpoint of its cumulative-weight block; linear interpolation between
    neighbouring midpoints, flat extrapolation outside."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return 0.0
    pts = []
    cum = 0.0
    for v, w in pairs:
        pts.append(((cum + w / 2.0) / tot, v))
        cum += w
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if tau <= pts[i][0]:
            p0, v0 = pts[i - 1]
            p1, v1 = pts[i]
            frac = (tau - p0) / (p1 - p0) if p1 > p0 else 0.0
            return v0 * (1 - frac) + v1 * frac
    return pts[-1][1]


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def release_hours(comp: dict) -> dict:
    """Comparable's release-day hour (1..24) -> viewers, keyed by timestamp."""
    day_start = parse_ts(comp.get("day_start"))
    if day_start is None:
        return {}
    out = {}
    hourly = comp.get("hourly") or []
    for pt in hourly:
        if not isinstance(pt, dict):
            continue
        t = parse_ts(pt.get("from"))
        if t is None:
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and h not in out:
            try:
                v = float(pt.get("viewers"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                out[h] = max(1.0, v)
    return out


def pool_include(comp: dict) -> bool:
    """False for launches whose shape does not match a rolling-midnight launch.

    Drops simultaneous-global unlocks (hour 1 is pre-launch quiet, the
    spike comes hours later), weak-tier analogues, and the target game's
    own pre-release history (it has no release day yet). Everything else
    (rolling midnight, other, early-access-contaminated) stays in.
    """
    text = str(comp.get("unlock") or "").lower()
    if "tier: target-history" in text or "tier: weak" in text:
        return False
    if "rolling" in text or "midnight local" in text:
        return True
    if "simultaneous" in text or "single global" in text:
        return False
    return True


def fit_rows(comparables, p1: float, hours):
    """Per-hour (weighted median log-ratio, spread) across the pooled launches.

    Each launch is weighted by a Gaussian kernel on log hour-1 viewers
    around this case's hour-1 print, so launches at the same point of the
    unlock wave dominate the shape.
    """
    log_p1 = math.log(max(1.0, p1))
    pairs = {h: [] for h in hours}
    flat = {h: [] for h in hours}
    for comp in comparables:
        if not pool_include(comp):
            continue
        rh = release_hours(comp)
        if 1 not in rh:
            continue
        w = math.exp(-((math.log(rh[1]) - log_p1) ** 2) / (2 * TAU * TAU))
        v1 = rh[1]
        for h in hours:
            if h in rh:
                r = math.log(rh[h] / v1)
                pairs[h].append((r, w))
                flat[h].append(r)
    rows = {}
    for h in hours:
        if len(pairs[h]) >= MIN_POOL:
            m = wquantile(pairs[h], 0.5)
            s = max(0.05, (wquantile(pairs[h], 0.75) - wquantile(pairs[h], 0.25)) / 1.349)
            fs = max(0.05, (quantile(sorted(flat[h]), 0.75) - quantile(sorted(flat[h]), 0.25)) / 1.349)
            rows[h] = (m, max(s, SPREAD_FLOOR * fs) * BETA)
    return rows


def baseline_row(p1: float, h: int, z) -> list:
    return [p1 * math.exp(zi * SIGMA * math.sqrt(h - 1)) for zi in z]


def clean_row(qs) -> list:
    """Clamp to (0, 1e9], force non-decreasing, round half-up to 2 decimals."""
    out = []
    prev = 1.0
    for q in qs:
        try:
            v = float(q)
        except (TypeError, ValueError):
            v = 1.0
        if not math.isfinite(v) or v <= 0:
            v = 1.0
        v = min(v, 1e9)
        v = max(v, prev)
        r = round2(v)
        out.append(r)
        prev = float(r)
    return out


def forecast_case(case: dict) -> dict:
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    z = [norm_ppf(float(tau)) for tau in levels]
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = []
    for row in rows:
        try:
            hours.append(int(row["hour"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not hours:
        hours = list(range(2, 25))

    p1 = 1.0
    try:
        first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
        if first is not None:
            p1 = max(1.0, float(first["viewers"]))
        else:
            hist = case.get("history_hourly") or []
            if hist:
                p1 = max(1.0, float(hist[-1]["viewers"]))
    except (TypeError, ValueError, KeyError):
        p1 = 1.0

    shape = {}
    try:
        shape = fit_rows(case.get("comparables") or [], p1, hours)
    except Exception:
        shape = {}

    out_hours = []
    for h in hours:
        if h in shape:
            m, s = shape[h]
            qs = [p1 * math.exp(m + zi * s) for zi in z]
        else:
            qs = baseline_row(p1, h, z)
        out_hours.append({"hour": h, "q": clean_row(qs)})
    return {"hours": out_hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never emit an invalid case: fall back to a flat forecast at 1.
            forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
