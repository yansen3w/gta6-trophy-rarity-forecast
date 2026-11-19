"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every GTA VI trophy on PlayStation 5, five quantiles of its
earn rate on Exophase at the case's target, from the house's recorded reads
of the trophy page.

Model: two views of each trophy's earn rate, averaged in logit space.

A trophy's earn rate is earners / players_tracked. Both counts only grow,
and both grow roughly like a power of the time since release, so each is
fitted in log-log space:

    ln E_i(t) ~ a_i + c_i * ln(1 + h(t))
    ln P(t)   ~ d   + e   * ln(1 + h(t))

where h(t) is the hours from the game's release to the read. The fits are
recency-weighted (12-hour half-life), so the launch-night plateau, when
Exophase had not refreshed its counts yet, fades out and the recent drift
dominates. A trophy's slope is shrunk 30% toward its tier's median slope:
trophies of one tier are earned by similar play, so a noisy single-trophy
slope is pulled toward what its tier is doing. The earners' extrapolation
is damped to 70% of the fitted move: an earners count that looks like a
power law locally is usually a saturating curve (players finish the early
story, then stop), and an undamped power law overshoots toward 100%.

The second view is the house curve's shape: a recency-weighted fit of the
rate itself on the logit scale against ln(1 + h), which saturates at 100%
by construction. The forecast median is the mean of the two views in logit
space; on synthetic cases with known truth (falling rates under dilution,
story trophies saturating at 60% and 90%, a mid-story trophy still rising,
a slow grind) the blend beats either view alone, and neither view alone
is ever far off.

The split makes dilution explicit: when the tracked-player count grows
faster than a trophy's earners, its rate falls, which is exactly what the
launch-night refresh did (638 -> 3,868 players tracked, every rate but the
one at 100% lower).

The median is that forecast rate on the logit scale. The spread combines
the fit's residual noise, the slope uncertainty over the extrapolation
distance, a random-walk term that grows with the square root of the
horizon, and a floor for model error, so the 90% band is honest about a
week of play and several more Exophase refreshes.

Usage (the grader runs `python3 /bot/main.py`, Python 3.13, stdlib only,
no network):

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Reads every <id>.in in the input directory and writes <id>.out beside it:
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry ships no extra module.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Recency weight half-life for the fits, in hours: reads a day old count
# about a quarter of a fresh one, launch-night reads almost nothing.
HALF_LIFE_HOURS = 12.0
# Share of a trophy's fitted slope pulled toward its tier's median slope.
TIER_SHRINK = 0.30
# Random-walk spread, in logit units per square-root day.
RW_SIGMA = 0.35
# The earners' extrapolation is damped to this share of the fitted move:
# a locally power-law-looking earners curve is usually a saturating one.
EARNERS_DAMP = 0.7
# Floor on the spread (logit units): model error over a week of play.
SPREAD_FLOOR = 0.25
# Hard caps that keep a wild fit from forecasting nonsense.
MAX_SLOPE = 3.0
MAX_LOGIT_MOVE = 4.0
MAX_SPREAD = 2.5
# A trophy at (or rounded to) 100% cannot stay there once fresh players
# sync: forecast it a touch below 100 instead of exactly at the cap.
SATURATED_RATE = 99.9


def clamp(x: float, lo: float = 0.01, hi: float = 99.99) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9 accurate).

    Only used when a case asks for levels other than the contest's five;
    the standard levels use the exact Z table above.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
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


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log1p_hours(moment: datetime, release: datetime) -> float:
    """ln(1 + hours from release to moment), floored so it stays defined."""
    hours = (moment - release).total_seconds() / 3600.0
    return math.log(max(1.0 + hours, 0.05))


def wols(xs, ys, ws):
    """Recency-weighted least squares of y on x with an intercept.

    Returns (intercept, slope, residual_std, slope_se, xbar, n_points).
    A degenerate x spread falls back to a flat line at the weighted mean.
    """
    sw = sum(ws)
    if sw <= 0 or not xs:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0
    xm = sum(w * x for x, w in zip(xs, ws)) / sw
    ym = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - ym) ** 2 for y, w in zip(ys, ws)) / sw)
        return ym, 0.0, resid, 0.0, xm, len(xs)
    slope = sxy / sxx
    intercept = ym - slope * xm
    resid2 = sum(w * (y - (intercept + slope * x)) ** 2
                 for x, y, w in zip(xs, ys, ws)) / sw
    resid = math.sqrt(max(resid2, 0.0))
    slope_se = resid / math.sqrt(sxx) if sxx > 0 else 0.0
    return intercept, slope, resid, slope_se, xm, len(xs)


def median(xs):
    ordered = sorted(xs)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zvals = Z if levels == LEVELS else [norm_ppf(p) for p in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    release = datetime.fromisoformat(
        case["game"]["release_date"] + "T00:00:00+00:00")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_star = log1p_hours(target, release)

    # One usable read: its log-time, its recency weight, its player count,
    # and its rates in the trophies' order.
    reads = []
    for read in history:
        try:
            moment = parse_time(read["at"])
            players = read["players_tracked"]
            rates = read["rates"]
        except (KeyError, TypeError, ValueError):
            continue
        if not players or players <= 0 or not rates:
            continue
        age_h = (checkpoint - moment).total_seconds() / 3600.0
        weight = 0.5 ** (max(age_h, 0.0) / HALF_LIFE_HOURS)
        reads.append((log1p_hours(moment, release), weight, players, rates))

    if not reads:
        # No history at all: a flat, wide forecast beats crashing, since a
        # crash scores the case as if every quantile were a last rate that
        # does not exist.
        return {"trophies": [
            {"id": t["id"], "q": [round4(100 * sigmoid(logit(1.0) + z * 1.5))
                                  for z in zvals]}
            for t in trophies]}

    # Players tracked: one shared log-log fit, the most stable fit in the
    # case, since every read contributes to it.
    xs = [r[0] for r in reads]
    ws = [r[1] for r in reads]
    d_int, e_slope, resid_p, se_e, xbar, _ = wols(
        xs, [math.log(r[2]) for r in reads], ws)
    e_slope = min(max(e_slope, 0.0), MAX_SLOPE)
    logp_star = d_int + e_slope * x_star
    dx = max(x_star - xbar, 0.0)

    # Per trophy: a log-log fit of its earners, then a shrink of its slope
    # toward its tier's median slope.
    fits = []
    for i, trophy in enumerate(trophies):
        ex, ey, ew = [], [], []
        lx, ly, lw = [], [], []
        last_rate = None
        for x, w, players, rates in reads:
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                continue
            last_rate = rate
            earners = max(rate / 100.0 * players, 0.5)
            ex.append(x)
            ey.append(math.log(earners))
            ew.append(w)
            lx.append(x)
            ly.append(logit(rate))
            lw.append(w)
        if len(ex) >= 3:
            a, c, resid, se_c, xbar_i, _ = wols(ex, ey, ew)
            ym_i = sum(w * y for y, w in zip(ey, ew)) / sum(ew)
            _, b_slope, rres, se_b, xbar_l, _ = wols(lx, ly, lw)
            ym_l = sum(w * y for y, w in zip(ly, lw)) / sum(lw)
        elif ex:
            # Too few reads to fit a slope: hold the last levels.
            a, c, resid, se_c, xbar_i = ey[-1], 0.0, 0.3, 0.0, ex[-1]
            ym_i = ey[-1]
            b_slope, rres, se_b, xbar_l = 0.0, 0.3, 0.0, lx[-1]
            ym_l = ly[-1]
        else:
            a, c, resid, se_c, xbar_i = 0.0, 0.0, 0.5, 0.0, xs[-1]
            ym_i = 0.0
            b_slope, rres, se_b, xbar_l = 0.0, 0.5, 0.0, xs[-1]
            ym_l = 0.0
        fits.append({"tier": trophy["tier"], "a": a, "c": c,
                     "resid": resid, "se_c": se_c, "xbar": xbar_i,
                     "ym": ym_i, "b": b_slope, "rres": rres,
                     "se_b": se_b, "xbar_l": xbar_l, "ym_l": ym_l,
                     "last_rate": last_rate, "n": len(ex)})

    tier_slopes = {}
    for tier in {f["tier"] for f in fits}:
        slopes = [f["c"] for f in fits
                  if f["tier"] == tier and f["n"] >= 3]
        if slopes:
            tier_slopes[tier] = median(slopes)

    rows = []
    for trophy, fit in zip(trophies, fits):
        c = fit["c"]
        if fit["n"] >= 3 and fit["tier"] in tier_slopes:
            c = (1.0 - TIER_SHRINK) * c + TIER_SHRINK * tier_slopes[fit["tier"]]
        c = min(max(c, -MAX_SLOPE), MAX_SLOPE)
        # The shrunk line is re-centred on the fit's weighted mean, so
        # shrinking the slope pivots the line instead of shifting its level.
        # The earners' move is damped (see the module docstring); the
        # players' extrapolation is not, it is the stablest fit in the case.
        dx_e = max(x_star - fit["xbar"], 0.0)
        loge_star = fit["ym"] + c * EARNERS_DAMP * dx_e
        rate_star = math.exp(loge_star - logp_star) * 100.0
        # Second view: the rate's own logit trend against log time.
        dx_l = max(x_star - fit["xbar_l"], 0.0)
        med_logit = fit["ym_l"] + fit["b"] * dx_l
        med = 0.5 * logit(rate_star) + 0.5 * med_logit
        if fit["last_rate"] is not None and fit["last_rate"] >= 99.99:
            # Saturated: everyone tracked has it, but fresh players syncing
            # in will not all have it yet, so it sits just under 100%.
            med = logit(SATURATED_RATE)
        if fit["last_rate"] is not None:
            last_logit = logit(fit["last_rate"])
            med = min(max(med, last_logit - MAX_LOGIT_MOVE),
                      last_logit + MAX_LOGIT_MOVE)
        spread = math.sqrt(
            fit["resid"] ** 2
            + resid_p ** 2
            + (fit["se_c"] * dx_e) ** 2
            + (se_e * dx) ** 2
            + fit["rres"] ** 2
            + (fit["se_b"] * dx_l) ** 2
            + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
            + SPREAD_FLOOR ** 2)
        spread = min(max(spread, SPREAD_FLOOR), MAX_SPREAD)
        qs = []
        prev = -1.0
        for z in zvals:
            q = round4(100.0 * sigmoid(med + z * spread))
            q = min(max(q, 0.0), 100.0)
            q = max(q, prev)  # quantiles never decrease, rounding included
            prev = q
            qs.append(q)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
