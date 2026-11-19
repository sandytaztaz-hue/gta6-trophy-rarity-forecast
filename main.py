"""GTA 6 trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of the case, five quantiles (5/25/50/75/95%) of
the earn rate Exophase will show at the case's target instant.

Design (all arithmetic is in logit space, because the scorer is a pinball
loss on logits and a miss from 1% to 2% costs about as much as 33% to 50%):

  * History is piecewise constant: Exophase refreshes hours apart and the
    page is unchanged between refreshes.  Consecutive reads with identical
    players-tracked and identical rates are collapsed to one snapshot
    before any fitting, or the flat stretches would overweight the fit.
  * Counts, not rounded rates, are the modelled quantity.  A trophy's
    earned count is recovered exactly as round(rate / 100 * N), and the
    rate is count / N.  Counts never fall and never exceed N; rates can
    fall when dilution (new tracked players holding few trophies) outruns
    progression.  Every fitted median is checked against that guardrail.
  * Trend branch (two or more distinct snapshots): per trophy, a
    least-squares line of logit(rate) on ln(1 + hours since launch) over
    the last 24 h of distinct snapshots, read at the target and damped
    (w = 0.7) toward the last observed rate so a 4.5-day extrapolation
    cannot overshoot.  The tracked-players denominator N(target) comes
    from the same log-time fit on log N, slope clamped >= 0.
  * Flat branch (fewer than two distinct snapshots, the launch-night
    shape): no trend is estimable, so the median assumes the next refresh
    dilutes like the first observed one did (638 -> 3,868 players, a
    x6.06 growth, new cohort ~1.0 trophies/player): median =
    old_count / N_first_refresh.  The new cohort's leftover earns
    concentrate in the earliest trophies, so ids 3-7 get a boost from the
    ~103 leftover earns (shares 40/25/15/10/10).  Near-100% trophies stay
    near 100%.  Spreads here are asymmetric: tight below the median
    (s_down = 0.12), wider above it (s_up = 0.5), because a refresh puts
    the truth at the diluted median while no refresh puts it far above,
    and pinball punishes under-forecasting at high quantiles hardest.
  * Trend-branch spreads are symmetric: the fitted trend puts the truth
    roughly symmetric around the median.  The random-walk floor
    (0.5 * sqrt(days)) is scaled per trophy by its slope relative to the
    median slope, so near-static trophies stop paying the full diffusion
    floor; fit residuals and one inter-refresh gap of expected move are
    added in quadrature.

Standard library only, no network, deterministic: pure functions of the
case file, trophies iterated in id order, fixed float formatting.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# First observed Exophase refresh (launch night): 638 -> 3,868 players
# tracked, 943 -> 4,276 trophies earned.  The new cohort held ~1.03
# trophies/player, so the next refresh dilutes every rate by ~6.06x.
FIRST_N0 = 638
FIRST_N1 = 3868
FIRST_E0 = 943
FIRST_E1 = 4276
DILUTION = FIRST_N1 / FIRST_N0
# Leftover earns of the first refresh's new cohort, after one trophy
# each: (E1 - E0) - (N1 - N0) = 3333 - 3230 = 103.  They concentrate in
# the earliest trophies (story order), ids 3-7, in these shares.
LEFTOVER_SHARES = {3: 0.40, 4: 0.25, 5: 0.15, 6: 0.10, 7: 0.10}

DAMPING = 0.7            # trend-branch extrapolation damping toward last
S_DOWN_FLAT = 0.12       # flat branch: logit spread below the median
S_UP_FLAT = 0.5          # flat branch: logit spread above the median
RW_SIGMA = 0.5           # house random-walk floor, logit units per sqrt day
NEAR_FULL = 99.5         # a rate at/above this stays near 100%


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_launch(moment):
    return (moment - LAUNCH).total_seconds() / 3600.0


def x_of(moment):
    """Log-time coordinate used by every fit."""
    return math.log1p(max(0.0, hours_since_launch(moment)))


def least_squares(xs, ys):
    """Closed-form line fit; returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def dedupe(history):
    """Collapse consecutive identical reads (same N and same rates)."""
    snaps = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, read))
    return [read for _, read in snaps]


def last_known(series):
    """Last non-null value of a per-snapshot series, else None."""
    for value in reversed(series):
        if value is not None:
            return value
    return None


def forecast_case(case):
    trophies = sorted(case["trophies"], key=lambda t: t["id"])
    n_troph = len(trophies)
    history = dedupe(case.get("history") or [])
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = x_of(target)

    # Per-trophy rate series over distinct snapshots (None = missing read).
    series = []
    for i in range(n_troph):
        series.append([read["rates"][i] if i < len(read["rates"]) else None
                       for read in history])
    last_rates = [last_known(s) for s in series]

    # Fallback for a trophy never observed: the tier median of last rates.
    tier_last = {}
    for trophy, rate in zip(trophies, last_rates):
        if rate is not None:
            tier_last.setdefault(trophy["tier"], []).append(rate)
    tier_median = {tier: median(vals) for tier, vals in tier_last.items()}
    global_median = median([r for r in last_rates if r is not None]) \
        if any(r is not None for r in last_rates) else 1.0
    for i, trophy in enumerate(trophies):
        if last_rates[i] is None:
            last_rates[i] = tier_median.get(trophy["tier"], global_median)

    # --- Forecast the tracked-players denominator N(target). ---
    n_last = history[-1]["players_tracked"] if history else FIRST_N0
    n_target = float(n_last)
    if len(history) >= 2:
        xs = [x_of(parse_time(r["at"])) for r in history]
        ys = [math.log(max(1, r["players_tracked"])) for r in history]
        _, slope_n = least_squares(xs, ys)
        slope_n = max(0.0, slope_n)
        n_target = max(float(n_last), math.exp(
            least_squares(xs, ys)[0] + slope_n * x_target))

    # --- Per-trophy trend fits (last 24 h of distinct snapshots). ---
    window_start = at.timestamp() - 24 * 3600
    fits = []  # (intercept, slope, residuals, last_logit, last_count)
    for i in range(n_troph):
        pts = []
        for read, rate in zip(history, series[i]):
            if rate is None:
                continue
            moment = parse_time(read["at"])
            if moment.timestamp() < window_start:
                continue
            pts.append((x_of(moment), logit(rate),
                        read["players_tracked"]))
        if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            intercept, slope = least_squares(xs, ys)
            resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        else:
            intercept, slope, resid = logit(last_rates[i]), 0.0, []
        last_count = round(last_rates[i] / 100.0 * n_last)
        fits.append((intercept, slope, resid, logit(last_rates[i]),
                     last_count))

    # Pooled residual spread for trophies with too few points to judge.
    all_resid = [r for _, _, resid, _, _ in fits for r in resid]
    pooled_sd = 0.0
    if len(all_resid) >= 2:
        mean_r = sum(all_resid) / len(all_resid)
        pooled_sd = math.sqrt(
            sum((r - mean_r) ** 2 for r in all_resid) / len(all_resid))

    # Median |slope| for the adaptive random-walk floor.
    slopes = [abs(slope) for _, slope, _, _, _ in fits]
    med_slope = median(slopes) if slopes else 0.0

    # Tier ceilings: the highest rate observed so far among other trophies
    # of the same tier (platinum/gold saturate low, early bronze high).
    tier_ceiling = {}
    for trophy in trophies:
        others = [last_rates[j] for j, other in enumerate(trophies)
                  if other["tier"] == trophy["tier"]
                  and other["id"] != trophy["id"]]
        if others:
            tier_ceiling[trophy["id"]] = max(others)

    # Typical gap between distinct snapshots, in hours (refresh cadence).
    gap_hours = 0.0
    if len(history) >= 2:
        gaps = []
        for a, b in zip(history, history[1:]):
            gaps.append((parse_time(b["at"]) - parse_time(a["at"]))
                        .total_seconds() / 3600.0)
        gap_hours = median(gaps)

    flat = len(history) < 2
    rows = []
    for i, trophy in enumerate(trophies):
        intercept, slope, resid, lg_last, c_last = fits[i]
        last_rate = last_rates[i]

        if flat:
            # ---- Flat branch: dilution prior, no trend estimable. ----
            if last_rate >= NEAR_FULL:
                # Every tracked player holds it and every new player
                # earns it within minutes: the truth stays at the top of
                # the scorer's clamp, so sit there too (99.9 cost ~0.03
                # of mean loss on the refresh scenarios in testing).
                med = 99.99
            else:
                n_first = n_last * DILUTION
                med = c_last / n_first * 100.0
                boost = LEFTOVER_SHARES.get(trophy["id"])
                if boost is not None:
                    leftover = (FIRST_E1 - FIRST_E0) - (FIRST_N1 - FIRST_N0)
                    med = (c_last + leftover * boost) / n_first * 100.0
            med = clamp_pct(med)
            lg_med = logit(med)
            qs = []
            for z in Z:
                spread = S_DOWN_FLAT if z < 0 else (S_UP_FLAT if z > 0 else 0.0)
                qs.append(clamp_pct(inv_logit(lg_med + z * spread)))
        else:
            # ---- Trend branch: damped log-time extrapolation. ----
            lg_extrap = intercept + slope * x_target
            lg_med = lg_last + (lg_extrap - lg_last) * DAMPING
            med = clamp_pct(inv_logit(lg_med))
            # Count-decomposition guardrail: counts never fall, never
            # exceed the tracked players at the target.
            c_target = med / 100.0 * n_target
            c_target = min(max(c_target, float(c_last)), n_target)
            med = clamp_pct(c_target / n_target * 100.0)
            # Tier ceiling (only where other trophies define the tier).
            ceiling = tier_ceiling.get(trophy["id"])
            if ceiling is not None:
                med = min(med, max(ceiling, last_rate))
            med = clamp_pct(med)
            lg_med = logit(med)
            # Symmetric spread: adaptive random-walk floor + fit
            # residuals + one refresh gap of expected move at this slope.
            ratio = abs(slope) / med_slope if med_slope > 0 else 0.2
            ratio = min(1.0, max(0.2, ratio))
            floor = RW_SIGMA * math.sqrt(horizon_days) * ratio
            sd_resid = pooled_sd
            if len(resid) >= 3:
                mean_r = sum(resid) / len(resid)
                sd_resid = math.sqrt(
                    sum((r - mean_r) ** 2 for r in resid) / len(resid))
            h_target = max(0.0, hours_since_launch(target))
            s_refresh = abs(slope) * gap_hours / (1.0 + h_target)
            spread = math.sqrt(floor * floor + sd_resid ** 2
                               + s_refresh ** 2)
            qs = [clamp_pct(inv_logit(lg_med + z * spread)) for z in Z]

        qs = [round4(q) for q in qs]
        # Enforce non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
