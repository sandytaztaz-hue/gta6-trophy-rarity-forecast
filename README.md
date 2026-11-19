# GTA 6 Trophy Rarity Forecaster

Forecasts, for every trophy in a case, five quantiles (5/25/50/75/95%) of
the earn rate Exophase will show at the case's target instant. Entry for
make.fun `gta6-trophy-rarity-forecast-01`.

## How to run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case.
Python 3.13, standard library only, no network, deterministic
(byte-identical output across runs).

## Design

All arithmetic is in logit space, because the scorer is a pinball loss on
logits: a miss from 1% to 2% costs about as much as 33% to 50%.

1. **Dedupe.** Exophase refreshes hours apart and the page is unchanged
   between refreshes, so consecutive reads with identical players-tracked
   and identical rates collapse to one snapshot before any fitting.
2. **Counts, not rounded rates.** A trophy's earned count is recovered as
   `round(rate / 100 * N)`. Counts never fall and never exceed N; rates
   can fall when dilution outruns progression. Every fitted median is
   checked against that guardrail.
3. **Trend branch** (2+ distinct snapshots): per-trophy least-squares fit
   of `logit(rate)` on `ln(1 + hours since launch)` over the last 24 h,
   read at the target and damped (w = 0.7) toward the last observed rate.
   `N(target)` comes from the same log-time fit on `log N`, slope
   clamped >= 0. Spreads are symmetric: an adaptive random-walk floor
   (scaled per trophy by its slope relative to the median slope) plus fit
   residuals plus one refresh gap of expected move.
4. **Flat branch** (fewer than 2 distinct snapshots, the launch-night
   shape): no trend is estimable, so the median assumes the next refresh
   dilutes like the first observed one (638 -> 3,868 players, x6.06, new
   cohort ~1.0 trophies/player): `median = old_count / N_first_refresh`.
   The new cohort's ~103 leftover earns concentrate in the earliest
   trophies, so ids 3-7 get a boost (shares 40/25/15/10/10). Near-100%
   trophies stay near 100%. Spreads are asymmetric — tight below the
   median (0.12), wider above it (0.5) — because a refresh puts the truth
   at the diluted median while no refresh puts it far above, and pinball
   punishes under-forecasting at high quantiles hardest.
5. **Guardrails.** Tier ceilings cap extrapolated rates at the highest
   rate observed among other trophies of the same tier; implied counts
   are clamped to `[last count, N(target)]`.

## Verification (local)

Format/validity + byte-identical determinism on the kit practice cases
(p00, p01); starter parity (outputs valid and different from the house
persistence baseline); synthetic refresh stress tests where the forecast
beats the starter when a refresh lands and is not worse when none does;
a synthetic final-like case (long history, 6,510-minute horizon) for
runtime and sanity. See the prep PLAN for the numbers.
