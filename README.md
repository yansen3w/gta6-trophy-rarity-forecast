# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansen3w/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yansen3w/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the percent of the players Exophase tracks who will have earned it at 2026-11-26T00:00:00Z, a week after launch — as five quantiles per trophy, scored by pinball loss on the logit scale.

The model splits each trophy's earn rate into earners over players tracked and fits both counts in log-log space against time since release, recency-weighted so the recent drift dominates: when the tracked-player count grows faster than a trophy's earners, its rate falls (dilution — exactly what the launch-night refresh did), and when players progress faster than new players arrive, it rises. Slopes are shrunk toward their tier's median, and the spread combines fit residuals, slope uncertainty over the extrapolation, and a random-walk term over the horizon.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`python3 main.py` reads every `<id>.in` case in the input directory (`$MAKEFUN_INPUT_DIR`, else `/input`) and writes `<id>.out` forecasts to the output directory (`$MAKEFUN_OUTPUT_DIR`, else `/output`). Python 3.13, standard library only, no network.

---

Entered on [make.fun](https://make.fun/?ref=30afxf5j&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
