# ERA5 feature choice for hourly rainfall

## Recommended predictor set

The selected fields are available from the ERA5 hourly single-level catalogue and are listed in `configs/default.yaml` / `app/data/era5_download.py`.

| Internal feature | ERA5 field | Why it is retained |
| --- | --- | --- |
| `rainfall_1h` and lags/rolling summaries | Total precipitation (`tp`) | Recent rainfall persistence and storm development; only values ending at or before prediction time are inputs. |
| `temperature`, `dew_point`, derived `relative_humidity` | 2 m temperature, 2 m dewpoint | Near-surface moisture and thermodynamic state. RH is derived with a saturation-vapour-pressure relationship. |
| `total_column_water_vapour` | Total column water vapour (`tcwv`) | Column moisture supply. |
| `CAPE`, `CIN` | Convective available potential energy; convective inhibition | Convective instability and the energy barrier to convection. |
| `cloud_cover` | Total cloud cover (`tcc`) | Cloud development and broad precipitation-system state. |
| `wind_u_10m`, `wind_v_10m` and lags | 10 m U and V wind | Uses vector components without circular wind-direction encoding; transports moisture and storms. |
| `pressure` and lags/rolling summaries | Mean sea-level pressure (`msl`) | Synoptic pressure patterns; recent history provides a pressure tendency proxy. |
| `boundary_layer_height` | Boundary layer height (`blh`) | Mixing and low-level moisture transport context. |
| `orography_m` | Surface geopotential (`z`) divided by standard gravity | Static terrain elevation for orographic effects. |
| latitude/longitude and cyclical time | Grid coordinates and timestamp | Regional gradients and season/diurnal climatology. |

These are physically motivated starting predictors, not a claim that one fixed subset is optimal for every city or season. After the real files are imported, XGBoost permutation importance on the **validation period** is written to `experiments/validation_permutation_importance_*.csv`. Use those rankings and a validation-only ablation study to refine the input set. Never choose fields based on the final test scores.

## Target, timing and leakage

ERA5 hourly total precipitation is an accumulation over the hour ending at its validity time and is supplied in metres; the converter multiplies by 1000 to make millimetres. With a one-hour horizon, predictors at time `t` estimate precipitation for the next hourly interval (`t` to `t+1`, labelled by its ending time). Historical precipitation at `t` is permitted; `target_rainfall_mm` and its event label are never input columns. The final test period remains chronological, and a horizon-sized gap purges boundary labels.

ERA5 is a reanalysis, not a real-time forecast. Retrospective skill on ERA5 analysis inputs does not directly measure operational forecast skill. For deployment, use forecast fields available at issue time and compare against gauge/radar observations.

## Spatial limit

The linked ERA5 atmospheric grid is hourly at 0.25° (~28 km north-south). The ingestion path retains native ERA5 grid cells. Repeating or interpolating these cells onto 1 km squares does not create independent 1 km truth and can make evaluation scores misleading. Real 1 km evaluation needs fine-resolution radar/gauge targets or a separately validated downscaling method.

References: [ERA5 single-level dataset overview](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview), [CDS variable selection](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), [ERA5 precipitation accumulation and units](https://confluence.ecmwf.int/display/CKB/Conversion+table+for+accumulated+variables+%28total+precipitation%2Ffluxes%29).
