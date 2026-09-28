# Data schema

Required columns: UTC `timestamp`, `grid_id`, `latitude`, `longitude`. Core weather values: temperature, relative humidity, pressure, wind speed/direction, cloud cover, dew point and rainfall accumulations at 1/3/6/12/24 hours. Optional columns such as CAPE, soil moisture, radiation and visibility are tolerated. Rain event threshold and forecast horizon are configurable; an H-hour target sums hourly rainfall from t+1 through t+H.

CSV ingestion validates required columns and timestamps. Missing values are not globally filled; numeric imputation with indicators is fitted only on training features. Synthetic rows have weather-correlated rainfall and sparse pressure gaps; every generated file must be treated as synthetic.

## ERA5 fields and units

The optional ERA5 NetCDF adapter uses native single-level fields: `t2m`, `d2m`, `u10`, `v10`, `msl`, `tcc`, `tp`, `tcwv`, `cape`, `cin`, `blh`, and `z`. It converts kelvin to °C, pressure Pa to hPa, fractional cloud to percent, total precipitation m to mm, derives RH and wind magnitude/direction, and retains native ERA5 grid-cell IDs. The target is precipitation over the following forecast horizon; rainfall at or before the prediction time may enter only as past/current observed rainfall.
