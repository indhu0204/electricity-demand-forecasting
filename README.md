# Forecasting German Electricity Demand

This repository contains a reproducible time-series forecasting project for modelling and forecasting **weekly German electricity demand**.

The project uses German electricity load data together with weather-derived temperature features, aggregates the demand series to weekly level, and compares benchmark, statistical, feature-based, and neural forecasting approaches.

## Project aim

The aim of this project is to forecast weekly German electricity demand and compare the accuracy, interpretability, and complexity of different forecasting models.

The main research questions are:

1. How well do simple benchmark methods forecast weekly German electricity demand?
2. Does a SARIMA or SARIMAX-type model improve on the benchmark forecasts?
3. Do engineered temperature and related features improve forecast accuracy?
4. How do feature-based machine-learning models compare with classical statistical models?
5. Does an LSTM model justify its additional complexity on weekly data?
6. Which model is most appropriate for operational forecasting?

## Data

The target variable in this project is weekly electricity demand in gigawatts:

```text
load_gw
```

The workflow starts from raw hourly German electricity demand data and additional temperature data. These are cleaned and transformed into processed datasets used for modelling.

Main data folders:

```text
data/raw/
data/interim/
data/processed/
```

Key processed files include:

```text
weekly_load_temperature_features.csv
weekly_temperature_features.csv
hourly_load_temperature_features.csv
```

Additional modelling outputs for later project parts are also stored in `data/processed/`, including feature-model predictions, feature importance files, LSTM predictions, and tuning results.

## Repository structure

```text
electricity-demand-forecasting/
│
├── data
│   ├── raw
│   │   └── time_series_60min_singleindex.csv
│   ├── interim
│   └── processed
│       ├── hourly_load_temperature_features.csv
│       ├── part4_feature_model_metrics.csv
│       ├── part4_feature_model_predictions.csv
│       ├── part4_random_forest_feature_importance.csv
│       ├── part5_feature_model_metrics.csv
│       ├── part5_feature_model_predictions.csv
│       ├── part5_random_forest_feature_importance.csv
│       ├── part6_lstm_metrics.csv
│       ├── part6_lstm_predictions.csv
│       ├── part6_lstm_tuning_results.csv
│       ├── weekly_load_temperature_features.csv
│       └── weekly_temperature_features.csv
│
├── src
│   └── electricity_demand
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       ├── plotting.py
│       ├── stationarity.py
│       └── models
│           ├── __init__.py
│           ├── benchmarks.py
│           ├── feature_models.py
│           ├── neural.py
│           ├── plotting.py
│           └── sarimax.py
│
├── scripts
│   ├── build_final_comparison.py
│   ├── download_data.py
│   ├── download_temperature.py
│   ├── make_features.py
│   ├── run_pipeline.py
│   ├── run_sarima_ci_plot.py
│   └── run_stationarity.py
│
├── outputs
│   ├── figures
│   │   ├── part1
│   │   │   ├── weekly_load_acf_diff.png
│   │   │   ├── weekly_load_acf_raw.png
│   │   │   ├── weekly_load_pacf_diff.png
│   │   │   ├── weekly_load_pacf_raw.png
│   │   │   ├── weekly_load_temperature_eda.png
│   │   │   └── weekly_load_timeseries.png
│   │   ├── part2
│   │   │   ├── benchmark_forecasts_2y.png
│   │   │   └── model_forecasts_2y.png
│   │   ├── part3
│   │   │   ├── sarima_forecast_ci_2y.png
│   │   │   ├── sarima_residual_acf.png
│   │   │   ├── sarima_residual_histogram.png
│   │   │   ├── sarima_residual_qq.png
│   │   │   └── sarima_weekly_forecast.png
│   │   ├── part4
│   │   └── part7
│   │       └── final_model_comparison.png
│   ├── forecasts
│   │   ├── all_model_forecasts.csv
│   │   └── part2
│   │       └── all_forecasts.csv
│   ├── metrics
│   │   ├── part2
│   │   │   └── model_comparison.csv
│   │   ├── part3
│   │   │   └── sarima_metrics.csv
│   │   └── part7
│   │       └── all_model_metrics_comparison.csv
│   ├── model_objects
│   └── results
│       ├── part1
│       │   ├── weekly_load_adf_diff.csv
│       │   ├── weekly_load_adf_raw.csv
│       │   ├── weekly_load_kpss_diff.csv
│       │   └── weekly_load_kpss_raw.csv
│       ├── part2
│       │   ├── benchmark_metrics.csv
│       │   ├── test_index.csv
│       │   ├── train_index.csv
│       │   └── train_test_sizes.csv
│       ├── part3
│       │   ├── sarima_aic_grid.csv
│       │   ├── sarima_best_model.csv
│       │   ├── sarima_diagnostics_summary.csv
│       │   ├── sarima_forecast_ci_values.csv
│       │   ├── sarima_ljung_box.csv
│       │   └── sarima_residuals.csv
│       └── part4
│
├── reports
│   └── figures
│       ├── part4_actual_vs_predicted.png
│       ├── part5_actual_vs_predicted.png
│       └── part6_lstm_actual_vs_predicted.png
│
├── README.md
└── requirements.txt
```

## Pipeline overview

The main modelling workflow is coordinated through:

```text
scripts/run_pipeline.py
```

This script uses reusable code from the `src/electricity_demand/` package.

The project workflow includes the following stages:

1. Download raw electricity demand data.
2. Download supporting temperature data.
3. Clean and prepare the datasets.
4. Aggregate and transform the demand data to weekly level.
5. Create temperature and other forecasting features.
6. Run stationarity analysis and save ADF/KPSS outputs.
7. Generate benchmark forecasts.
8. Fit SARIMA or SARIMAX-style models.
9. Fit feature-based machine-learning models.
10. Fit an LSTM model.
11. Compare model forecasts on a common test set.
12. Save figures, forecasts, metrics, and diagnostic outputs.

## Main scripts

### Data preparation

```bash
python scripts/download_data.py
python scripts/download_temperature.py
python scripts/make_features.py
```

### Statistical analysis

```bash
python scripts/run_stationarity.py
python scripts/run_sarima_ci_plot.py
```

### End-to-end pipeline

```bash
python scripts/run_pipeline.py
```

### Final comparison figure and summary outputs

```bash
python scripts/build_final_comparison.py
```

## Models

The project compares several forecasting approaches.

### Benchmark models

The benchmark methods include:

```text
Mean forecast
Naive forecast
Seasonal naive forecast
Drift forecast
```

These provide simple reference points for judging whether more advanced models are actually worthwhile.

### SARIMA / SARIMAX model

The statistical forecasting stage uses SARIMA or SARIMAX-style modelling to capture serial dependence and annual seasonality in the weekly series.

Outputs from this part include:

```text
outputs/figures/part3/sarima_weekly_forecast.png
outputs/figures/part3/sarima_forecast_ci_2y.png
outputs/metrics/part3/sarima_metrics.csv
outputs/results/part3/sarima_aic_grid.csv
outputs/results/part3/sarima_best_model.csv
```

### Feature-based machine-learning models

The repository includes feature-based modelling code in:

```text
src/electricity_demand/models/feature_models.py
```

These models use engineered predictors derived from past demand and temperature-related variables.

Related processed outputs include:

```text
part4_feature_model_metrics.csv
part4_feature_model_predictions.csv
part4_random_forest_feature_importance.csv
part5_feature_model_metrics.csv
part5_feature_model_predictions.csv
part5_random_forest_feature_importance.csv
```

### Neural model

The repository also includes a neural forecasting stage in:

```text
src/electricity_demand/models/neural.py
```

This part includes an LSTM-based model, with outputs such as:

```text
part6_lstm_metrics.csv
part6_lstm_predictions.csv
part6_lstm_tuning_results.csv
```

## Evaluation

All models are evaluated on a common test period so that performance comparisons are fair and directly comparable.

The main evaluation metrics used in the project are:

```text
MAE
RMSE
MASE
Bias
```

The repository saves comparison outputs in locations such as:

```text
outputs/metrics/part2/model_comparison.csv
outputs/metrics/part3/sarima_metrics.csv
outputs/metrics/part7/all_model_metrics_comparison.csv
```

Forecast outputs are saved in:

```text
outputs/forecasts/all_model_forecasts.csv
outputs/forecasts/part2/all_forecasts.csv
```

## Figures and outputs

The project generates outputs for different analysis stages.

### Part 1: Exploratory analysis and stationarity

Examples:

```text
outputs/figures/part1/weekly_load_timeseries.png
outputs/figures/part1/weekly_load_temperature_eda.png
outputs/figures/part1/weekly_load_acf_raw.png
outputs/figures/part1/weekly_load_pacf_raw.png
outputs/results/part1/weekly_load_adf_raw.csv
outputs/results/part1/weekly_load_kpss_raw.csv
```

### Part 2: Benchmark forecasting

Examples:

```text
outputs/figures/part2/benchmark_forecasts_2y.png
outputs/figures/part2/model_forecasts_2y.png
outputs/results/part2/benchmark_metrics.csv
outputs/results/part2/train_test_sizes.csv
```

### Part 3: SARIMA diagnostics and forecast intervals

Examples:

```text
outputs/figures/part3/sarima_forecast_ci_2y.png
outputs/figures/part3/sarima_residual_acf.png
outputs/figures/part3/sarima_residual_histogram.png
outputs/figures/part3/sarima_residual_qq.png
outputs/results/part3/sarima_ljung_box.csv
outputs/results/part3/sarima_residuals.csv
```

### Final comparison

```text
outputs/figures/part7/final_model_comparison.png
outputs/metrics/part7/all_model_metrics_comparison.csv
outputs/forecasts/all_model_forecasts.csv
```

## Installation

Create and activate a virtual environment.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Reproducing the analysis

A typical workflow from a fresh clone is:

```bash
python scripts/download_data.py
python scripts/download_temperature.py
python scripts/make_features.py
python scripts/run_stationarity.py
python scripts/run_pipeline.py
python scripts/build_final_comparison.py
```

If needed, the SARIMA confidence-interval plot can also be generated separately with:

```bash
PYTHONPATH=src python scripts/run_sarima_ci_plot.py
```

## Report

The repository also includes report figures under:

```text
reports/figures/
```

These include visual outputs for feature-based models and the LSTM stage:

```text
part4_actual_vs_predicted.png
part5_actual_vs_predicted.png
part6_lstm_actual_vs_predicted.png
```

## Good practice

This project follows standard forecasting good practice:

- Uses a time-based train-test split rather than a random split.
- Keeps reusable code inside `src/`.
- Saves intermediate and final outputs systematically.
- Compares advanced models against simple benchmark forecasts.
- Includes statistical diagnostics, residual checks, and forecast-interval outputs.
- Separates raw, interim, processed, and final output data.

## Expected contents of the submission

The final project submission includes:

```text
README.md
requirements.txt
source code in src/
pipeline and utility scripts in scripts/
processed outputs in outputs/
report figures in reports/figures/
data files in data/
```
