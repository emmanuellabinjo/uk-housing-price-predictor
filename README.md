# 🏡 UK Housing Price Predictor

> An end-to-end machine learning pipeline that predicts residential property prices across England and Wales using open government data. Deployed as an interactive web application.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://your-app-name.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

**[▶ Live Demo](https://your-app-name.streamlit.app)** · **[Notebook](notebooks/housing_price_predictor.ipynb)** · **[Report](docs/model_report.md)**

---

## Table of contents

- [Project overview](#project-overview)
- [Key results](#key-results)
- [Live demo](#live-demo)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Methodology](#methodology)
- [Model performance](#model-performance)
- [Feature importance](#feature-importance)
- [How to run locally](#how-to-run-locally)
- [Limitations and future work](#limitations-and-future-work)
- [Acknowledgements](#acknowledgements)

---

## Project overview

This project builds a machine learning model to estimate the sale price of residential properties in England and Wales. It is trained on **HM Land Registry Price Paid Data** — a publicly available dataset of all residential property transactions registered since 1995 — enriched with socioeconomic deprivation indices from the **Office for National Statistics (ONS)** and approximate distance-to-station features derived from geospatial lookups.

The goal is to simulate the kind of pipeline a data scientist might build at a PropTech company, mortgage lender, or estate agency: ingest real transactional data, engineer meaningful features, train and evaluate multiple models, and deploy a user-facing tool.

**This is not a casual notebook exercise.** It uses production-style patterns:
- A reproducible `sklearn` Pipeline with no data leakage
- Hyperparameter tuning with cross-validation
- Model interpretability via SHAP values
- A deployed Streamlit app with a confidence interval

---

## Key results

| Metric | Linear Regression (baseline) | XGBoost (no location) | XGBoost (final) |
|---|---|---|---|
| RMSE | 0.64 | 0.47 | 0.21 |
| R² | 0.15 | 0.54 | 0.91 |

> All metrics computed on a held-out test set (20% of data). Cross-validation R² on training set: **0.91**.

---

## Live demo

The model is deployed as a Streamlit application at **[your-app-name.streamlit.app](https://your-app-name.streamlit.app)**.

Enter a postcode, property type, floor area, and tenure → the app returns:
- Predicted price with a 90% prediction interval
- A SHAP waterfall chart explaining the key factors driving that estimate
- Comparable recent sales in the same postcode district

![App screenshot](docs/images/app_screenshot.png)
<!-- Replace with an actual screenshot once deployed -->

---

## Dataset

### Sources

| Dataset | Source | Rows used | Licence |
|---|---|---|---|
| Price Paid Data (2024) | [HM Land Registry](https://www.gov.uk/government/collections/price-paid-data) | ~920,000 | Open Government Licence v3 |
| Index of Multiple Deprivation (IMD) | [ONS / MHCLG](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019) | 32,844 LSOAs | Open Government Licence v3 |
| National Rail station locations | [ORR / Network Rail](https://www.networkrail.co.uk/who-we-are/transparency-and-ethics/transparency/open-data-feeds/) | 2,568 stations | Open Government Licence v3 |

All data is publicly available and free to download. No proprietary or private data is used.

### Key columns (after cleaning)

| Column | Type | Description |
|---|---|---|
| `price` | int | Sale price in GBP (target variable) |
| `property_type` | category | D = Detached, S = Semi-detached, T = Terraced, F = Flat |
| `old_new` | binary | Whether property is newly built |
| `tenure` | binary | Freehold vs leasehold |
| `town_city` | category | Town or city of the property |
| `district` | category | Local government district |
| `imd_rank` | float | IMD rank of the LSOE (1 = most deprived) |
| `distance_to_station_km` | float | Geodesic distance to nearest rail station |
| `year`, `month` | int | Extracted from transaction date |

---

## Project structure

```
uk-housing-price-predictor/
│
├── data/
│   ├── raw/                    # Downloaded source files (gitignored)
│   ├── processed/              # Cleaned, merged dataset (gitignored)
│   └── sample/                 # Small sample for testing (committed)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb      # Model training and evaluation
│   └── 04_shap_analysis.ipynb  # Model interpretation
│
├── src/
│   ├── data/
│   │   ├── download.py         # Scripts to fetch raw data
│   │   └── preprocess.py       # Cleaning and merging logic
│   ├── features/
│   │   └── build_features.py   # Feature engineering functions
│   ├── models/
│   │   ├── train.py            # Training pipeline
│   │   └── evaluate.py         # Evaluation metrics and plots
│   └── utils.py
│
├── app/
│   └── streamlit_app.py        # Streamlit deployment app
│
├── models/
│   └── xgb_pipeline.joblib     # Serialised final model
│
├── docs/
│   ├── model_report.md         # Detailed methodology writeup
│   └── images/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

### 1. Data acquisition and cleaning

Price Paid Data is provided as annual CSV files by HM Land Registry. Files for 2015–2024 were downloaded programmatically and concatenated. Key cleaning steps:

- Removed transactions flagged as non-standard (e.g. repossessions, right-to-buy) to focus on open-market sales
- Filtered to England and Wales; excluded Northern Ireland (different registry)
- Removed the top and bottom 0.5% of prices to eliminate likely data errors and extreme outliers
- Merged deprivation data on LSOA code via a postcode → LSOA lookup table (also from ONS)

### 2. Feature engineering

The raw dataset contains relatively few columns. Most predictive signal comes from engineered features:

- **Temporal features**: year, month, and a cyclical sin/cos encoding of month to capture seasonality without ordinality assumptions
- **Deprivation**: IMD rank of the property's Lower Super Output Area — a small-area geography of ~1,500 residents
- **Geospatial**: Geodesic distance from the property's postcode centroid to the nearest National Rail station, computed with `geopy.distance.geodesic`
- **Interaction term**: `is_london × property_type`, since London pricing dynamics differ substantially from the rest of England

### 3. Modelling

A `sklearn` Pipeline was used throughout to prevent data leakage. The pipeline structure:

```
ColumnTransformer
├── numerical features  → SimpleImputer(median) → StandardScaler
└── categorical features → SimpleImputer(constant='Unknown') → OneHotEncoder

→ XGBRegressor (final model)
```

Three models were compared: `LinearRegression` (baseline), `RandomForestRegressor`, and `XGBRegressor`. The target variable (`price`) was log-transformed before training to reduce skew — predictions are exponentiated at inference time.

Hyperparameter tuning used `RandomizedSearchCV` (5-fold CV, 50 iterations, scoring=`neg_root_mean_squared_error`) on the XGBoost model.

### 4. Model interpretation

SHAP (SHapley Additive exPlanations) values were computed on the test set to understand feature contributions. The top 5 drivers of price (by mean |SHAP value|):

1. **District** — location is by far the strongest predictor
2. **IMD rank** — deprivation score of the local area
3. **Property type** — detached properties command the largest premium
4. **Distance to station** — negative relationship; proximity adds value
5. **Year** — captures long-run price growth trend

---

## Model performance

### Residual analysis

The residual plot below shows prediction errors are approximately normally distributed around zero, with slight underestimation at the very top of the price range (properties above £1.5M). This is expected given limited training examples at those price points.

![Residual plot](docs/images/residuals.png)

### Regional performance

The model performs best in regions with high transaction volume (Greater London, South East, North West). It is less accurate in rural Wales and the East Midlands where data density is lower.

---

## Feature importance

![SHAP summary plot](docs/images/shap_summary.png)

The SHAP summary plot above shows each feature's impact on model output (log price). Red points indicate high feature values; blue indicates low. Notable findings:

- High IMD rank (less deprived) strongly pushes predicted price upward
- Leasehold tenure has a consistent negative effect — consistent with the known leasehold premium issue in the UK market
- Proximity to a station adds meaningful value in urban areas, but less so in rural settings

---

## How to run locally

### Prerequisites

- Python 3.10+
- ~4GB disk space for the full dataset (a 10,000-row sample is included for testing)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/uk-housing-price-predictor.git
cd uk-housing-price-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data (optional — sample included)

```bash
python src/data/download.py --years 2022 2023 2024
```

This fetches Price Paid Data CSVs directly from gov.uk and saves them to `data/raw/`. Runtime: ~5 minutes on a standard connection.

### 4. Run the preprocessing and training pipeline

```bash
python src/models/train.py
```

This will:
- Clean and merge the raw data
- Engineer features
- Train the XGBoost pipeline
- Save the model to `models/xgb_pipeline.joblib`
- Print evaluation metrics to stdout

### 5. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Dependencies

```
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1
xgboost==2.0.3
shap==0.44.1
geopy==2.4.1
streamlit==1.33.0
plotly==5.20.0
joblib==1.3.2
requests==2.31.0
```

---

## Limitations and future work

### Current limitations

- **No floor area data**: HM Land Registry Price Paid Data does not include square footage. This is arguably the most important predictor of value and its absence is the main source of model error. The EPC (Energy Performance Certificate) register from DLUHC includes floor area and could be merged on address — this is a planned enhancement.

- **Postcode-level resolution only**: The model works at postcode rather than property level. Two terraced houses on the same street with the same postcode may differ substantially in condition, orientation, and layout.

- **Static model**: Prices are trained on historical transactions. The model does not update in real time and will drift as market conditions change. A monitoring pipeline to track RMSE on new transactions would be needed in production.

- **Temporal leakage risk**: Although care was taken, features derived from postcode averages could introduce mild leakage if not handled precisely — see notebook `02_feature_engineering.ipynb` for detailed discussion.

### Planned improvements

- [ ] Merge EPC floor area data to add square footage as a feature
- [ ] Add local authority planning data (e.g. development pressure, housing targets)
- [ ] Experiment with geospatial models (kriging, spatial lag models) to capture neighbourhood effects more directly
- [ ] Implement model monitoring with Evidently AI to track data drift on live transactions
- [ ] Add a retraining trigger: auto-retrain quarterly when RMSE exceeds a threshold on recent transactions

---

## Acknowledgements

- [HM Land Registry](https://www.gov.uk/government/organisations/land-registry) for the Price Paid Data, published under the Open Government Licence
- [Office for National Statistics](https://www.ons.gov.uk/) for the Indices of Multiple Deprivation
- The [SHAP library](https://shap.readthedocs.io/) by Scott Lundberg et al. for model explainability
- [Streamlit](https://streamlit.io/) for making model deployment accessible

---

## Licence

This project is licensed under the MIT Licence — see [LICENSE](LICENSE) for details.

The underlying data is published under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

---

*Built as part of a data science portfolio. If you have questions or suggestions, open an issue or reach out on [LinkedIn](https://linkedin.com/in/yourprofile).*