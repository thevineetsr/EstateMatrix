# EstateMatrix 🏢
### Pipeline Driven Property Valuation & Content-Based Discovery System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Project Overview

**EstateMatrix** is an end-to-end Machine Learning and Data Science pipeline built to accurately predict real estate property prices and recommend personalized residential properties across **Gurgaon, India**.

The project transforms raw, unstructured web-scraped listings of flats and independent houses through rigorous stages of data cleaning, imputation, outlier treatment, feature engineering, and ensemble regression modeling, culminating in a multi-modal content-based recommendation engine.

---

## 🚀 Key Features & Highlights

1. **Comprehensive Data Wrangling & Preprocessing**:
   - Cleaned messy real estate listings (extracted numeric values for area, bedrooms, bathrooms, balconies, floor numbers, and facing directions).
   - Unified flat and independent house listings with sector-level standardization.
2. **Exhaustive Exploratory Data Analysis (EDA)**:
   - Univariate and multivariate statistical distributions across price, area, bedroom counts, and sectors.
   - Generated automated profiling reports (`output_report.html`) and geospatial analysis using sector latitude/longitude mappings.
3. **Robust Data Preprocessing & Missing Value Imputation**:
   - Domain-specific imputation strategies based on sector medians, property types, and iterative predictors.
   - IQR-based outlier detection and capping for realistic valuation boundaries.
4. **Advanced Feature Engineering & Selection**:
   - Extracted **Luxury Score** based on parsed property amenities.
   - Created derived metrics like price per square foot, furnished ratios, and location-based density.
   - Filtered high-signal predictors using correlation analysis, Mutual Information, and tree-based feature importance.
5. **Predictive Valuation Engine**:
   - Evaluated baseline linear models against ensemble architectures (Random Forest, Gradient Boosting, XGBoost, LightGBM).
   - Utilized Scikit-learn `Pipeline` and `ColumnTransformer` with target log-transformation (`log1p`) to mitigate skewness.
6. **Multi-Factor Content-Based Recommendation Engine**:
   - Multi-vector similarity matrix weighting:
     $$\text{Similarity} = 30 \cdot S_{\text{facilities (TF-IDF)}} + 20 \cdot S_{\text{characteristics (OHE)}} + 8 \cdot S_{\text{location (Distance)}}$$
   - Recommends the top-N closest properties matching user preferences across amenities, layout, and proximity.

---

## 📂 Project Architecture & Workflow

The repository is structured to follow the progressive data science life cycle:

```
EstateMatrix/
├── 01 Data Ingestion & Preprocessing
│   ├── data-preprocessing-flats.ipynb           # Cleaning raw flat listings
│   ├── data-preprocessing-houses.ipynb          # Cleaning raw house listings
│   ├── merge-flats-and-house.ipynb              # Merging into consolidated properties dataset
│   └── data-preprocessing-level-2.ipynb         # Sector standardization & secondary cleaning
│
├── 02 Exploratory Data Analysis
│   ├── eda-univariate-analysis.ipynb            # Univariate distributions & summaries
│   ├── eda-multivariate-analysis.ipynb          # Correlational & multi-feature analysis
│   ├── eda-pandas-profiling.ipynb               # HTML report generation
│   └── data-visualization.ipynb                 # Interactive geospatial & visual charts
│
├── 03 Data Preparation & Engineering
│   ├── missing-value-imputation.ipynb           # Domain & sector-wise imputation
│   ├── outlier-treatment.ipynb                  # Extreme value handling
│   ├── feature-engineering.ipynb                # Feature extraction (Luxury Score, etc.)
│   ├── feature-selection.ipynb                  # Feature selection experiments
│   └── feature-selection-and-feature-engineering.ipynb
│
├── 04 Predictive Modeling
│   ├── baseline model.ipynb                     # Linear, Ridge, Lasso benchmarks
│   └── model-selection.ipynb                    # Ensemble tuning & final regression pipelines
│
├── 05 Recommendation & Analytics
│   ├── recommender-system.ipynb                 # Multi-factor content-based recommendation engine
│   └── insights-module.ipynb                    # Market analytics & investment insights
│
└── Datasets & Outputs
    ├── appartments.csv                          # Curated apartment facilities data
    ├── flats.csv / houses.csv                   # Raw scraped listings
    ├── gurgaon_properties.csv                   # Merged property dataset
    ├── gurgaon_properties_cleaned_v1 & v2.csv   # Cleaned intermediary datasets
    ├── gurgaon_properties_missing_value_imputation.csv
    ├── gurgaon_properties_outlier_treated.csv
    ├── gurgaon_properties_post_feature_selection.csv
    ├── latlong.csv                              # Latitude & longitude coordinates of Gurgaon sectors
    └── output_report.html                       # Comprehensive pandas-profiling report
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/thevineetsr/EstateMatrix.git
cd EstateMatrix
```

### 2. Set up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or install core data science packages: `pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm jupyter`)*

---

## 📊 Notebook Execution Order

To reproduce the experiments from scratch, execute the notebooks in the following order:

1. `data-preprocessing-flats.ipynb` & `data-preprocessing-houses.ipynb`
2. `merge-flats-and-house.ipynb`
3. `data-preprocessing-level-2.ipynb`
4. `eda-univariate-analysis.ipynb` & `eda-multivariate-analysis.ipynb`
5. `missing-value-imputation.ipynb`
6. `outlier-treatment.ipynb`
7. `feature-engineering.ipynb`
8. `feature-selection.ipynb`
9. `baseline model.ipynb`
10. `model-selection.ipynb`
11. `recommender-system.ipynb`

---

## 📈 Tech Stack

- **Core**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
- **Feature Engineering & NLP**: TfidfVectorizer, Cosine Similarity
- **Visualization**: Matplotlib, Seaborn, Folium / Geopandas
- **Reporting**: YData Profiling (Pandas Profiling)

---

## 👤 Author

**Vineet Singh**
- GitHub: [@thevineetsr](https://github.com/thevineetsr)
