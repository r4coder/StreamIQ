<div align="center">

# 🎬 StreamIQ
### OTT Churn Prediction & Retention Intelligence System

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/XGBoost-2.0+-orange?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white"/>

<br/>

> **A production-grade ML analytics platform that predicts OTT subscriber churn and delivers AI-powered retention strategies — wrapped in a Netflix-inspired premium dark UI.**

<br/>

[✨ Features](#-features) • [🏗️ Architecture](#️-architecture) • [⚙️ Installation](#️-installation) • [🚀 Deployment](#-deployment) • [🤖 ML Workflow](#-ml-workflow) • [📊 Dataset](#-dataset)

---

![StreamIQ Banner](https://via.placeholder.com/1200x400/080b14/e50914?text=StreamIQ+·+Churn+Intelligence+Platform)

</div>

---

## 🎯 What is StreamIQ?

StreamIQ is an end-to-end **Customer Churn Prediction & Retention Intelligence System** built for OTT (Over-The-Top) streaming platforms like Netflix, Prime Video, and Hotstar.

It combines **advanced machine learning** with a **stunning analytics dashboard** to help product and growth teams:

- 🔮 **Predict** which subscribers are likely to cancel their plans
- 📊 **Analyse** platform usage, engagement, and revenue patterns
- 🎯 **Segment** customers into Low / Medium / High churn risk buckets
- 💡 **Recommend** personalised, data-driven retention strategies
- 💸 **Quantify** revenue at risk before it's lost

---

## ✨ Features

### 🏠 Executive Dashboard
- 6 animated KPI cards — Total Customers, Churn Rate, Revenue at Risk, Avg Watch Hours, High-Risk Users, Retention Rate
- Churn composition donut chart with live annotation
- Churn rate by account age (colour-coded bar chart)
- Watch hours distribution split by churn status
- Subscription tier churn analysis

### 📊 Analytics & EDA
- Interactive filters — subscription type, gender, age range
- Genre-level churn heatmap
- Binge watch frequency vs churn correlation
- Stacked revenue breakdown (safe vs at-risk)
- Payment method churn distribution
- Session time scatter plots coloured by churn status
- Days-since-login trend line
- Full feature correlation heatmap

### 🤖 ML Model Comparison
- 4 algorithms trained and compared head-to-head
- Full metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Interactive overlaid ROC curves
- Radar chart for visual model comparison
- Auto best-model selection by ROC-AUC
- Feature importance chart (top 12 predictive features)

### 🔮 Churn Predictor
- Beautiful 4-row customer profile input form
- Real-time churn probability scoring
- Animated gauge chart with colour-coded zones
- Risk badge — Low / Medium / High
- Personalised AI retention recommendations

### 💡 Business Intelligence
- Risk segmentation KPIs with revenue exposure
- Revenue by risk segment (donut chart)
- Risk distribution across subscription tiers
- Top 20 high-risk customer table with colour-coded rows
- 6 strategic, data-backed retention recommendations

---

## 🏗️ Architecture

```
StreamIQ/
│
├── 📁 data/
│   ├── generate_data.py          # Synthetic OTT dataset generator (5,000 rows)
│   └── ott_churn_data.csv        # Auto-generated dataset
│
├── 📁 src/
│   ├── preprocessing.py          # Missing value imputation, encoding, scaling
│   └── model_training.py         # Model training, evaluation, risk logic
│
├── 📁 models/                    # Persisted ML artifacts (auto-generated)
│   ├── best_model.pkl            # Best performing model
│   ├── all_results.pkl           # All model results + ROC data
│   ├── feature_importance.pkl    # Feature importance scores
│   ├── encoders.pkl              # Label encoders per categorical column
│   ├── scaler.pkl                # StandardScaler
│   └── feature_names.pkl        # Ordered feature list
│
├── 📁 app/
│   └── streamlit_app.py          # Main dashboard — 5 pages, ~800 lines
│
├── 📁 .streamlit/
│   └── config.toml               # Dark theme + server config
│
├── train.py                      # One-shot training pipeline
├── requirements.txt
└── README.md
```

---

## 🤖 ML Workflow

```
┌─────────────────────────────────────────────────────┐
│              Raw OTT Dataset (5,000 rows)            │
│                    19 Features                       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Preprocessing                       │
│  ├── Median imputation for missing numeric values    │
│  ├── LabelEncoder for 7 categorical features         │
│  ├── StandardScaler normalisation                    │
│  └── 80/20 stratified train-test split               │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               Model Training & Evaluation            │
│  ├── Logistic Regression                             │
│  ├── Decision Tree      (max_depth=6)                │
│  ├── Random Forest      (200 estimators)             │
│  └── XGBoost            (200 estimators, lr=0.05)    │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│         Auto Best Model Selection (ROC-AUC)          │
│              Artifacts saved via Joblib              │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit Dashboard                     │
│  ├── Executive KPIs & Trend Charts                   │
│  ├── Interactive EDA Visualisations                  │
│  ├── Model Leaderboard & ROC Curves                  │
│  ├── Real-time Churn Prediction + Gauge              │
│  └── Risk Segmentation & Business Insights           │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

The dataset is synthetically generated with **realistic business logic** for churn — users with high inactivity, low watch hours, many support tickets, or disabled auto-renewal have higher churn probability.

| Feature | Type | Description |
|---|---|---|
| `customer_id` | ID | Unique identifier |
| `age` | Numeric | Customer age (18–70) |
| `gender` | Categorical | Male / Female / Other |
| `subscription_type` | Categorical | Basic / Standard / Premium |
| `monthly_watch_hours` | Numeric | Hours watched per month |
| `favorite_genre` | Categorical | Action, Drama, Comedy, etc. |
| `devices_used` | Numeric | Number of streaming devices |
| `avg_session_time` | Numeric | Average session length (hours) |
| `monthly_subscription_cost` | Numeric | Monthly plan cost (₹) |
| `payment_method` | Categorical | Credit Card / UPI / Wallet etc. |
| `support_tickets` | Numeric | Number of support tickets raised |
| `ads_tolerance` | Categorical | Low / Medium / High |
| `days_since_last_login` | Numeric | Inactivity indicator |
| `number_of_profiles` | Numeric | Sub-profiles on account |
| `binge_watch_frequency` | Categorical | Never → Always scale |
| `content_rating_given` | Numeric | Average content rating (1–5) |
| `account_age_months` | Numeric | Tenure in months |
| `auto_renew_enabled` | Binary | Auto-renewal status |
| `churn` | **Target** | 0 = Retained · 1 = Churned |

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.833 | 0.250 | 0.025 | 0.046 | **0.694** |
| Decision Tree | 0.827 | 0.196 | 0.181 | 0.188 | 0.639 |
| Random Forest | 0.835 | 0.375 | 0.047 | 0.083 | 0.688 |
| XGBoost | 0.828 | 0.231 | 0.079 | 0.118 | 0.655 |

> Best model selected automatically based on ROC-AUC score.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/r4coder/StreamIQ.git
cd StreamIQ

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train models  (one-time, ~30–60 seconds)
python train.py

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

Then open **http://localhost:8501** in your browser.

> 💡 If you skip step 4, the dashboard will auto-train models on first launch.

---

## 🚀 Deployment

### ☁️ Streamlit Cloud *(recommended)*
1. Push repo to GitHub *(include the `models/` folder)*
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Click **New app** → select your repo
4. Set **Main file path** → `app/streamlit_app.py`
5. Click **Deploy** — live in ~2 minutes ✅

### 🟣 Render
```yaml
# render.yaml
services:
  - type: web
    name: streamiq
    env: python
    buildCommand: pip install -r requirements.txt && python train.py
    startCommand: streamlit run app/streamlit_app.py --server.port $PORT
```

### 🤗 Hugging Face Spaces
```
SDK: Streamlit
App file: app/streamlit_app.py
Add requirements.txt at repo root
```

> ⚠️ **Important:** Always commit the `models/` folder to your repo before deploying. Streamlit Cloud's free tier may timeout during on-the-fly training.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data Engineering | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualisation | Plotly, Seaborn, Matplotlib |
| Dashboard | Streamlit + Custom CSS |
| Model Persistence | Joblib |
| Fonts | Syne, DM Sans (Google Fonts) |
| Deployment | Streamlit Cloud / Render / HF Spaces |

---

## 🔮 Future Improvements

- [ ] SHAP explainability — per-prediction feature contribution breakdown
- [ ] Cohort retention analysis by signup month
- [ ] Time-series churn forecasting with LSTM
- [ ] REST API endpoint for real-time scoring integration
- [ ] A/B test simulation module for retention campaigns
- [ ] User authentication with role-based access (Admin / Analyst)
- [ ] Automated model retraining pipeline with data drift detection
- [ ] Email alert system for high-risk customer threshold breaches

---

## 📋 Resume-Ready Description

> **StreamIQ — OTT Churn Prediction & Retention Intelligence System**
>
> Built a production-grade ML analytics platform to predict OTT subscriber churn using Python, XGBoost, and Streamlit. Engineered a 5,000-row synthetic dataset with 19 features and realistic churn business logic. Trained and compared 4 classification models achieving 83.5% accuracy and 0.694 ROC-AUC. Designed a premium Netflix-inspired dark-mode Streamlit dashboard with 5 interactive pages, Plotly visualisations, real-time churn scoring with probability gauge, risk segmentation (Low/Medium/High), and AI-powered personalised retention recommendations.
>
> **Tech:** Python · Pandas · NumPy · Scikit-learn · XGBoost · Plotly · Streamlit · Joblib

---

## 👨‍💻 Author

**r4coder**
- GitHub: [@r4coder](https://github.com/r4coder)

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

*Built with ❤️ by r4coder*

</div>
