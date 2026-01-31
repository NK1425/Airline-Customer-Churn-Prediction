# SkyGuard - Airline Customer Churn Intelligence Platform

> A production-grade machine learning application that predicts airline loyalty program churn using LightGBM, explains predictions with SHAP, and recommends targeted retention strategies.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Problem Statement

Customer churn is a critical challenge for airline loyalty programs. Acquiring a new customer costs 5-25x more than retaining an existing one, yet many airlines lack the predictive capabilities to identify at-risk members before they leave.

**SkyGuard** addresses this by:
- Predicting churn probability with 93%+ F1 Score
- Explaining predictions in plain English using SHAP
- Recommending personalized retention strategies with ROI analysis

## Key Results

| Metric | LightGBM | Logistic Regression |
|--------|----------|---------------------|
| **F1 Score (Churn)** | 93.2% | 93.0% |
| **AUC-ROC** | 99.4% | 99.3% |
| **AUC-PR** | 97.9% | 97.6% |
| **Recall (Churn)** | 91.9% | 91.1% |
| **Precision (Churn)** | 94.5% | 94.9% |

## Technical Highlights

- **Feature Engineering**: 31 engineered features from transactional flight data including `flight_trend_yoy` (the most predictive feature)
- **Class Imbalance**: Three-pronged strategy (SMOTE + scale_pos_weight + threshold tuning)
- **Explainability**: SHAP waterfall plots for every individual prediction
- **Production Pipeline**: Scikit-learn compatible with serialized artifacts
- **Apple-Inspired UI**: Premium Streamlit interface with custom CSS

## Architecture

```
Customer Data     Feature          ML Pipeline        Predictions
     |            Engineering          |                   |
     v                v                v                   v
[Flight Activity] --> [31 Features] --> [LightGBM] --> [Probability]
[Loyalty History]         |               |                |
                          v               v                v
                    [SMOTE + Scaling] [Threshold] --> [Risk Level]
                                          |                |
                                          v                v
                                    [SHAP Explainer] [Recommendations]
```

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat&logo=lightgbm&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-8B5CF6?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## Project Structure

```
airline-churn-predictor/
├── app.py                      # Main Streamlit entry point
├── pages/
│   ├── 1_Dashboard.py          # Executive overview dashboard
│   ├── 2_Customer_Prediction.py # Single customer analysis + SHAP
│   ├── 3_Batch_Prediction.py   # CSV upload for bulk predictions
│   ├── 4_Segment_Analysis.py   # Deep-dive by segment
│   ├── 5_Model_Performance.py  # Metrics, curves, SHAP global
│   └── 6_Retention_Strategies.py # ROI calculator & recommendations
├── model/
│   ├── lgbm_churn_model.joblib # Trained LightGBM model
│   ├── lr_model.joblib         # Logistic Regression baseline
│   ├── shap_values_test.npy    # Pre-computed SHAP values
│   ├── threshold.json          # Optimized decision threshold
│   └── feature_names.json      # Feature list
├── data/
│   ├── Customer_Loyalty_History.csv
│   ├── Customer_Flight_Activity.csv
│   └── processed_features.csv
├── src/
│   ├── feature_engineering.py  # 31 feature engineering pipeline
│   ├── train_model.py          # Model training script
│   ├── model_utils.py          # Prediction & SHAP utilities
│   ├── data_loader.py          # Cached data loading
│   └── ui_components.py        # Styled UI components
├── assets/
│   └── style.css               # Apple-inspired theme
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NK1425/Airline-Customer-Churn-Prediction.git
cd Airline-Customer-Churn-Prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Training the Model (Optional)

If you want to retrain the model:

```bash
# Feature engineering
python src/feature_engineering.py

# Train model
python src/train_model.py
```

## Features

### 1. Executive Dashboard
- Real-time loyalty program health metrics
- Churn rate by tier, enrollment type, geography
- Monthly flight trend visualization
- Top SHAP-based risk factors

### 2. Individual Prediction
- Search by Loyalty Number
- Risk probability gauge with color coding
- SHAP waterfall explanation
- Plain English risk factor explanations
- Personalized retention recommendations

### 3. Batch Prediction
- CSV upload for bulk scoring
- Risk distribution visualization
- Downloadable results (full + high-risk only)
- Processing summary report

### 4. Segment Analysis
- Interactive filters (tier, enrollment, geography, salary, tenure)
- Demographic breakdowns
- Active vs Churned behavior comparison
- Flight activity distributions

### 5. Model Performance
- LightGBM vs Logistic Regression comparison
- Confusion matrix with business interpretation
- ROC and Precision-Recall curves
- Interactive threshold analysis
- SHAP global explanations (beeswarm + dependence plots)

### 6. Retention Strategies
- Risk tier summary with CLV at risk
- Strategy matrix by churn driver
- Interactive ROI calculator
- Downloadable action list

## Dataset

The synthetic dataset models a Northern Lights Air (NLA) loyalty program:

- **Customer_Loyalty_History.csv**: 16,737 customers with demographics and enrollment info
- **Customer_Flight_Activity.csv**: 401,688 monthly flight records (Jan 2017 - Dec 2018)
- **Churn Rate**: ~17% (85:15 class imbalance)

### Key Features Engineered

| Feature | Description | Importance |
|---------|-------------|------------|
| `flight_trend_yoy` | Change in avg monthly flights (2018 vs 2017) | Highest |
| `recency_last_flight` | Months since last flight | High |
| `total_flights_24m` | Total flights in 24 months | High |
| `is_promotion_enrollee` | Enrolled via 2018 promotion | High |
| `points_utilization` | Redeemed / Accumulated points ratio | Medium |

## Model Details

### Why LightGBM?

- Fastest gradient boosting for tabular data
- Native categorical feature support
- Exact tree SHAP computation
- Built-in `scale_pos_weight` for class imbalance
- Handles missing values natively

### Class Imbalance Strategy

1. **SMOTE**: Applied to training set only (sampling_strategy=0.4)
2. **Class Weights**: LightGBM `scale_pos_weight` parameter
3. **Threshold Tuning**: Optimized on validation set for F1

### Hyperparameters (Optuna-tuned)

```python
{
    'n_estimators': 300,
    'max_depth': 3,
    'learning_rate': 0.1,
    'num_leaves': 15,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 0.1
}
```

## Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file: `app.py`
5. Deploy

### Environment Variables

No API keys required - the app is self-contained.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset inspired by Northern Lights Air schema
- SHAP library by Scott Lundberg
- Streamlit for the amazing framework

## Author

**NK1425**

- GitHub: [@NK1425](https://github.com/NK1425)
- Email: nmanthri@memphis.edu

---

Built with Claude Code
