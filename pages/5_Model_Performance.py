"""
Model Performance Page - Metrics, Curves, and SHAP Global Explanations
Full transparency on model quality for technical audiences.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_loader import load_test_data, load_metrics, load_shap_values, load_feature_names
from src.model_utils import load_lgbm_model, load_lr_model, get_threshold, prepare_features
from src.ui_components import (
    load_css, section_header, gradient_divider, metric_card,
    COLORS, CHART_LAYOUT
)

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, calibration_curve
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Page config
st.set_page_config(page_title="Model Performance - SkyGuard", page_icon=";", layout="wide")
load_css()

# Header
st.markdown("""
<div style="margin-bottom: 30px;">
    <h1 style="color: #1D1D1F; font-size: 32px; font-weight: 700; margin: 0;">
        Model Performance
    </h1>
    <p style="color: #6E6E73; font-size: 16px; margin: 5px 0 0 0;">
        Comprehensive model evaluation metrics and SHAP explanations
    </p>
</div>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def get_test_data():
    return load_test_data()

@st.cache_data
def get_predictions():
    X_test, y_test = load_test_data()
    model = load_lgbm_model()
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_test, y_proba

metrics = load_metrics()
X_test, y_test = get_test_data()
y_test, y_proba_lgbm = get_predictions()

# Model comparison table
section_header("Model Comparison", "LightGBM vs Logistic Regression baseline")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### LightGBM (Primary)")
    lgbm_metrics = metrics.get('lgbm', {})

    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.metric("Accuracy", f"{lgbm_metrics.get('accuracy', 0):.1%}")
        st.metric("Precision", f"{lgbm_metrics.get('precision', 0):.1%}")
    with met_col2:
        st.metric("Recall", f"{lgbm_metrics.get('recall', 0):.1%}")
        st.metric("F1 Score", f"{lgbm_metrics.get('f1', 0):.1%}")
    with met_col3:
        st.metric("AUC-ROC", f"{lgbm_metrics.get('auc_roc', 0):.1%}")
        st.metric("AUC-PR", f"{lgbm_metrics.get('auc_pr', 0):.1%}")

with col2:
    st.markdown("#### Logistic Regression (Baseline)")
    lr_metrics = metrics.get('lr', {})

    met_col1, met_col2, met_col3 = st.columns(3)
    with met_col1:
        st.metric("Accuracy", f"{lr_metrics.get('accuracy', 0):.1%}")
        st.metric("Precision", f"{lr_metrics.get('precision', 0):.1%}")
    with met_col2:
        st.metric("Recall", f"{lr_metrics.get('recall', 0):.1%}")
        st.metric("F1 Score", f"{lr_metrics.get('f1', 0):.1%}")
    with met_col3:
        st.metric("AUC-ROC", f"{lr_metrics.get('auc_roc', 0):.1%}")
        st.metric("AUC-PR", f"{lr_metrics.get('auc_pr', 0):.1%}")

gradient_divider()

# Confusion Matrix
section_header("Confusion Matrix", "LightGBM predictions on test set")

col1, col2 = st.columns([1, 1])

with col1:
    threshold = get_threshold()
    y_pred = (y_proba_lgbm >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate percentages
    cm_pct = cm / cm.sum() * 100

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Active', 'Predicted Churned'],
        y=['Actual Active', 'Actual Churned'],
        colorscale='Blues',
        showscale=True,
        text=[[f"{cm[0,0]}<br>({cm_pct[0,0]:.1f}%)", f"{cm[0,1]}<br>({cm_pct[0,1]:.1f}%)"],
              [f"{cm[1,0]}<br>({cm_pct[1,0]:.1f}%)", f"{cm[1,1]}<br>({cm_pct[1,1]:.1f}%)"]],
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate='%{y} & %{x}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=f"Threshold: {threshold:.2f}", font=dict(size=14)),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ['xaxis', 'yaxis']}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Confusion matrix interpretation
    tn, fp, fn, tp = cm.ravel()

    st.markdown("""
    <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
        <h4 style="margin: 0 0 15px 0;">Interpretation</h4>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    - **True Negatives (TN):** {tn:,} - Active customers correctly identified
    - **False Positives (FP):** {fp:,} - Active customers incorrectly flagged as churners
    - **False Negatives (FN):** {fn:,} - Churners missed by the model
    - **True Positives (TP):** {tp:,} - Churners correctly identified

    **Business Impact:**
    - Catching {tp/(tp+fn)*100:.1f}% of churners ({tp:,} out of {tp+fn:,})
    - Precision of {tp/(tp+fp)*100:.1f}% means {fp:,} false alarms
    """)

    st.markdown("</div>", unsafe_allow_html=True)

gradient_divider()

# ROC and PR Curves
section_header("ROC and Precision-Recall Curves")

col1, col2 = st.columns(2)

with col1:
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba_lgbm)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    # Add AUC fill
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        fill='tozeroy',
        fillcolor='rgba(0, 113, 227, 0.1)',
        line=dict(color=COLORS['primary'], width=2),
        name=f'LightGBM (AUC = {roc_auc:.3f})',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))

    # Add diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color=COLORS['gray'], dash='dash'),
        name='Random Classifier',
        showlegend=True
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROC Curve", font=dict(size=16)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba_lgbm)
    pr_auc = average_precision_score(y_test, y_proba_lgbm)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        fill='tozeroy',
        fillcolor='rgba(52, 199, 89, 0.1)',
        line=dict(color=COLORS['green'], width=2),
        name=f'LightGBM (AP = {pr_auc:.3f})',
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))

    # Add baseline
    baseline = y_test.mean()
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=COLORS['gray'],
        annotation_text=f"Baseline ({baseline:.2f})"
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Precision-Recall Curve", font=dict(size=16)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

gradient_divider()

# Threshold Analysis
section_header("Threshold Analysis", "Adjust decision threshold to see impact on metrics")

threshold_slider = st.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(get_threshold()),
    step=0.05
)

y_pred_adj = (y_proba_lgbm >= threshold_slider).astype(int)
cm_adj = confusion_matrix(y_test, y_pred_adj)
tn, fp, fn, tp = cm_adj.ravel()

precision_adj = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_adj = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_adj = 2 * (precision_adj * recall_adj) / (precision_adj + recall_adj) if (precision_adj + recall_adj) > 0 else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Precision", f"{precision_adj:.1%}")
with col2:
    st.metric("Recall", f"{recall_adj:.1%}")
with col3:
    st.metric("F1 Score", f"{f1_adj:.1%}")
with col4:
    st.metric("Churners Caught", f"{tp:,} of {tp+fn:,}")

# Threshold impact visualization
thresholds_to_plot = np.arange(0.1, 0.9, 0.05)
metrics_by_threshold = []

for t in thresholds_to_plot:
    y_p = (y_proba_lgbm >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_p)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    metrics_by_threshold.append({
        'Threshold': t,
        'Precision': prec,
        'Recall': rec,
        'F1': f1_t
    })

metrics_df = pd.DataFrame(metrics_by_threshold)

fig = go.Figure()
for metric, color in [('Precision', COLORS['primary']), ('Recall', COLORS['green']), ('F1', COLORS['orange'])]:
    fig.add_trace(go.Scatter(
        x=metrics_df['Threshold'],
        y=metrics_df[metric],
        name=metric,
        line=dict(color=color, width=2),
        hovertemplate=f'{metric}: %{{y:.3f}}<extra></extra>'
    ))

fig.add_vline(x=threshold_slider, line_dash="dash", line_color=COLORS['red'],
              annotation_text=f"Current: {threshold_slider:.2f}")

fig.update_layout(
    **CHART_LAYOUT,
    title=dict(text="Metrics by Threshold", font=dict(size=14)),
    xaxis_title="Threshold",
    yaxis_title="Score",
    height=350
)
st.plotly_chart(fig, use_container_width=True)

gradient_divider()

# SHAP Global Explanations
section_header("SHAP Global Explanations", "Feature importance with directionality")

if HAS_SHAP:
    try:
        shap_values, expected_value = load_shap_values()
        feature_names = load_feature_names()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Feature Importance (Mean |SHAP|)")

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=True).tail(15)

            fig = go.Figure(data=[go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                marker_color=COLORS['primary'],
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            )])

            fig.update_layout(
                **CHART_LAYOUT,
                xaxis_title="Mean |SHAP Value|",
                yaxis_title="",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### SHAP Summary (Beeswarm)")

            # Create beeswarm plot with matplotlib
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, max_display=15, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        gradient_divider()

        # SHAP Dependence Plots
        section_header("SHAP Dependence Plots", "Top 3 features interaction effects")

        top_features = importance_df.tail(3)['Feature'].tolist()[::-1]

        cols = st.columns(3)
        for i, feature in enumerate(top_features):
            with cols[i]:
                st.markdown(f"**{feature}**")
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.dependence_plot(
                    feature, shap_values, X_test,
                    feature_names=feature_names,
                    show=False, ax=ax
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.warning(f"Could not load SHAP values: {str(e)}")
else:
    st.info("SHAP library not installed. Install shap for explainability features.")

gradient_divider()

# Class Imbalance Strategy
section_header("Class Imbalance Strategy", "Three-pronged approach used in training")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #0071E3;
        height: 100%;
    ">
        <h4 style="color: #0071E3; margin: 0 0 10px 0;">1. SMOTE</h4>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            Synthetic Minority Oversampling applied to training set only.
            Brought minority class to 40% of majority (not 50/50 to avoid overfitting).
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #34C759;
        height: 100%;
    ">
        <h4 style="color: #34C759; margin: 0 0 10px 0;">2. Class Weights</h4>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            LightGBM scale_pos_weight parameter set to ~2.5 (after SMOTE).
            Gives additional signal to correctly classify minority class.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #FF9500;
        height: 100%;
    ">
        <h4 style="color: #FF9500; margin: 0 0 10px 0;">3. Threshold Tuning</h4>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            Optimized decision threshold on validation set to maximize F1.
            Final threshold: {get_threshold():.2f} (vs default 0.5).
        </p>
    </div>
    """, unsafe_allow_html=True)
