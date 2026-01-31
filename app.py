"""
SkyGuard - Airline Churn Intelligence Platform
Main Streamlit Application Entry Point
"""

import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui_components import load_css

# Page configuration
st.set_page_config(
    page_title="SkyGuard - Airline Churn Intelligence",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# SkyGuard\nAirline Customer Churn Intelligence Platform"
    }
)

# Load custom CSS
load_css()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #0071E3; font-size: 28px; margin: 0; font-weight: 700;">SkyGuard</h1>
        <p style="color: #86868B; font-size: 12px; margin: 5px 0 0 0; text-transform: uppercase; letter-spacing: 1px;">
            Churn Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="padding: 10px 0;">
        <p style="color: #6E6E73; font-size: 13px; margin: 0;">
            <strong>Navigation</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation info
    st.markdown("""
    <div style="color: #86868B; font-size: 12px; padding: 10px 0;">
        <p>Use the sidebar to navigate between pages:</p>
        <ul style="padding-left: 20px; margin: 10px 0;">
            <li><b>Dashboard</b> - Executive overview</li>
            <li><b>Customer Prediction</b> - Individual analysis</li>
            <li><b>Batch Prediction</b> - Bulk scoring</li>
            <li><b>Segment Analysis</b> - Deep dive</li>
            <li><b>Model Performance</b> - Metrics & SHAP</li>
            <li><b>Retention Strategies</b> - Recommendations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 20px; padding: 10px;">
        <p style="color: #86868B; font-size: 11px; margin: 0;">
            Powered by LightGBM + SHAP<br>
            Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


# Main content
st.markdown("""
<div style="text-align: center; padding: 60px 0;">
    <h1 style="color: #1D1D1F; font-size: 48px; font-weight: 700; margin: 0; letter-spacing: -1px;">
        Welcome to SkyGuard
    </h1>
    <p style="color: #6E6E73; font-size: 20px; margin: 20px 0 40px 0; max-width: 600px; margin-left: auto; margin-right: auto;">
        Airline Customer Churn Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
        height: 200px;
    ">
        <div style="font-size: 40px; margin-bottom: 15px;">:</div>
        <h3 style="color: #1D1D1F; font-size: 18px; margin: 0 0 10px 0;">Predictive Analytics</h3>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            93%+ F1 Score churn prediction using LightGBM with optimized thresholds
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
        height: 200px;
    ">
        <div style="font-size: 40px; margin-bottom: 15px;">!</div>
        <h3 style="color: #1D1D1F; font-size: 18px; margin: 0 0 10px 0;">Explainable AI</h3>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            SHAP-powered explanations for every prediction with actionable insights
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
        height: 200px;
    ">
        <div style="font-size: 40px; margin-bottom: 15px;">$</div>
        <h3 style="color: #1D1D1F; font-size: 18px; margin: 0 0 10px 0;">Retention ROI</h3>
        <p style="color: #6E6E73; font-size: 14px; margin: 0;">
            Data-driven retention strategies with cost-benefit analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Quick stats
st.markdown("""
<div style="text-align: center; padding: 40px 0;">
    <p style="color: #86868B; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px;">
        Model Performance
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("F1 Score", "93.2%", "Churn Class")
with col2:
    st.metric("AUC-ROC", "99.4%", "Discrimination")
with col3:
    st.metric("Recall", "91.9%", "Churner Detection")
with col4:
    st.metric("Features", "31", "Engineered")

# Call to action
st.markdown("""
<div style="text-align: center; padding: 40px 0;">
    <p style="color: #6E6E73; font-size: 16px;">
        Navigate to <b>Dashboard</b> in the sidebar to get started
    </p>
</div>
""", unsafe_allow_html=True)
