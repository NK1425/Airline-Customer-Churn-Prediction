"""
UI Components for SkyGuard Streamlit App
Reusable styled components with Apple-inspired design.
"""

from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_css():
    """Load custom CSS styles."""
    css_path = get_project_root() / "assets" / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Color palette
COLORS = {
    'primary': '#0071E3',
    'green': '#34C759',
    'red': '#FF3B30',
    'orange': '#FF9500',
    'gray': '#86868B',
    'dark': '#1D1D1F',
    'light': '#F5F5F7',
    'text_secondary': '#6E6E73',
}

TIER_COLORS = {
    'Star': '#86868B',
    'Nova': '#0071E3',
    'Aurora': '#AF52DE',
}

CHURN_COLORS = {
    'Active': '#34C759',
    'Churned': '#FF3B30',
    0: '#34C759',
    1: '#FF3B30',
}

RISK_COLORS = {
    'High': '#FF3B30',
    'Medium': '#FF9500',
    'Low': '#34C759',
}

# Chart layout template
CHART_LAYOUT = dict(
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", color="#1D1D1F"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False),
    hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#D2D2D7"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def metric_card(label: str, value: str, delta: str = None, delta_color: str = "green", icon: str = ""):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "#34C759" if delta_color == "green" else "#FF3B30"
        arrow = "^" if delta_color == "green" else "v"
        delta_html = f'<p style="color:{color}; font-size:14px; margin:0;">{arrow} {delta}</p>'

    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    ">
        <p style="color:#86868B; font-size:13px; margin:0 0 8px 0; text-transform:uppercase; letter-spacing:0.5px;">{icon} {label}</p>
        <p style="color:#1D1D1F; font-size:36px; font-weight:700; margin:0; letter-spacing:-0.5px;">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = None):
    """Create a section header."""
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="color:#1D1D1F; font-size:1.5rem; font-weight:600; margin:0; letter-spacing:-0.02em;">{title}</h2>
        {f'<p style="color:#6E6E73; font-size:0.95rem; margin:0.5rem 0 0 0;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def gradient_divider():
    """Create a gradient divider."""
    st.markdown("""
    <div style="
        height: 2px;
        background: linear-gradient(90deg, #0071E3, #34C759, #0071E3);
        border-radius: 1px;
        margin: 2rem 0;
    "></div>
    """, unsafe_allow_html=True)


def risk_badge(risk_level: str):
    """Create a risk level badge."""
    colors = {
        'High': ('#FF3B30', 'rgba(255, 59, 48, 0.1)'),
        'Medium': ('#FF9500', 'rgba(255, 149, 0, 0.1)'),
        'Low': ('#34C759', 'rgba(52, 199, 89, 0.1)'),
    }
    color, bg = colors.get(risk_level, ('#86868B', 'rgba(134, 134, 139, 0.1)'))

    return f"""
    <span style="
        background-color: {bg};
        color: {color};
        padding: 4px 12px;
        border-radius: 980px;
        font-weight: 500;
        font-size: 14px;
    ">{risk_level} Risk</span>
    """


def tier_badge(tier: str):
    """Create a tier badge."""
    colors = {
        'Star': ('#86868B', 'rgba(134, 134, 139, 0.1)'),
        'Nova': ('#0071E3', 'rgba(0, 113, 227, 0.1)'),
        'Aurora': ('#AF52DE', 'rgba(175, 82, 222, 0.1)'),
    }
    color, bg = colors.get(tier, ('#86868B', 'rgba(134, 134, 139, 0.1)'))

    return f"""
    <span style="
        background-color: {bg};
        color: {color};
        padding: 4px 12px;
        border-radius: 980px;
        font-weight: 500;
        font-size: 14px;
    ">{tier}</span>
    """


def progress_ring(percentage: float, size: int = 120, color: str = None):
    """Create a progress ring visualization."""
    if color is None:
        if percentage >= 60:
            color = COLORS['red']
        elif percentage >= 30:
            color = COLORS['orange']
        else:
            color = COLORS['green']

    # Create SVG progress ring
    stroke_width = 8
    radius = (size - stroke_width) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference - (percentage / 100) * circumference

    return f"""
    <div style="display: flex; flex-direction: column; align-items: center;">
        <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
            <circle
                cx="{size/2}" cy="{size/2}" r="{radius}"
                fill="none" stroke="#E5E5EA" stroke-width="{stroke_width}"
            />
            <circle
                cx="{size/2}" cy="{size/2}" r="{radius}"
                fill="none" stroke="{color}" stroke-width="{stroke_width}"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                stroke-linecap="round"
                style="transition: stroke-dashoffset 0.5s ease;"
            />
        </svg>
        <div style="
            position: relative;
            top: -{size/2 + 15}px;
            text-align: center;
        ">
            <span style="font-size: 28px; font-weight: 700; color: {color};">{percentage:.0f}%</span>
        </div>
    </div>
    """


def customer_card(customer_data: dict):
    """Create a customer profile card."""
    tier = customer_data.get('Loyalty Card', customer_data.get('loyalty_tier', 'Star'))
    tier_badge_html = tier_badge(tier)

    clv = customer_data.get('CLV', 0)
    tenure = customer_data.get('tenure_months', 0)

    st.markdown(f"""
    <div style="
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #E5E5EA;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <h3 style="margin: 0; color: #1D1D1F; font-size: 18px;">
                Customer #{customer_data.get('Loyalty Number', 'N/A')}
            </h3>
            {tier_badge_html}
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">CLV</p>
                <p style="color: #1D1D1F; font-size: 16px; font-weight: 600; margin: 0;">${clv:,.2f}</p>
            </div>
            <div>
                <p style="color: #86868B; font-size: 12px; margin: 0;">Tenure</p>
                <p style="color: #1D1D1F; font-size: 16px; font-weight: 600; margin: 0;">{int(tenure)} months</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def styled_plotly_chart(fig, use_container_width: bool = True):
    """Apply consistent styling to a Plotly chart and display it."""
    fig.update_layout(**CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=use_container_width)


def create_donut_chart(values: list, labels: list, colors: list = None, title: str = ""):
    """Create a styled donut chart."""
    if colors is None:
        colors = [COLORS['primary'], COLORS['green'], COLORS['orange']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo='percent+label',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#1D1D1F')),
        showlegend=False,
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ['xaxis', 'yaxis']}
    )

    return fig


def create_bar_chart(x, y, colors=None, title: str = "", orientation: str = 'v',
                     x_title: str = "", y_title: str = ""):
    """Create a styled bar chart."""
    if colors is None:
        colors = COLORS['primary']

    fig = go.Figure(data=[go.Bar(
        x=x if orientation == 'v' else y,
        y=y if orientation == 'v' else x,
        marker_color=colors,
        orientation=orientation,
        hovertemplate='<b>%{x}</b>: %{y}<extra></extra>' if orientation == 'v' else '<b>%{y}</b>: %{x}<extra></extra>'
    )])

    layout = CHART_LAYOUT.copy()
    layout['title'] = dict(text=title, font=dict(size=16, color='#1D1D1F'))
    layout['xaxis']['title'] = x_title
    layout['yaxis']['title'] = y_title

    fig.update_layout(**layout)

    return fig


def create_line_chart(x, y_data: dict, title: str = "", x_title: str = "", y_title: str = ""):
    """Create a styled line chart with multiple series."""
    fig = go.Figure()

    for name, y in y_data.items():
        color = CHURN_COLORS.get(name, COLORS['primary'])
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f'<b>{name}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>'
        ))

    layout = CHART_LAYOUT.copy()
    layout['title'] = dict(text=title, font=dict(size=16, color='#1D1D1F'))
    layout['xaxis']['title'] = x_title
    layout['yaxis']['title'] = y_title

    fig.update_layout(**layout)

    return fig


def create_heatmap(z, x, y, title: str = "", colorscale: str = "RdYlGn_r"):
    """Create a styled heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        hovertemplate='<b>%{y}</b> x <b>%{x}</b><br>Value: %{z}<extra></extra>'
    ))

    layout = CHART_LAYOUT.copy()
    layout['title'] = dict(text=title, font=dict(size=16, color='#1D1D1F'))

    fig.update_layout(**layout)

    return fig


def create_histogram(data, nbins: int = 30, title: str = "", x_title: str = "",
                     color: str = None, show_mean: bool = True):
    """Create a styled histogram."""
    if color is None:
        color = COLORS['primary']

    fig = go.Figure(data=[go.Histogram(
        x=data,
        nbinsx=nbins,
        marker_color=color,
        opacity=0.8,
        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
    )])

    if show_mean:
        mean_val = data.mean()
        fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS['red'],
                      annotation_text=f"Mean: {mean_val:.2f}")

    layout = CHART_LAYOUT.copy()
    layout['title'] = dict(text=title, font=dict(size=16, color='#1D1D1F'))
    layout['xaxis']['title'] = x_title
    layout['yaxis']['title'] = 'Count'

    fig.update_layout(**layout)

    return fig


def info_box(text: str, icon: str = "i"):
    """Create an info box."""
    st.markdown(f"""
    <div style="
        background: rgba(0, 113, 227, 0.1);
        border-left: 4px solid #0071E3;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    ">
        <p style="color: #0071E3; margin: 0; font-size: 14px;">
            <strong>{icon}</strong> {text}
        </p>
    </div>
    """, unsafe_allow_html=True)


def warning_box(text: str):
    """Create a warning box."""
    st.markdown(f"""
    <div style="
        background: rgba(255, 149, 0, 0.1);
        border-left: 4px solid #FF9500;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    ">
        <p style="color: #FF9500; margin: 0; font-size: 14px;">
            <strong>!</strong> {text}
        </p>
    </div>
    """, unsafe_allow_html=True)


def success_box(text: str):
    """Create a success box."""
    st.markdown(f"""
    <div style="
        background: rgba(52, 199, 89, 0.1);
        border-left: 4px solid #34C759;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    ">
        <p style="color: #34C759; margin: 0; font-size: 14px;">
            <strong>+</strong> {text}
        </p>
    </div>
    """, unsafe_allow_html=True)
