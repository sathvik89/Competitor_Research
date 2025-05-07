import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from io import BytesIO
import base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from datetime import datetime
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Startup Competitor Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1565C0;
        margin-bottom: 1.2rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: #333333; 
        margin-bottom: 1.2rem;
    }

    .metric-card 
        # border-radius: 0.6rem;
        # padding: 1.8rem;
        # box-shadow: 0 0.2rem 0.4rem rgba(0, 0, 0, 0.1);
        # margin-bottom: 1.2rem;
        # border: 1px solid #e0e0e0;
        # transition: all 0.3s ease-in-out;
    }

    .metric-card:hover {
        box-shadow: 0 0.3rem 0.6rem rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1565C0; 
        line-height: 1.4;
    }

    .metric-label {
        font-size: 0.95rem;
        font-weight: 500;
        color: #5F5F5F; 
        margin-bottom: 0.3rem;
        display: block;
    }

    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /*Improve table readability in PDF export section */
    .dataframe {
        font-size: 0.9rem;
        color: #333;
    }

    .dataframe th {
        background-color: #f1f1f1;
        color: #1565C0;
        font-weight: 600;
    }

    .dataframe td {
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Function to parse and convert funding values
@st.cache_data
def parse_funding(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    # Extract numeric part
    match = re.search(r'(\d+\.?\d*)', value)
    if not match:
        return np.nan
    
    amount = float(match.group(1))
    
    # Determine multiplier
    if 'K' in value:
        return amount * 1_000
    elif 'M' in value:
        return amount * 1_000_000
    elif 'B' in value:
        return amount * 1_000_000_000
    else:
        return amount

# Function to parse team size
@st.cache_data
def parse_team_size(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    # Handle ranges like "51-200"
    if '-' in value:
        parts = value.split('-')
        # Take average of range
        return (float(re.search(r'(\d+)', parts[0]).group(1)) + 
                float(re.search(r'(\d+)', parts[1]).group(1))) / 2
    
    # Handle values with + like "800+"
    if '+' in value:
        return float(re.search(r'(\d+)', value).group(1))
    
    # Try to extract any number
    match = re.search(r'(\d+)', value)
    if match:
        return float(match.group(1))
    
    return np.nan

# Function to parse customer base
@st.cache_data
def parse_customer_base(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    # Handle values with + like "10000+"
    if '+' in value:
        base = re.search(r'(\d+)', value)
        if base:
            return float(base.group(1))
    
    # Handle values with M+ like "5M+"
    if 'M+' in value:
        base = re.search(r'(\d+)', value)
        if base:
            return float(base.group(1)) * 1_000_000
    
    # Try to extract any number
    match = re.search(r'(\d+)', value)
    if match:
        return float(match.group(1))
    
    return np.nan

# Function to parse revenue
@st.cache_data
def parse_revenue(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    # Extract numeric part
    match = re.search(r'(\d+\.?\d*)', value)
    if not match:
        return np.nan
    
    amount = float(match.group(1))
    
    # Determine multiplier
    if 'K' in value:
        return amount * 1_000
    elif 'M' in value:
        return amount * 1_000_000
    elif 'B' in value:
        return amount * 1_000_000_000
    else:
        return amount

# Load and clean data
@st.cache_data
def load_and_clean_data(csv_data):
    # Load data
    df = pd.read_csv(csv_data)
    
    # Replace "null" strings with NaN
    df = df.replace("null", np.nan)
    
    # Convert string columns to appropriate types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Founding Year'] = pd.to_numeric(df['Founding Year'], errors='coerce')
    df['Age (Years)'] = pd.to_numeric(df['Age (Years)'], errors='coerce')
    df['Number of Funding Rounds'] = pd.to_numeric(df['Number of Funding Rounds'], errors='coerce')
    
    # Parse funding values
    df['Funding_Numeric'] = df['Total Funding Raised (USD or INR)'].apply(parse_funding)
    
    # Parse team size
    df['Team_Size_Numeric'] = df['Team Size'].apply(parse_team_size)
    
    # Parse customer base
    df['Customer_Base_Numeric'] = df['Customer/User Base Size'].apply(parse_customer_base)
    
    # Parse revenue
    df['Revenue_Numeric'] = df['Revenue'].apply(parse_revenue)
    
    # Extract primary industry
    df['Primary_Industry'] = df['Industry/Category Tags'].apply(
        lambda x: x.split(';')[0] if isinstance(x, str) and ';' in x else x
    )
    
    # Group by industry to fill missing values with industry median
    industry_groups = df.groupby('Primary_Industry')
    
    # Fill missing values with industry median where possible
    for col in ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric', 'Revenue_Numeric']:
        df[col] = df.groupby('Primary_Industry')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    # For remaining NaNs, use global median
    for col in ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric', 'Revenue_Numeric']:
        df[col] = df[col].fillna(df[col].median())
    
    # Calculate maturity score (scale 5-25)
    # Normalize each component to 0-10 scale
    max_age = df['Age (Years)'].max()
    max_funding = df['Funding_Numeric'].max()
    max_team = df['Team_Size_Numeric'].max()
    
    df['Age_Score'] = df['Age (Years)'].apply(lambda x: min(10, (x / max_age) * 10))
    df['Funding_Score'] = df['Funding_Numeric'].apply(lambda x: min(10, (x / max_funding) * 10))
    df['Team_Score'] = df['Team_Size_Numeric'].apply(lambda x: min(10, (x / max_team) * 10))
    
    # Weighted maturity score (5-25 scale)
    df['Maturity_Score'] = 5 + (
        df['Age_Score'] * 0.3 +
        df['Funding_Score'] * 0.5 +
        df['Team_Score'] * 0.2
    )
    
    # Calculate funding efficiency
    # Group by company and get latest year's funding
    latest_funding = df.sort_values('Year', ascending=False).drop_duplicates('Company Name')['Funding_Numeric']
    company_total_funding = df.groupby('Company Name')['Funding_Numeric'].sum()
    
    # Create a mapping dictionary
    latest_funding_dict = latest_funding.to_dict()
    total_funding_dict = company_total_funding.to_dict()
    
    # Calculate funding efficiency (latest year's funding / total funding)
    df['Funding_Efficiency'] = df['Company Name'].apply(
        lambda x: latest_funding_dict.get(x, 0) / total_funding_dict.get(x, 1) 
        if total_funding_dict.get(x, 0) > 0 else 0
    )
    
    # Calculate revenue efficiency (revenue / funding)
    df['Revenue_Efficiency'] = df.apply(
        lambda row: row['Revenue_Numeric'] / row['Funding_Numeric'] 
        if row['Funding_Numeric'] > 0 and not pd.isna(row['Revenue_Numeric']) else 0,
        axis=1
    )
    
    return df

# Function to create plots
def create_time_series_plot(df, company, metric, title):
    company_data = df[df['Company Name'] == company].sort_values('Year')
    
    if company_data.empty or company_data[metric].isna().all():
        return go.Figure().update_layout(
            title=f"No {metric} data available for {company}",
            xaxis_title="Year",
            yaxis_title=metric
        )
    
    fig = px.line(
        company_data, 
        x='Year', 
        y=metric,
        markers=True,
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=metric,
        plot_bgcolor='white',
        hovermode="x unified"
    )
    
    return fig

def create_multi_metric_plot(df, company, metrics, title):
    company_data = df[df['Company Name'] == company].sort_values('Year')
    
    if company_data.empty:
        return go.Figure().update_layout(
            title=f"No data available for {company}",
            xaxis_title="Year"
        )
    
    fig = go.Figure()
    
    for metric in metrics:
        if not company_data[metric].isna().all():
            # Normalize values to make them comparable on same scale
            max_val = company_data[metric].max()
            normalized = company_data[metric] / max_val if max_val > 0 else company_data[metric]
            
            fig.add_trace(go.Scatter(
                x=company_data['Year'],
                y=normalized,
                mode='lines+markers',
                name=metric
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Normalized Value",
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_maturity_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Maturity Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [5, 25], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1E88E5"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [5, 12], 'color': '#ffcdd2'},
                {'range': [12, 19], 'color': '#fff9c4'},
                {'range': [19, 25], 'color': '#c8e6c9'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
    )
    
    return fig

def create_efficiency_bar_chart(df, company, metric, title):
    # Get industry of the selected company
    company_industry = df[df['Company Name'] == company]['Primary_Industry'].iloc[0]
    
    # Filter companies in the same industry
    industry_df = df[df['Primary_Industry'] == company_industry]
    
    # Get latest year data for each company
    latest_data = industry_df.sort_values('Year', ascending=False).drop_duplicates('Company Name')
    
    # Sort by the metric
    sorted_data = latest_data.sort_values(metric, ascending=False)
    
    # Take top 10 companies
    plot_data = sorted_data.head(10)
    
    # Create bar chart
    fig = px.bar(
        plot_data,
        x='Company Name',
        y=metric,
        color=metric,
        color_continuous_scale=px.colors.sequential.Blues,
        title=title
    )
    
    # Highlight the selected company
    for i, company_name in enumerate(plot_data['Company Name']):
        if company_name == company:
            fig.add_shape(
                type="rect",
                x0=i-0.4, x1=i+0.4,
                y0=0, y1=plot_data[plot_data['Company Name'] == company][metric].iloc[0],
                line=dict(color="red", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    
    fig.update_layout(
        xaxis_title="Company",
        yaxis_title=metric,
        plot_bgcolor='white',
        xaxis={'categoryorder': 'total descending'}
    )
    
    return fig

def create_customer_revenue_plot(df, company):
    company_data = df[df['Company Name'] == company].sort_values('Year')
    
    if company_data.empty or (company_data['Customer_Base_Numeric'].isna().all() and 
                             company_data['Revenue_Numeric'].isna().all()):
        return go.Figure().update_layout(
            title=f"No customer/revenue data available for {company}",
            xaxis_title="Year"
        )
    
    fig = go.Figure()
    
    # Add customer base line
    if not company_data['Customer_Base_Numeric'].isna().all():
        fig.add_trace(go.Scatter(
            x=company_data['Year'],
            y=company_data['Customer_Base_Numeric'],
            mode='lines+markers',
            name='Customer Base',
            line=dict(color='#1E88E5', width=3)
        ))
    
    # Add revenue line with secondary y-axis
    if not company_data['Revenue_Numeric'].isna().all():
        fig.add_trace(go.Scatter(
            x=company_data['Year'],
            y=company_data['Revenue_Numeric'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#43A047', width=3),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="Customer Base vs Revenue Over Time",
        xaxis_title="Year",
        yaxis_title="Customer Base",
        yaxis2=dict(
            title="Revenue (USD)",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        plot_bgcolor='white',
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Function to generate PDF report
def generate_pdf_report(df, company_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add title
    title_style = styles["Title"]
    elements.append(Paragraph(f"{company_name} - Startup Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Get company data
    company_data = df[df['Company Name'] == company_name].sort_values('Year', ascending=False)
    latest_data = company_data.iloc[0]
    
    # Add company overview
    elements.append(Paragraph("Company Overview", styles["Heading1"]))
    elements.append(Spacer(1, 10))
    
    overview_data = [
        ["Metric", "Value"],
        ["Industry", str(latest_data['Primary_Industry'])],
        ["Founding Year", str(int(latest_data['Founding Year']))],
        ["Age", f"{int(latest_data['Age (Years)'])} years"],
        ["Total Funding", f"${latest_data['Funding_Numeric']:,.0f}"],
        ["Team Size", f"{int(latest_data['Team_Size_Numeric']):,}"],
        ["Maturity Score", f"{latest_data['Maturity_Score']:.1f}/25"],
        ["Funding Efficiency", f"{latest_data['Funding_Efficiency']:.2f}"],
        ["Revenue Efficiency", f"{latest_data['Revenue_Efficiency']:.2f}"]
    ]
    
    overview_table = Table(overview_data, colWidths=[200, 300])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(overview_table)
    elements.append(Spacer(1, 20))
    
    # Add tech stack and product launches
    elements.append(Paragraph("Technology & Products", styles["Heading1"]))
    elements.append(Spacer(1, 10))
    
    tech_data = [
        ["Category", "Details"],
        ["Tech Stack / Core Capabilities", str(latest_data['Tech Stack / Core AI Capabilities'])],
        ["Recent Product Launches", str(latest_data['Product Launches / Updates (last 2 years)'])]
    ]
    
    tech_table = Table(tech_data, colWidths=[200, 300])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(tech_table)
    elements.append(Spacer(1, 20))
    
    # Add industry comparison
    elements.append(Paragraph("Industry Comparison", styles["Heading1"]))
    elements.append(Spacer(1, 10))
    
    # Get industry of the selected company
    company_industry = latest_data['Primary_Industry']
    
    # Filter companies in the same industry
    industry_df = df[df['Primary_Industry'] == company_industry]
    
    # Get latest year data for each company
    industry_latest = industry_df.sort_values('Year', ascending=False).drop_duplicates('Company Name')
    
    # Sort by maturity score
    # Rank only among companies in the filtered industry data (with year filter applied)
    industry_ranked = industry_latest.sort_values('Maturity_Score', ascending=False)
    industry_ranked['Rank_in_Industry'] = range(1, len(industry_ranked) + 1)

    # Get rank for selected company
    company_rank_row = industry_ranked[industry_ranked['Company Name'] == company_name]
    company_rank = int(company_rank_row['Rank_in_Industry'].values[0]) if not company_rank_row.empty else 'N/A'
    # Get rank of the selected company
    
    
    # Get top 5 companies in the industry
    top_companies = industry_ranked.head(5)
    
    elements.append(Paragraph(f"Rank in {company_industry} Industry based on Maturity Score: #{company_rank}", styles["Normal"]))
    elements.append(Spacer(1, 10))
    
    # Add top companies table
    elements.append(Paragraph("Top Companies in Industry", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    
    top_data = [["Rank", "Company", "Maturity Score", "Funding"]]
    
    for i, (_, row) in enumerate(top_companies.iterrows(), 1):
        top_data.append([
            str(i),
            row['Company Name'],
            f"{row['Maturity_Score']:.1f}",
            f"${row['Funding_Numeric']:,.0f}"
        ])
    
    top_table = Table(top_data, colWidths=[50, 200, 100, 150])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (3, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (3, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (3, 0), 12),
        ('BOTTOMPADDING', (0, 0), (3, 0), 12),
        ('BACKGROUND', (0, 1), (3, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(top_table)
    elements.append(Spacer(1, 20))
    
    # Add footer with generation date
    footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(footer_text, styles["Normal"]))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer

# Main app
def main():
    st.markdown('<div class="main-header">🚀 Startup Competitor Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('Analyze and compare Indian startups across funding, maturity, and industry metrics.')
    
    # Load data
    df = load_and_clean_data("company_competitor_data.csv")
    
    # Sidebar filters
    st.sidebar.markdown("## Filters")
    
    # Industry filter
    industries = sorted(df['Primary_Industry'].unique())
    selected_industry = st.sidebar.selectbox("Select Industry", ["All"] + list(industries))
    
    # Filter by industry if selected
    if selected_industry != "All":
        filtered_df = df[df['Primary_Industry'] == selected_industry]
    else:
        filtered_df = df
    
    # Company selection
    companies = sorted(filtered_df['Company Name'].unique())
    selected_company = st.sidebar.selectbox("Select Company", companies)
    
    # Year range filter
    years = sorted(df['Year'].unique())
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(min(years)),
        max_value=int(max(years)),
        value=(int(min(years)), int(max(years)))
    )
    
    # Apply year filter
    filtered_df = filtered_df[(filtered_df['Year'] >= year_range[0]) & (filtered_df['Year'] <= year_range[1])]
    
    # Get company data
    company_data = filtered_df[filtered_df['Company Name'] == selected_company]
    
    if company_data.empty:
        st.error(f"No data available for {selected_company} in the selected year range.")
        return
    
    # Latest data for the selected company
    latest_data = company_data.sort_values('Year', ascending=False).iloc[0]
    
    # Company overview section
    st.markdown('<div class="sub-header">📊 Company Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Industry</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{latest_data['Primary_Industry']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Founded</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{int(latest_data['Founding Year'])}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Total Funding</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>${latest_data['Funding_Numeric']:,.0f}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>Team Size</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{int(latest_data['Team_Size_Numeric']):,}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Maturity score gauge
        fig = create_maturity_gauge(latest_data['Maturity_Score'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Behavior over time section
    st.markdown('<div class="sub-header">📈 Performance Metrics Over Time</div>', unsafe_allow_html=True)
    
    metric_tabs = st.tabs(["Funding", "Team Size", "Customer Base", "Multi-Metric", "Customer vs Revenue"])
    
    with metric_tabs[0]:
        funding_fig = create_time_series_plot(
            company_data, 
            selected_company, 
            'Funding_Numeric', 
            'Funding Over Time'
        )
        st.plotly_chart(funding_fig, use_container_width=True)
    
    with metric_tabs[1]:
        team_fig = create_time_series_plot(
            company_data, 
            selected_company, 
            'Team_Size_Numeric', 
            'Team Size Over Time'
        )
        st.plotly_chart(team_fig, use_container_width=True)
    
    with metric_tabs[2]:
        customer_fig = create_time_series_plot(
            company_data, 
            selected_company, 
            'Customer_Base_Numeric', 
            'Customer Base Over Time'
        )
        st.plotly_chart(customer_fig, use_container_width=True)
    
    with metric_tabs[3]:
        multi_fig = create_multi_metric_plot(
            company_data,
            selected_company,
            ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric'],
            'Multiple Metrics Over Time (Normalized)'
        )
        st.plotly_chart(multi_fig, use_container_width=True)
    
    with metric_tabs[4]:
        customer_revenue_fig = create_customer_revenue_plot(company_data, selected_company)
        st.plotly_chart(customer_revenue_fig, use_container_width=True)
    
    # Efficiency metrics section
    st.markdown('<div class="sub-header">⚡ Efficiency Metrics</div>', unsafe_allow_html=True)
    
    efficiency_tabs = st.tabs(["Funding Efficiency", "Revenue Efficiency"])
    
    with efficiency_tabs[0]:
        funding_eff_fig = create_efficiency_bar_chart(
            filtered_df,
            selected_company,
            'Funding_Efficiency',
            'Funding Efficiency Comparison (Higher is Better)'
        )
        st.plotly_chart(funding_eff_fig, use_container_width=True)
    
    with efficiency_tabs[1]:
        revenue_eff_fig = create_efficiency_bar_chart(
            filtered_df,
            selected_company,
            'Revenue_Efficiency',
            'Revenue Efficiency Comparison (Higher is Better)'
        )
        st.plotly_chart(revenue_eff_fig, use_container_width=True)
    
    # Industry comparison section
    st.markdown('<div class="sub-header">🏢 Industry Comparison</div>', unsafe_allow_html=True)
    
    # Get industry of the selected company
    company_industry = latest_data['Primary_Industry']
    
    # Filter companies in the same industry
    industry_df = filtered_df[filtered_df['Primary_Industry'] == company_industry]
    
    # Get latest year data for each company
    industry_latest = industry_df.sort_values('Year', ascending=False).drop_duplicates('Company Name')
    
    # Sort by maturity score
    industry_ranked = industry_latest.sort_values('Maturity_Score', ascending=False)
    
    # Create bar chart for industry comparison
    industry_fig = px.bar(
        industry_ranked,
        x='Company Name',
        y='Maturity_Score',
        color='Maturity_Score',
        color_continuous_scale=px.colors.sequential.Blues,
        title=f'Maturity Score Comparison in {company_industry} Industry'
    )
    
    # Highlight the selected company
    for i, company_name in enumerate(industry_ranked['Company Name']):
        if company_name == selected_company:
            industry_fig.add_shape(
                type="rect",
                x0=i-0.4, x1=i+0.4,
                y0=0, y1=industry_ranked[industry_ranked['Company Name'] == selected_company]['Maturity_Score'].iloc[0],
                line=dict(color="red", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    
    industry_fig.update_layout(
        xaxis_title="Company",
        yaxis_title="Maturity Score",
        plot_bgcolor='white',
        xaxis={'categoryorder': 'total descending'}
    )
    
    st.plotly_chart(industry_fig, use_container_width=True)
    
    # PDF Report Generation
    st.markdown('<div class="sub-header">📄 PDF Report</div>', unsafe_allow_html=True)
    
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_buffer = generate_pdf_report(df, selected_company)
            
            # Create download link
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{selected_company}_analysis.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success("PDF report generated successfully!")

if __name__ == "__main__":
    main()