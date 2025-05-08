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
        font-size: 1.8rem;
        font-weight: 800;
        color: #1565C0; 
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #1565C0;
        padding-bottom: 0.5rem;
    }

    .metric-card {
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
    
    match = re.search(r'(\d+\.?\d*)', value)
    if not match:
        return np.nan
    
    amount = float(match.group(1))
    
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
def parseTeamSize(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    if '-' in value:
        parts = value.split('-')
        return (float(re.search(r'(\d+)', parts[0]).group(1)) + 
                float(re.search(r'(\d+)', parts[1]).group(1))) / 2
    
    if '+' in value:
        return float(re.search(r'(\d+)', value).group(1))
    
    match = re.search(r'(\d+)', value)
    if match:
        return float(match.group(1))
    
    return np.nan

# Function to parse customer base
@st.cache_data
def parse_customer_base(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    if '+' in value:
        base = re.search(r'(\d+)', value)
        if base:
            return float(base.group(1))
    
    if 'M+' in value:
        base = re.search(r'(\d+)', value)
        if base:
            return float(base.group(1)) * 1_000_000
    
    match = re.search(r'(\d+)', value)
    if match:
        return float(match.group(1))
    
    return np.nan

# Function to parse revenue
@st.cache_data
def parseRevenue(value):
    if pd.isna(value) or value == "null" or not isinstance(value, str):
        return np.nan
    
    match = re.search(r'(\d+\.?\d*)', value)
    if not match:
        return np.nan
    
    amount = float(match.group(1))
    
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
def loadAndCleanData(csv_data):
    df = pd.read_csv(csv_data)
    
    df = df.replace("null", np.nan)
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Founding Year'] = pd.to_numeric(df['Founding Year'], errors='coerce')
    df['Age (Years)'] = pd.to_numeric(df['Age (Years)'], errors='coerce')
    df['Number of Funding Rounds'] = pd.to_numeric(df['Number of Funding Rounds'], errors='coerce')
    
    df['Funding_Numeric'] = df['Total Funding Raised (USD or INR)'].apply(parse_funding)
    df['Team_Size_Numeric'] = df['Team Size'].apply(parseTeamSize)
    df['Customer_Base_Numeric'] = df['Customer/User Base Size'].apply(parse_customer_base)
    df['Revenue_Numeric'] = df['Revenue'].apply(parseRevenue)
    
    df['Primary_Industry'] = df['Industry/Category Tags'].apply(
        lambda x: x.split(';')[0] if isinstance(x, str) and ';' in x else x
    )
    
    industry_groups = df.groupby('Primary_Industry')
    
    for col in ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric', 'Revenue_Numeric']:
        df[col] = df.groupby('Primary_Industry')[col].transform(
            lambda x: x.fillna(x.median())
        )
    
    for col in ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric', 'Revenue_Numeric']:
        df[col] = df[col].fillna(df[col].median())
    
    max_age = df['Age (Years)'].max()
    max_funding = df['Funding_Numeric'].max()
    max_team = df['Team_Size_Numeric'].max()
    
    df['Age_Score'] = df['Age (Years)'].apply(lambda x: min(10, (x / max_age) * 10))
    df['Funding_Score'] = df['Funding_Numeric'].apply(lambda x: min(10, (x / max_funding) * 10))
    df['Team_Score'] = df['Team_Size_Numeric'].apply(lambda x: min(10, (x / max_team) * 10))
    
    df['Maturity_Score'] = 5 + (
        df['Age_Score'] * 0.3 +
        df['Funding_Score'] * 0.5 +
        df['Team_Score'] * 0.2
    )
    
    latest_funding = df.sort_values('Year', ascending=False).drop_duplicates('Company Name')['Funding_Numeric']
    company_total_funding = df.groupby('Company Name')['Funding_Numeric'].sum()
    
    latest_funding_dict = latest_funding.to_dict()
    total_funding_dict = company_total_funding.to_dict()
    
    df['Funding_Efficiency'] = df['Company Name'].apply(
        lambda x: latest_funding_dict.get(x, 0) / total_funding_dict.get(x, 1) 
        if total_funding_dict.get(x, 0) > 0 else 0
    )
    
    df['Revenue_Efficiency'] = df.apply(
        lambda row: row['Revenue_Numeric'] / row['Funding_Numeric'] 
        if row['Funding_Numeric'] > 0 and not pd.isna(row['Revenue_Numeric']) else 0,
        axis=1
    )
    
    return df

# Function to create plots
def createTimeSeriesPlot(df, company, metric, title):
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

def createMultiMetricPlot(df, company, metrics, title):
    company_data = df[df['Company Name'] == company].sort_values('Year')
    
    if company_data.empty:
        return go.Figure().update_layout(
            title=f"No data available for {company}",
            xaxis_title="Year"
        )
    
    fig = go.Figure()
    
    for metric in metrics:
        if not company_data[metric].isna().all():
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

def createMaturityGauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Maturity Score", 'font': {'size': 24, 'color': 'black'}},
        gauge={
            'axis': {'range': [5, 25], 'tickwidth': 1.5, 'tickcolor': "black", 'tickfont': {'size': 14, 'color': 'black'}},
            'bar': {'color': "#1E88E5", 'thickness': 0.7},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [5, 12], 'color': '#ff6b6b'},
                {'range': [12, 19], 'color': '#ffda77'},
                {'range': [19, 25], 'color': '#4caf50'}
            ],
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
    )
    
    return fig

def createEfficiencyBarChart(df, company, metric, title):
    company_industry = df[df['Company Name'] == company]['Primary_Industry'].iloc[0]
    industry_df = df[df['Primary_Industry'] == company_industry]
    latest_data = industry_df.sort_values('Year', ascending=False).drop_duplicates('Company Name')
    sorted_data = latest_data.sort_values(metric, ascending=False)
    plot_data = sorted_data.head(10)
    
    fig = px.bar(
        plot_data,
        x='Company Name',
        y=metric,
        color=metric,
        color_continuous_scale=px.colors.sequential.Blues,
        title=title
    )
    
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

def createCustomerRevenuePlot(df, company):
    company_data = df[df['Company Name'] == company].sort_values('Year')
    
    if company_data.empty or (company_data['Customer_Base_Numeric'].isna().all() and 
                             company_data['Revenue_Numeric'].isna().all()):
        return go.Figure().update_layout(
            title=f"No customer/revenue data available for {company}",
            xaxis_title="Year"
        )
    
    fig = go.Figure()
    
    if not company_data['Customer_Base_Numeric'].isna().all():
        fig.add_trace(go.Scatter(
            x=company_data['Year'],
            y=company_data['Customer_Base_Numeric'],
            mode='lines+markers',
            name='Customer Base',
            line=dict(color='#1E88E5', width=3)
        ))
    
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
def generatePdfReport(df, company_name):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    title_style = styles["Title"]
    heading1_style = styles["Heading1"]
    heading2_style = styles["Heading2"]
    normal_style = styles["Normal"]

    elements.append(Paragraph(f"Competitor Analysis Report for {company_name}", title_style))
    elements.append(Spacer(1, 20))

    company_data = df[df['Company Name'] == company_name].sort_values('Year', ascending=False)
    latest_data = company_data.iloc[0]
    company_industry = latest_data['Primary_Industry']

    industry_df = df[df['Primary_Industry'] == company_industry]
    industry_latest = industry_df.sort_values('Year', ascending=False).drop_duplicates('Company Name')
    industry_ranked = industry_latest.sort_values('Maturity_Score', ascending=False)
    competitors = industry_ranked[industry_ranked['Company Name'] != company_name].head(5)

    company_rank = industry_ranked['Company Name'].tolist().index(company_name) + 1
    total_companies = len(industry_ranked)

    elements.append(Paragraph("1. Executive Summary", heading1_style))
    elements.append(Spacer(1, 10))

    summary_data = [
        ["Metric", "Value"],
        ["Company Name", company_name],
        ["Industry", str(latest_data['Primary_Industry'])],
        ["Founding Year", str(int(latest_data['Founding Year']))],
        ["Industry Rank", f"#{company_rank} of {total_companies}"],
        ["Maturity Score", f"{latest_data['Maturity_Score']:.1f}/25"],
        ["Total Funding", f"${latest_data['Funding_Numeric']:,.0f}"],
        ["Team Size", f"{int(latest_data['Team_Size_Numeric']):,}"]
    ]

    summary_table = Table(summary_data, colWidths=[200, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("2. Competitor Overview", heading1_style))
    elements.append(Spacer(1, 10))

    for _, competitor in competitors.iterrows():
        elements.append(Paragraph(f"Competitor: {competitor['Company Name']}", heading2_style))
        comp_rank = industry_ranked['Company Name'].tolist().index(competitor['Company Name']) + 1
        comp_data = [
            ["Metric", "Value"],
            ["Industry", str(competitor['Primary_Industry'])],
            ["Industry Rank", f"#{comp_rank} of {total_companies}"],
            ["Founding Year", str(int(competitor['Founding Year']))],
            ["Funding", f"${competitor['Funding_Numeric']:,.0f}"],
            ["Team Size", f"{int(competitor['Team_Size_Numeric']):,}"],
            ["Maturity Score", f"{competitor['Maturity_Score']:.1f}/25"],
            ["Revenue Efficiency", f"{competitor['Revenue_Efficiency']:.2f}"]
        ]

        comp_table = Table(comp_data, colWidths=[200, 300])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(comp_table)
        elements.append(Spacer(1, 15))

    elements.append(Paragraph("3. Comparison Table", heading1_style))
    elements.append(Spacer(1, 10))

    comparison_data = [["Company", "Rank", "Funding", "Team Size", "Customer Base", "Revenue", "Maturity Score"]]
    comparison_data.append([
        company_name,
        f"#{company_rank}",
        f"${latest_data['Funding_Numeric']:,.0f}",
        f"{int(latest_data['Team_Size_Numeric']):,}",
        f"{int(latest_data['Customer_Base_Numeric']):,}" if not pd.isna(latest_data['Customer_Base_Numeric']) else "N/A",
        f"${latest_data['Revenue_Numeric']:,.0f}" if not pd.isna(latest_data['Revenue_Numeric']) else "N/A",
        f"{latest_data['Maturity_Score']:.1f}"
    ])
    for _, competitor in competitors.iterrows():
        comp_rank = industry_ranked['Company Name'].tolist().index(competitor['Company Name']) + 1
        comparison_data.append([
            competitor['Company Name'],
            f"#{comp_rank}",
            f"${competitor['Funding_Numeric']:,.0f}",
            f"{int(competitor['Team_Size_Numeric']):,}",
            f"{int(competitor['Customer_Base_Numeric']):,}" if not pd.isna(competitor['Customer_Base_Numeric']) else "N/A",
            f"${competitor['Revenue_Numeric']:,.0f}" if not pd.isna(competitor['Revenue_Numeric']) else "N/A",
            f"{competitor['Maturity_Score']:.1f}"
        ])

    comparison_table = Table(comparison_data, colWidths=[90, 40, 70, 70, 70, 70, 70])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (6, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (6, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (6, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (6, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (6, 0), 12),
        ('BOTTOMPADDING', (0, 0), (6, 0), 12),
        ('BACKGROUND', (0, 1), (6, 1), colors.lightgreen),
        ('BACKGROUND', (0, 2), (6, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(comparison_table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("4. Visual Comparison Chart", heading1_style))
    elements.append(Spacer(1, 10))

    chart_companies = pd.concat([
        pd.DataFrame({'Company': [company_name], 'Maturity_Score': [latest_data['Maturity_Score']], 'Rank': [company_rank]}),
        pd.DataFrame({'Company': competitors['Company Name'],
                      'Maturity_Score': competitors['Maturity_Score'],
                      'Rank': [industry_ranked['Company Name'].tolist().index(comp) + 1 for comp in competitors['Company Name']]})
    ])
    chart_companies = chart_companies.sort_values('Maturity_Score', ascending=False)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(chart_companies['Company'], chart_companies['Maturity_Score'],
                   color=['green' if x == company_name else 'blue' for x in chart_companies['Company']])
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"Rank #{chart_companies['Rank'].iloc[i]}", ha='center', fontweight='bold')
    plt.title('Maturity Scores by Company')
    plt.xlabel('Company')
    plt.ylabel('Maturity Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(chart_companies['Maturity_Score']) * 1.2)
    plt.tight_layout()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    img = Image(img_buffer, width=450, height=250)
    elements.append(img)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("5. Summary Insight", heading1_style))
    elements.append(Spacer(1, 10))

    industry_competitors = competitors[competitors['Primary_Industry'] == company_industry].copy()

    all_companies = pd.concat([
        pd.DataFrame({
            'Company': [company_name],
            'Funding': [latest_data['Funding_Numeric']],
            'Maturity': [latest_data['Maturity_Score']],
            'Revenue_Efficiency': [latest_data['Revenue_Efficiency']]
        }),
        industry_competitors[['Company Name', 'Funding_Numeric', 'Maturity_Score', 'Revenue_Efficiency']].rename(columns={
            'Company Name': 'Company',
            'Funding_Numeric': 'Funding',
            'Maturity_Score': 'Maturity'
        })
    ], ignore_index=True)

    all_companies['Company_clean'] = all_companies['Company'].str.strip().str.lower()
    company_clean = company_name.strip().lower()

    funding_sorted = all_companies.sort_values('Funding', ascending=False).reset_index(drop=True)
    maturity_sorted = all_companies.sort_values('Maturity', ascending=False).reset_index(drop=True)

    funding_rank = funding_sorted[funding_sorted['Company_clean'] == company_clean].index[0] + 1
    maturity_rank = maturity_sorted[maturity_sorted['Company_clean'] == company_clean].index[0] + 1
    total_companies = len(all_companies)

    insights = []

    if company_rank == 1:
        insights.append(f"• {company_name} is the top-ranked company in the {company_industry} industry based on maturity score.")
    elif company_rank <= 3:
        insights.append(f"• {company_name} is ranked #{company_rank} in the {company_industry} industry, placing it among the top performers.")
    else:
        insights.append(f"• {company_name} is ranked #{company_rank} out of {total_companies} companies in the {company_industry} industry.")

    if funding_rank == 1:
        insights.append(f"• In terms of funding, {company_name} leads all competitors in the {company_industry} industry.")
    else:
        top_funders = ", ".join(funding_sorted['Company'].iloc[:funding_rank-1])
        insights.append(f"• In terms of funding, {company_name} ranks #{funding_rank} behind {top_funders}.")

    if maturity_rank == 1:
        insights.append(f"• {company_name} has the highest maturity score among {company_industry} companies analyzed.")
    else:
        top_mature = ", ".join(maturity_sorted['Company'].iloc[:maturity_rank-1])
        insights.append(f"• For maturity, {company_name} ranks #{maturity_rank} behind {top_mature}.")

    if not pd.isna(latest_data['Revenue_Efficiency']):
        better_efficiency = all_companies[
            (all_companies['Company_clean'] != company_clean) &
            (all_companies['Revenue_Efficiency'] > latest_data['Revenue_Efficiency'])
        ]
        if len(better_efficiency) == 0:
            insights.append(f"• {company_name} leads in revenue efficiency compared to all {company_industry} competitors.")
        else:
            insights.append(f"• {len(better_efficiency)} {company_industry} competitors have better revenue efficiency than {company_name}.")

    for insight in insights:
        elements.append(Paragraph(insight, normal_style))
        elements.append(Spacer(1, 5))

    elements.append(Spacer(1, 15))
    elements.append(Paragraph("6. Appendix", heading1_style))
    elements.append(Spacer(1, 10))

    appendix_data = [
        ["Category", "Details"],
        ["Tech Stack / Core Capabilities", str(latest_data['Tech Stack / Core AI Capabilities'])],
        ["Recent Product Launches", str(latest_data['Product Launches / Updates (last 2 years)'])]
    ]

    appendix_table = Table(appendix_data, colWidths=[200, 300])
    appendix_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(appendix_table)
    elements.append(Spacer(1, 20))

    footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(footer_text, normal_style))

    doc.build(elements)
    buffer.seek(0)

    return buffer

# Main app
def main():
    st.markdown('<div class="main-header">🚀 Startup Competitor Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown('Analyze and compare Indian startups across funding, maturity, and industry metrics.')
    
    # Load data
    df = loadAndCleanData("company_competitor_data.csv")
    
    # Sidebar filters
    st.sidebar.markdown("## Filters")
    
    # Industry filter
    industries = sorted(df['Primary_Industry'].unique())
    selected_industry = st.sidebar.selectbox("Select Industry", ["All"] + list(industries), index=industries.index("PropTech")+1 if "PropTech" in industries else 0)
    
    # Filter by industry if selected
    if selected_industry != "All":
        filtered_df = df[df['Primary_Industry'] == selected_industry]
    else:
        filtered_df = df
    
    # Company selection
    companies = sorted(filtered_df['Company Name'].unique())
    default_company_index = companies.index("Propacity") if "Propacity" in companies else 0
    selected_company = st.sidebar.selectbox("Select Company", companies, index=default_company_index)
    
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
        # Maturity score gauge with improved contrast
        fig = createMaturityGauge(latest_data['Maturity_Score'])
        # Improve gauge contrast
        fig.update_traces(
            gauge={
                'axis': {'range': [5, 25], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "#1E88E5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "black",
                'steps': [
                    {'range': [5, 12], 'color': '#ff6b6b'},
                    {'range': [12, 19], 'color': '#ffda77'},
                    {'range': [19, 25], 'color': '#4caf50'}
                ],
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # PDF Report Generation - Placed after company overview
    st.markdown('<div class="sub-header">📄 PDF Report</div>', unsafe_allow_html=True)
    
    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_buffer = generatePdfReport(df, selected_company)
            
            # Create download link
            b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="{selected_company}_analysis.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.success("PDF report generated successfully!")
    
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
    
    # Behavior over time section
    st.markdown('<div class="sub-header">📈 Performance Metrics Over Time</div>', unsafe_allow_html=True)
    
    metric_tabs = st.tabs(["Funding", "Team Size", "Customer Base", "Multi-Metric", "Customer vs Revenue"])
    
    with metric_tabs[0]:
        funding_fig = createTimeSeriesPlot(
            company_data, 
            selected_company, 
            'Funding_Numeric', 
            'Funding Over Time'
        )
        st.plotly_chart(funding_fig, use_container_width=True)
    
    with metric_tabs[1]:
        team_fig = createTimeSeriesPlot(
            company_data, 
            selected_company, 
            'Team_Size_Numeric', 
            'Team Size Over Time'
        )
        st.plotly_chart(team_fig, use_container_width=True)
    
    with metric_tabs[2]:
        customer_fig = createTimeSeriesPlot(
            company_data, 
            selected_company, 
            'Customer_Base_Numeric', 
            'Customer Base Over Time'
        )
        st.plotly_chart(customer_fig, use_container_width=True)
    
    with metric_tabs[3]:
        multi_fig = createMultiMetricPlot(
            company_data,
            selected_company,
            ['Funding_Numeric', 'Team_Size_Numeric', 'Customer_Base_Numeric'],
            'Multiple Metrics Over Time (Normalized)'
        )
        st.plotly_chart(multi_fig, use_container_width=True)
    
    with metric_tabs[4]:
        customer_revenue_fig = createCustomerRevenuePlot(company_data, selected_company)
        st.plotly_chart(customer_revenue_fig, use_container_width=True)
    
    # Efficiency metrics section
    st.markdown('<div class="sub-header">⚡ Efficiency Metrics</div>', unsafe_allow_html=True)
    
    efficiency_tabs = st.tabs(["Funding Efficiency", "Revenue Efficiency"])
    
    with efficiency_tabs[0]:
        funding_eff_fig = createEfficiencyBarChart(
            filtered_df,
            selected_company,
            'Funding_Efficiency',
            'Funding Efficiency Comparison (Higher is Better)'
        )
        st.plotly_chart(funding_eff_fig, use_container_width=True)
    
    with efficiency_tabs[1]:
        revenue_eff_fig = createEfficiencyBarChart(
            filtered_df,
            selected_company,
            'Revenue_Efficiency',
            'Revenue Efficiency Comparison (Higher is Better)'
        )
        st.plotly_chart(revenue_eff_fig, use_container_width=True)

if __name__ == "__main__":
    main()