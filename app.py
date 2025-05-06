# app.py
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

# Set page config
st.set_page_config(page_title="Startup Research Assistant", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Function to convert values like '8M', '2000+', etc.
def convert_value(x):
    if pd.isna(x) or x == "null" or not isinstance(x, str):
        return np.nan
    x = x.strip().upper()
    match = re.search(r'([\d\.]+)', x)
    if not match:
        return np.nan
    number = float(match.group(1))
    if 'K' in x:
        return number * 1_000
    elif 'M' in x:
        return number * 1_000_000
    elif 'B' in x:
        return number * 1_000_000_000
    else:
        return number

# Load and clean data
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("company_competitor_data.csv")
    
    # Drop duplicates
    df.drop_duplicates(subset=["Company Name"], keep='first', inplace=True)

    # Convert string-based numeric fields
    df['total_funding_usd'] = df['Total Funding Raised (USD or INR)'].apply(convert_value)
    df['team_size_estimated'] = df['Team Size'].apply(lambda x: convert_value(str(x).split('-')[0]) if '-' in str(x) else convert_value(x))

    # Extract primary industry
    def extract_primary_industry(tags):
        if pd.isna(tags):
            return 'Unknown'
        return tags.split(';')[0].strip() if ';' in tags else tags.strip()

    df['primary_industry'] = df['Industry/Category Tags'].apply(extract_primary_industry)

    # Fill missing funding/team size using median per industry
    def fill_by_industry(df, col):
        return df.groupby('primary_industry')[col].transform(lambda x: x.fillna(x.median()))

    df['total_funding_usd'] = fill_by_industry(df, 'total_funding_usd')
    df['team_size_estimated'] = fill_by_industry(df, 'team_size_estimated')

    # If still missing, use global median
    df['total_funding_usd'] = df['total_funding_usd'].fillna(df['total_funding_usd'].median())
    df['team_size_estimated'] = df['team_size_estimated'].fillna(df['team_size_estimated'].median())

    # Maturity score calculation
    max_age = df['Age (Years)'].max()
    max_funding = df['total_funding_usd'].max()
    max_team = df['team_size_estimated'].max()

    df['age_score'] = (df['Age (Years)'] / max_age) * 100
    df['funding_score'] = (df['total_funding_usd'] / max_funding) * 100
    df['team_score'] = (df['team_size_estimated'] / max_team) * 100

    df['maturity_score'] = (
        df['age_score'] * 0.2 +
        df['funding_score'] * 0.5 +
        df['team_score'] * 0.3
    ).round(1)

    return df

# Load dataset
df = load_and_clean_data()

# Function to create startup profile
def create_startup_profile(startup_name, df):
    if startup_name not in df['Company Name'].values:
        return None
    row = df[df['Company Name'] == startup_name].iloc[0]
    profile = {
        'name': startup_name,
        'founding_year': int(row['Founding Year']),
        'age': int(row['Age (Years)']),
        'funding': f"${row['total_funding_usd']/1e6:.1f}M",
        'team_size': int(row['team_size_estimated']),
        'industry': row['primary_industry'],
        'maturity_score': round(row['maturity_score'], 1),
        'tech_stack': row['Tech Stack / Core AI Capabilities'],
        'product_launches': row['Product Launches / Updates (last 2 years)'],
        'customer_base': row['Customer/User Base Size']
    }

    # Find similar startups
    industry_df = df[df['primary_industry'] == profile['industry']]
    ranked = industry_df.sort_values('maturity_score', ascending=False)
    ranked['rank'] = range(1, len(ranked)+1)
    profile['rank_in_industry'] = ranked.loc[ranked['Company Name'] == startup_name, 'rank'].values[0]

    # Get top 3 competitors
    profile['top_competitors'] = ranked[ranked['Company Name'] != startup_name]['Company Name'].head(3).tolist()
    return profile

# Function to generate PDF report
def generate_pdf_report(profile):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    # Title
    content.append(Paragraph(f"{profile['name']} - Startup Analysis Report", styles["Title"]))
    content.append(Spacer(1, 12))

    # Executive Summary
    content.append(Paragraph("<b>Executive Summary</b>", styles["Normal"]))
    summary = f"{profile['name']} is a {profile['age']}-year-old {profile['industry']} startup founded in {profile['founding_year']}. "
    summary += f"It has raised {profile['funding']} in funding and has an estimated team size of ~{profile['team_size']}. "
    summary += f"It ranks #{profile['rank_in_industry']} in maturity among all {profile['industry']} startups."
    content.append(Paragraph(summary, styles["Normal"]))
    content.append(Spacer(1, 12))

    # Key Metrics Table
    data = [
    ["Metric", "Value"],
    ["Founding Year", str(profile['founding_year'])],
    ["Age", f"{profile['age']} years"],
    ["Total Funding", profile['funding']],
    ["Team Size", str(profile['team_size'])],
    ["Primary Industry", profile['industry']],
    ["Maturity Score", f"{profile['maturity_score']}/100"],
    ["Rank in Industry", f"#{profile['rank_in_industry']}"]  # Bold the rank
]

    # Create table with custom styles
    table = Table(data, colWidths=[200, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),  # Make header bold
        ('FONTSIZE', (0, 0), (1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(table)
    content.append(Spacer(1, 12))

    # Competitor Comparison Section
    content.append(Paragraph("<b>Top Competitors in Same Industry</b>", styles["Normal"]))
    for name in profile['top_competitors']:
        content.append(Paragraph(f"- {name}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Tech Stack / Products
    content.append(Paragraph("<b>Tech Stack / Product Launches</b>", styles["Normal"]))
    content.append(Paragraph(f"• Tech Stack: {profile['tech_stack']}", styles["Normal"]))
    content.append(Paragraph(f"• Recent Launches: {profile['product_launches']}", styles["Normal"]))
    content.append(Spacer(1, 12))

    # Build PDF
    doc.build(content)
    buffer.seek(0)
    return buffer

# Main App UI
st.title("🔍 Indian Startup Competitor Assistant")
st.markdown("A tool to compare startups across funding, maturity, and industry.")

# Sidebar Navigation
selected_startup = st.selectbox("Select a Startup", df["Company Name"].sort_values().unique())

if selected_startup:
    # Get startup profile
    profile = create_startup_profile(selected_startup, df)
    if not profile:
        st.error("Startup not found.")
        st.stop()

    # Display basic info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"### {profile['name']}")
        st.write(f"**Founded:** {profile['founding_year']} ({profile['age']} years ago)")
        st.write(f"**Industry:** {profile['industry']}")
        st.write(f"**Funding:** {profile['funding']}")
        st.write(f"**Team Size:** ~{profile['team_size']}")
        st.write(f"**Maturity Score:** {profile['maturity_score']}/100")
        st.write(f"**Rank in Industry:** #{profile['rank_in_industry']}")
        st.markdown('</div>')

    with col2:
        # Maturity Gauge Chart (optional)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=profile['maturity_score'],
            title={'text': "Maturity Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1E88E5"},
                'steps': [
                    {'range': [0, 33], 'color': "#FFCDD2"},
                    {'range': [33, 66], 'color': "#FFECB3"},
                    {'range': [66, 100], 'color': "#C8E6C9"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Bar Chart: Rank comparison in same industry
    st.markdown("### 📊 Standing in Industry")
    industry_startups = df[df['primary_industry'] == profile['industry']].copy()
    industry_startups = industry_startups.sort_values('maturity_score', ascending=False).reset_index(drop=True)
    industry_startups['Standing'] = industry_startups.index + 1

    fig_bar = px.bar(
        industry_startups,
        x='Company Name',
        y='maturity_score',
        color='maturity_score',
        labels={'maturity_score': 'Maturity Score'},
        title=f"Startups in {profile['industry']} Industry by Maturity"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Generate PDF button
    if st.button("📄 Generate PDF Report"):
        pdf_buffer = generate_pdf_report(profile)
        b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{profile["name"]}_report.pdf">⬇️ Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)