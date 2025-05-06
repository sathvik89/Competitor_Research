# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import base64
# For PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import re

# Set page config
st.set_page_config(page_title="Startup Comparator", layout="wide")

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

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("company_competitor_data.csv")
    return df

df = load_data()

# Convert string numbers like '8M', '2000+' to numeric
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

# Clean data and calculate maturity score
def clean_data(df):
    cleaned_df = df.copy()

    # Convert key fields
    cleaned_df['total_funding_usd'] = cleaned_df['Total Funding Raised (USD or INR)'].apply(convert_value)
    cleaned_df['team_size_estimated'] = cleaned_df['Team Size'].apply(lambda x: convert_value(str(x).split('-')[0]) if '-' in str(x) else convert_value(x))

    # Extract primary industry
    def extract_primary_industry(tags):
        if pd.isna(tags):
            return 'Unknown'
        return tags.split(';')[0].strip() if ';' in tags else tags.strip()

    cleaned_df['primary_industry'] = cleaned_df['Industry/Category Tags'].apply(extract_primary_industry)

    # Remove duplicates by Company Name
    cleaned_df.drop_duplicates(subset=['Company Name'], keep='first', inplace=True)

    # Fill missing funding/team size with median per industry
    def fill_by_industry(df, col):
        return df.groupby('primary_industry')[col].transform(lambda x: x.fillna(x.median()))

    cleaned_df['total_funding_usd'] = fill_by_industry(cleaned_df, 'total_funding_usd')
    cleaned_df['team_size_estimated'] = fill_by_industry(cleaned_df, 'team_size_estimated')

    # If still missing, use global median
    cleaned_df['total_funding_usd'] = cleaned_df['total_funding_usd'].fillna(cleaned_df['total_funding_usd'].median())
    cleaned_df['team_size_estimated'] = cleaned_df['team_size_estimated'].fillna(cleaned_df['team_size_estimated'].median())

    # Calculate Maturity Score (Age + Funding + Team Size)
    max_age = cleaned_df['Age (Years)'].max()
    max_funding = cleaned_df['total_funding_usd'].max()
    max_team = cleaned_df['team_size_estimated'].max()

    cleaned_df['age_score'] = (cleaned_df['Age (Years)'] / max_age) * 100
    cleaned_df['funding_score'] = (cleaned_df['total_funding_usd'] / max_funding) * 100
    cleaned_df['team_score'] = (cleaned_df['team_size_estimated'] / max_team) * 100

    cleaned_df['maturity_score'] = (
        cleaned_df['age_score'] * 0.2 +
        cleaned_df['funding_score'] * 0.5 +
        cleaned_df['team_score'] * 0.3
    ).round(1)

    return cleaned_df

# Load and clean dataset
df_cleaned = clean_data(load_data())

# Function to generate PDF report
def generate_pdf_report(startup_name, df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    content = []

    profile = df[df['Company Name'] == startup_name].iloc[0]
    content.append(Paragraph(f"{profile['Company Name']} - Startup Analysis", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph("📊 Executive Summary", styles['Heading2']))
    summary = f"{profile['Company Name']} is a {profile['Age (Years)']}-year-old {profile['primary_industry']} startup. "
    summary += f"It has raised ${profile['total_funding_usd']/1e6:.1f}M in funding and has a team of ~{int(profile['team_size_estimated'])}. "
    summary += f"Its maturity score is {profile['maturity_score']}/100."
    content.append(Paragraph(summary, styles['Normal']))
    content.append(Spacer(1, 12))

    data = [
        ['Metric', 'Value'],
        ['Founding Year', str(profile['Founding Year'])],
        ['Age', f"{profile['Age (Years)']} years"],
        ['Funding', f"${profile['total_funding_usd']/1e6:.1f}M"],
        ['Team Size', int(profile['team_size_estimated'])],
        ['Primary Industry', profile['primary_industry']],
        ['Maturity Score', f"{profile['maturity_score']}/100"]
    ]

    table = Table(data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    content.append(table)
    content.append(Spacer(1, 12))

    doc.build(content)
    buffer.seek(0)
    return buffer

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Competitor Report", "Raw Data"])

if page == "Competitor Report":
    st.title("🔍 Indian Startup Competitor Assistant")

    selected_startup = st.selectbox("Select a Startup", options=df_cleaned['Company Name'].sort_values().unique())

    # Get selected startup
    startup_row = df_cleaned[df_cleaned['Company Name'] == selected_startup].iloc[0]

    # Show basic info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📌 Basic Info")
        st.write(f"**Name:** {startup_row['Company Name']}")
        st.write(f"**Founded:** {startup_row['Founding Year']} ({startup_row['Age (Years)']} years ago)")
        st.write(f"**Industry:** {startup_row['primary_industry']}")
        st.write(f"**Funding:** ${startup_row['total_funding_usd']/1e6:.1f}M")
        st.write(f"**Team Size:** ~{int(startup_row['team_size_estimated'])}")

    with col2:
        # Maturity gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=startup_row['maturity_score'],
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
        st.plotly_chart(fig, use_container_width=True)

    # Find similar startups
    industry = startup_row['primary_industry']
    industry_startups = df_cleaned[df_cleaned['primary_industry'] == industry]
    similar_startups = industry_startups[industry_startups['Company Name'] != selected_startup].sort_values('maturity_score', ascending=False).head(3)

    st.subheader("🤝 Similar Startups in Same Industry")
    st.dataframe(similar_startups[['Company Name', 'maturity_score', 'total_funding_usd', 'team_size_estimated']])

    # Generate PDF button
    if st.button("📄 Generate PDF Report"):
        pdf_buffer = generate_pdf_report(selected_startup, df_cleaned)
        b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{selected_startup}_report.pdf">⬇️ Download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

elif page == "Raw Data":
    st.title("📋 Raw & Cleaned Data")
    st.dataframe(df_cleaned[['Company Name', 'primary_industry', 'total_funding_usd', 'team_size_estimated', 'maturity_score']])
    st.download_button("⬇️ Download CSV", df_cleaned.to_csv(index=False), file_name="cleaned_startup_data.csv")

# Footer
st.markdown("---")
st.markdown("💡 *Built by You — A Product Manager's Quick Tool to Compare Companies*")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# from wordcloud import WordCloud
# import re
# from io import BytesIO
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib import colors
# import base64

# # Set page config
# st.set_page_config(
#     page_title="Indian Startup Analysis",
#     page_icon="🚀",
#     layout="wide"
# )

# # Custom CSS to make the app look better
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1E88E5;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #424242;
#         margin-bottom: 0.5rem;
#     }
#     .metric-card {
#         background-color: #f5f5f5;
#         border-radius: 5px;
#         padding: 1rem;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .highlight {
#         color: #1E88E5;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Function to load and clean data
# @st.cache_data
# def load_and_clean_data():
#     # In a real scenario, you'd load from a CSV file
#     # For this example, I'll create a DataFrame with the data provided
    
#     data = """
# Company Name,Total Funding Raised (USD or INR),Number of Funding Rounds,Last Funding Date,Team Size,Founding Year,Age (Years),Tech Stack / Core AI Capabilities,Customer/User Base Size,Revenue,Product Launches / Updates (last 2 years),Industry/Category Tags
# Sharang Shakti,$600K,1,null,null,2023,2,UAV Technology / Defense Tech,null,null,hantR UAV,Defense Tech;UAV;Military;Security
# Swish,$2M,1,2024-08,null,2024,1,Food Delivery Platform,10000+,null,Cloud Kitchen Launch,FoodTech;Quick Commerce;Food Delivery
# Portkey.ai,null,null,null,null,2023,2,Generative AI;LLM Integration;Analytics,null,null,Model Management;Prompt Management;Analytics Tools,AI;Enterprise SaaS;Developer Tools
# Covrzy,null,null,null,null,2023,2,InsurTech;Recommendation Engine;WhatsApp Integration,150+,null,Insurance Recommendation Engine;WhatsApp Chatbot,InsurTech;SME Services;Business Insurance
# Aarogya Tech,null,null,null,null,2023,2,HealthTech;Patient Monitoring;Data Analytics,10+,null,Disease-specific Patient Monitoring System;DocSeva.com,HealthTech;Healthcare;MedTech
# Propacity,$771K,2,null,51-200,2019,6,Real Estate Analytics;Investment Platform,null,null,null,PropTech;Real Estate;Investment
# HouseEazy,$11.7M,1,null,null,2023,2,PropTech;Transaction Management,null,null,null,PropTech;Real Estate;Resale
# Landeed,null,null,null,null,2023,2,Property Title Search;Legal Tech,null,null,null,PropTech;LegalTech;Real Estate
# Flent,$777K,1,null,null,2024,1,Cloud Computing;Property Management,null,null,null,PropTech;Cloud Services;Real Estate
# Stylework Innovation Hub,$120K,1,null,null,2024,1,Workspace Management;Real Estate Tech,null,null,null,PropTech;Workspace;Real Estate
# Jivi,null,null,null,null,2024,1,AI;Healthcare;Diagnostics,null,null,null,HealthTech;AI;MedTech
# Alt Carbon,null,null,null,null,2023,2,Carbon Sequestration;Agriculture Tech,null,null,Carbon Removal Technology,CleanTech;AgriTech;Climate Tech
# Sahaj Gaming,null,null,null,null,2023,2,Gaming;Mobile Apps,25000+,null,1Ludo,Gaming;Entertainment;Mobile Gaming
# Salty,null,null,null,null,2022,3,E-commerce;Fashion Tech,null,$1M ARR,null,Fashion;E-commerce;Jewelry
# Ambiator,null,null,null,null,2024,1,Energy-efficient Cooling;CleanTech,null,null,Sustainable Cooling Systems,CleanTech;Energy;Sustainability
# Clueso,null,null,null,null,2024,1,Generative AI;Instructional Content,null,null,Training Video Generation,AI;EdTech;Enterprise SaaS
# MapMyCrop,null,null,null,null,2024,1,AgriTech;AI;Multispectral Imaging,null,null,Crop Data Analytics Platform,AgriTech;AI;Data Analytics
# Planet Electric,null,null,null,null,2024,1,Electric Vehicles;Last-mile Delivery,null,null,Electric Trucks,EV;Logistics;CleanTech
# SpanTrik,null,null,null,null,2024,1,Aerospace;Reusable Launch Vehicles,null,null,Raven Launch Vehicle,SpaceTech;Aerospace;Satellite
# TSAW Drones,null,null,null,null,2024,1,Autonomous Drones;Logistics,null,null,Logistics Drones,Drones;Logistics;AgriTech
# Umrit,null,null,null,null,2024,1,AI;Health Analytics;Disease Detection,null,null,Personal Health Assistant,HealthTech;AI;Wellness
# Vicharak,null,null,null,null,2024,1,Parallel Computing;Hardware,null,null,Computing Systems,Hardware;Computing;Enterprise Tech
# Vmaker AI,null,null,null,null,2024,1,AI;Video Editing;Content Creation,null,null,AI Video Editor,AI;Content Creation;Media Tech
# WhatsLoan,null,null,null,null,2024,1,FinTech;Loan Processing;Mobile Platform,null,null,Loan-as-a-Service Platform,FinTech;Lending;Mobile
# Segwise,$1.6M,1,null,null,2023,2,AI;Product Analytics;Growth Metrics,null,null,Product Development Insights Platform,AI;SaaS;Product Analytics
# Plugzmart,null,null,null,null,2024,1,EV Charging;CleanTech,null,null,Fast EV Charger,EV;CleanTech;Energy
# Go Do Good,null,null,null,null,2024,1,Sustainable Packaging;Agro-waste Recycling,null,null,Eco-friendly Packaging Solutions,Sustainability;Packaging;CleanTech
# Golden Feathers,null,null,null,null,2024,1,Upcycling;Textile;Sustainable Materials,null,null,Feather-based Textiles and Paper,Sustainability;Textile;Upcycling
# Meri Bhakti,null,null,null,null,2024,1,Digital Devotional Platform;Religious Tech,null,null,Astrology Consultations;Temple Locator,Religious Tech;Digital Services;Spirituality
# Sarla Aviation,null,null,null,null,2023,2,eVTOL;Urban Air Mobility,null,null,Flying Taxi Development,Aviation;Urban Mobility;Transportation
# Zepto,$1.4B,5,2023-08,null,2021,4,Quick Commerce;Logistics;Supply Chain,null,null,10-minute Grocery Delivery,Quick Commerce;Retail;Logistics
# Razorpay,$741M,8,2022-12,2000+,2014,11,Payment Gateway;FinTech;Banking Solutions,8M+,null,Razorpay Banking;Tax Payment Solution,FinTech;Payment Processing;Banking
# Ola Electric,$1.1B,6,2023-09,2000+,2017,8,Electric Vehicles;Battery Tech;Charging Infrastructure,null,null,S1 Pro;S1 Air;Electric Car Prototype,EV;Mobility;CleanTech
# CRED,$801M,7,2023-01,800+,2018,7,FinTech;Credit Card Management;Rewards Platform,11M+,null,CRED Mint;CRED Store;CRED Pay,FinTech;Consumer Tech;Credit Management
# Meesho,$1.1B,8,2022-09,2000+,2015,10,E-commerce;Social Commerce;Reselling Platform,125M+,null,Zero Commission Policy;Meesho Mall,E-commerce;Social Commerce;Retail
# """
    
#     # Create DataFrame from the CSV string
#     df = pd.read_csv(BytesIO(data.encode()), sep=',')
    
#     # Clean and transform the data
    
#     # 1. Convert funding to numeric values
#     def convert_funding(x):
#         if pd.isna(x) or x == 'null' or not isinstance(x, str):
#             return np.nan
#         x = x.strip().upper()
#         match = re.search(r'([\d\.]+)', x)
#         if not match:
#             return np.nan
#         number = float(match.group(1))
#         if 'K' in x:
#             return number * 1_000
#         elif 'M' in x:
#             return number * 1_000_000
#         elif 'B' in x:
#             return number * 1_000_000_000
#         else:
#             return float(number)
    
#     df['Total Funding Raised (USD)'] = df['Total Funding Raised (USD or INR)'].apply(convert_funding)
    
#     # 2. Convert team size to numeric
#     def convert_team_size(x):
#         if pd.isna(x) or x == 'null' or not isinstance(x, str):
#             return np.nan
#         x = x.strip()
#         if '-' in x:
#             low, high = map(int, re.findall(r'\d+', x))
#             return (low + high) / 2
#         elif '+' in x:
#             return float(x.replace('+', ''))
#         elif re.search(r'\d+', x):
#             return float(re.search(r'\d+', x).group())
#         return np.nan
    
#     df['Team Size (Numeric)'] = df['Team Size'].apply(convert_team_size)
    
#     # 3. Convert customer base to numeric
#     def convert_customer_base(x):
#         if pd.isna(x) or x == 'null' or not isinstance(x, str):
#             return np.nan
#         x = x.strip().upper()
#         match = re.search(r'([\d\.]+)', x)
#         if not match:
#             return np.nan
#         number = float(match.group(1))
#         if 'M' in x:
#             return number * 1_000_000
#         elif 'K' in x:
#             return number * 1_000
#         elif '+' in x:
#             return number
#         return number
    
#     df['Customer Base (Numeric)'] = df['Customer/User Base Size'].apply(convert_customer_base)
    
#     # 4. Extract primary industry
#     def extract_primary_industry(tags):
#         if pd.isna(tags) or tags == 'null':
#             return 'Unknown'
#         if isinstance(tags, str):
#             # Split by semicolon and take the first tag
#             return tags.split(';')[0].strip()
#         return 'Unknown'
    
#     df['Primary Industry'] = df['Industry/Category Tags'].apply(extract_primary_industry)
    
#     # 5. Fill missing values with appropriate defaults or aggregates
#     # For numeric columns, fill with median of the same industry
#     for industry in df['Primary Industry'].unique():
#         industry_mask = df['Primary Industry'] == industry
        
#         # Fill funding
#         median_funding = df.loc[industry_mask, 'Total Funding Raised (USD)'].median()
#         if not pd.isna(median_funding):
#             df.loc[industry_mask, 'Total Funding Raised (USD)'] = df.loc[industry_mask, 'Total Funding Raised (USD)'].fillna(median_funding)
        
#         # Fill team size
#         median_team = df.loc[industry_mask, 'Team Size (Numeric)'].median()
#         if not pd.isna(median_team):
#             df.loc[industry_mask, 'Team Size (Numeric)'] = df.loc[industry_mask, 'Team Size (Numeric)'].fillna(median_team)
    
#     # For remaining NaN values, use global median
#     df['Total Funding Raised (USD)'] = df['Total Funding Raised (USD)'].fillna(df['Total Funding Raised (USD)'].median())
#     df['Team Size (Numeric)'] = df['Team Size (Numeric)'].fillna(df['Team Size (Numeric)'].median())
#     df['Number of Funding Rounds'] = df['Number of Funding Rounds'].fillna(1)  # Assume at least 1 round if unknown
    
#     # 6. Create funding stage category
#     def determine_funding_stage(funding, rounds):
#         if pd.isna(funding) or funding < 500000:
#             return 'Pre-seed'
#         elif funding < 3000000:
#             return 'Seed'
#         elif funding < 15000000:
#             return 'Series A'
#         elif funding < 50000000:
#             return 'Series B'
#         elif funding < 100000000:
#             return 'Series C'
#         else:
#             return 'Late Stage'
    
#     df['Funding Stage'] = df.apply(lambda x: determine_funding_stage(x['Total Funding Raised (USD)'], x['Number of Funding Rounds']), axis=1)
    
#     # 7. Create maturity score (0-100) based on age, funding, team size
#     # Normalize each factor to 0-100 scale and take weighted average
#     max_age = df['Age (Years)'].max()
#     max_funding = df['Total Funding Raised (USD)'].max()
#     max_team = df['Team Size (Numeric)'].max()
    
#     df['Age Score'] = (df['Age (Years)'] / max_age) * 100
#     df['Funding Score'] = (df['Total Funding Raised (USD)'] / max_funding) * 100
#     df['Team Score'] = (df['Team Size (Numeric)'] / max_team) * 100
    
#     # Weighted average: Age (20%), Funding (50%), Team Size (30%)
#     df['Maturity Score'] = (df['Age Score'] * 0.2) + (df['Funding Score'] * 0.5) + (df['Team Score'] * 0.3)
    
#     return df

# # Load and clean the data
# df = load_and_clean_data()

# # Function to extract all unique industries from tags
# def extract_all_industries(df):
#     all_industries = set()
#     for tags in df['Industry/Category Tags'].dropna():
#         if isinstance(tags, str):
#             industries = [tag.strip() for tag in tags.split(';')]
#             all_industries.update(industries)
#     return sorted(list(all_industries))

# all_industries = extract_all_industries(df)

# # Function to create a word cloud from industry tags
# def generate_industry_wordcloud(df):
#     all_tags = ' '.join([tags for tags in df['Industry/Category Tags'].dropna() if isinstance(tags, str)])
#     wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(all_tags)
    
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     return fig

# # Function to create funding distribution by industry
# def plot_funding_by_industry(df):
#     industry_funding = df.groupby('Primary Industry')['Total Funding Raised (USD)'].sum().sort_values(ascending=False).head(10)
    
#     fig = px.bar(
#         x=industry_funding.index,
#         y=industry_funding.values,
#         labels={'x': 'Industry', 'y': 'Total Funding (USD)'},
#         title='Top 10 Industries by Total Funding',
#         color=industry_funding.values,
#         color_continuous_scale='Viridis'
#     )
    
#     fig.update_layout(xaxis_tickangle=-45)
#     return fig

# # Function to create age vs funding scatter plot
# def plot_age_vs_funding(df):
#     fig = px.scatter(
#         df,
#         x='Age (Years)',
#         y='Total Funding Raised (USD)',
#         color='Primary Industry',
#         size='Team Size (Numeric)',
#         hover_name='Company Name',
#         log_y=True,
#         title='Startup Age vs Funding (size represents team size)'
#     )
#     return fig

# # Function to create funding stage distribution
# def plot_funding_stage_distribution(df):
#     stage_counts = df['Funding Stage'].value_counts().reset_index()
#     stage_counts.columns = ['Funding Stage', 'Count']
    
#     # Define stage order
#     stage_order = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Late Stage']
#     stage_counts['Funding Stage'] = pd.Categorical(stage_counts['Funding Stage'], categories=stage_order, ordered=True)
#     stage_counts = stage_counts.sort_values('Funding Stage')
    
#     fig = px.pie(
#         stage_counts,
#         values='Count',
#         names='Funding Stage',
#         title='Distribution of Startups by Funding Stage',
#         color='Funding Stage',
#         color_discrete_sequence=px.colors.sequential.Viridis
#     )
#     return fig

# # Function to create maturity score distribution
# def plot_maturity_score_distribution(df):
#     fig = px.histogram(
#         df,
#         x='Maturity Score',
#         nbins=20,
#         color='Primary Industry',
#         title='Distribution of Startup Maturity Scores',
#         marginal='box'
#     )
#     return fig

# # Function to create startup comparison radar chart
# def create_startup_comparison_radar(selected_startups, df):
#     if not selected_startups:
#         return None
    
#     # Metrics to compare
#     metrics = ['Age (Years)', 'Total Funding Raised (USD)', 'Team Size (Numeric)', 'Number of Funding Rounds', 'Maturity Score']
    
#     # Normalize metrics to 0-1 scale for radar chart
#     df_radar = df[df['Company Name'].isin(selected_startups)].copy()
    
#     for metric in metrics:
#         max_val = df[metric].max()
#         min_val = df[metric].min()
#         if max_val > min_val:
#             df_radar[f'{metric}_norm'] = (df_radar[metric] - min_val) / (max_val - min_val)
#         else:
#             df_radar[f'{metric}_norm'] = 0
    
#     # Create radar chart
#     fig = go.Figure()
    
#     for _, row in df_radar.iterrows():
#         fig.add_trace(go.Scatterpolar(
#             r=[row[f'{metric}_norm'] for metric in metrics],
#             theta=metrics,
#             fill='toself',
#             name=row['Company Name']
#         ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )
#         ),
#         title='Startup Comparison Radar Chart',
#         showlegend=True
#     )
    
#     return fig

# # Function to create startup profile
# def create_startup_profile(startup_name, df):
#     if startup_name not in df['Company Name'].values:
#         return None
    
#     startup = df[df['Company Name'] == startup_name].iloc[0]
    
#     # Create a profile dictionary
#     profile = {
#         'name': startup_name,
#         'founding_year': int(startup['Founding Year']),
#         'age': int(startup['Age (Years)']),
#         'funding': f"${startup['Total Funding Raised (USD)']:,.0f}" if not pd.isna(startup['Total Funding Raised (USD)']) else 'Unknown',
#         'funding_rounds': int(startup['Number of Funding Rounds']) if not pd.isna(startup['Number of Funding Rounds']) else 'Unknown',
#         'team_size': int(startup['Team Size (Numeric)']) if not pd.isna(startup['Team Size (Numeric)']) else 'Unknown',
#         'industry': startup['Primary Industry'],
#         'all_tags': startup['Industry/Category Tags'] if not pd.isna(startup['Industry/Category Tags']) else 'Unknown',
#         'tech_stack': startup['Tech Stack / Core AI Capabilities'] if not pd.isna(startup['Tech Stack / Core AI Capabilities']) else 'Unknown',
#         'customer_base': startup['Customer/User Base Size'] if not pd.isna(startup['Customer/User Base Size']) else 'Unknown',
#         'product_launches': startup['Product Launches / Updates (last 2 years)'] if not pd.isna(startup['Product Launches / Updates (last 2 years)']) else 'Unknown',
#         'funding_stage': startup['Funding Stage'],
#         'maturity_score': round(startup['Maturity Score'], 1)
#     }
    
#     # Find similar startups (same primary industry)
#     similar_startups = df[(df['Primary Industry'] == profile['industry']) & (df['Company Name'] != startup_name)]
#     similar_startups = similar_startups.sort_values('Maturity Score', ascending=False).head(3)
    
#     profile['similar_startups'] = similar_startups['Company Name'].tolist()
    
#     # Calculate industry position (percentile)
#     industry_df = df[df['Primary Industry'] == profile['industry']]
#     profile['industry_percentile'] = round(percentileofscore(industry_df['Maturity Score'], startup['Maturity Score']), 1)
    
#     return profile

# # Function to generate PDF report
# def generate_pdf_report(startup_name, df):
#     from reportlab.lib.pagesizes import letter
#     from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
#     from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
#     from reportlab.lib import colors
#     import matplotlib.pyplot as plt
#     import io
    
#     # Get startup profile
#     profile = create_startup_profile(startup_name, df)
#     if not profile:
#         return None
    
#     # Create a buffer for the PDF
#     buffer = BytesIO()
    
#     # Create the PDF document
#     doc = SimpleDocTemplate(buffer, pagesize=letter)
#     styles = getSampleStyleSheet()
    
#     # Create custom styles
#     title_style = ParagraphStyle(
#         'Title',
#         parent=styles['Title'],
#         fontSize=24,
#         spaceAfter=12
#     )
    
#     heading_style = ParagraphStyle(
#         'Heading',
#         parent=styles['Heading2'],
#         fontSize=16,
#         spaceAfter=6
#     )
    
#     normal_style = ParagraphStyle(
#         'Normal',
#         parent=styles['Normal'],
#         fontSize=12,
#         spaceAfter=6
#     )
    
#     # Create the content
#     content = []
    
#     # Title
#     content.append(Paragraph(f"{profile['name']} - Startup Analysis Report", title_style))
#     content.append(Spacer(1, 12))
    
#     # Executive Summary
#     content.append(Paragraph("Executive Summary", heading_style))
#     summary = f"{profile['name']} is a {profile['age']}-year-old {profile['industry']} startup founded in {profile['founding_year']}. "
#     summary += f"With {profile['funding']} in funding across {profile['funding_rounds']} rounds, it is currently at the {profile['funding_stage']} stage. "
#     summary += f"The company has a maturity score of {profile['maturity_score']}/100, placing it in the {profile['industry_percentile']}th percentile among {profile['industry']} startups."
#     content.append(Paragraph(summary, normal_style))
#     content.append(Spacer(1, 12))
    
#     # Key Metrics Table
#     content.append(Paragraph("Key Metrics", heading_style))
#     data = [
#         ["Metric", "Value"],
#         ["Founding Year", str(profile['founding_year'])],
#         ["Age", f"{profile['age']} years"],
#         ["Total Funding", profile['funding']],
#         ["Funding Rounds", str(profile['funding_rounds'])],
#         ["Team Size", str(profile['team_size'])],
#         ["Funding Stage", profile['funding_stage']],
#         ["Maturity Score", f"{profile['maturity_score']}/100"],
#         ["Industry Percentile", f"{profile['industry_percentile']}%"]
#     ]
    
#     table = Table(data, colWidths=[200, 300])
#     table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
#         ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
#         ('ALIGN', (0, 0), (1, 0), 'CENTER'),
#         ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (1, 0), 14),
#         ('BOTTOMPADDING', (0, 0), (1, 0), 12),
#         ('BACKGROUND', (0, 1), (1, -1), colors.beige),
#         ('GRID', (0, 0), (-1, -1), 1, colors.black)
#     ]))
    
#     content.append(table)
#     content.append(Spacer(1, 12))
    
#     # Technology and Products
#     content.append(Paragraph("Technology and Products", heading_style))
#     tech_text = f"Tech Stack / Core Capabilities: {profile['tech_stack']}"
#     content.append(Paragraph(tech_text, normal_style))
    
#     product_text = f"Recent Product Launches: {profile['product_launches']}"
#     content.append(Paragraph(product_text, normal_style))
#     content.append(Spacer(1, 12))
    
#     # Market Position
#     content.append(Paragraph("Market Position", heading_style))
#     market_text = f"Industry Categories: {profile['all_tags']}"
#     content.append(Paragraph(market_text, normal_style))
    
#     customer_text = f"Customer/User Base: {profile['customer_base']}"
#     content.append(Paragraph(customer_text, normal_style))
#     content.append(Spacer(1, 12))
    
#     # Similar Startups
#     content.append(Paragraph("Similar Startups", heading_style))
#     similar_text = "Companies in the same industry: " + ", ".join(profile['similar_startups'])
#     content.append(Paragraph(similar_text, normal_style))
#     content.append(Spacer(1, 12))
    
#     # Maturity Score Visualization
#     content.append(Paragraph("Maturity Score Comparison", heading_style))
    
#     # Create a simple bar chart for maturity score comparison
#     fig, ax = plt.subplots(figsize=(6, 4))
    
#     # Get industry average
#     industry_df = df[df['Primary Industry'] == profile['industry']]
#     industry_avg = industry_df['Maturity Score'].mean()
    
#     # Get overall average
#     overall_avg = df['Maturity Score'].mean()
    
#     # Create bar chart
#     bars = ax.bar(['This Startup', 'Industry Average', 'Overall Average'], 
#                  [profile['maturity_score'], industry_avg, overall_avg],
#                  color=['#1E88E5', '#FFC107', '#4CAF50'])
    
#     ax.set_ylabel('Maturity Score (0-100)')
#     ax.set_title('Maturity Score Comparison')
    
#     # Save the figure to a buffer
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
#     img_buffer.seek(0)
    
#     # Add the image to the PDF
#     img = Image(img_buffer, width=400, height=300)
#     content.append(img)
    
#     # Build the PDF
#     doc.build(content)
#     buffer.seek(0)
    
#     return buffer

# # Function to get download link for PDF
# def get_download_link(buffer, filename, text):
#     b64 = base64.b64encode(buffer.getvalue()).decode()
#     href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
#     return href

# # Function to calculate percentile score
# def percentileofscore(a, score):
#     """
#     Calculate the percentile rank of a score relative to a list of scores.
#     """
#     a = np.asarray(a)
#     n = len(a)
#     if n == 0:
#         return 0.0
    
#     a = np.sort(a)
#     idx = np.searchsorted(a, score, side='right')
#     pct = (idx) * 100.0 / n
#     return pct

# # Main app layout
# def main():
#     st.markdown('<div class="main-header">🚀 Indian Startup Ecosystem Analysis</div>', unsafe_allow_html=True)
    
#     # Sidebar for navigation
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Select a page", ["Overview", "Startup Explorer", "Industry Analysis", "Funding Analysis", "Raw Data"])
    
#     if page == "Overview":
#         st.markdown('<div class="sub-header">📊 Ecosystem Overview</div>', unsafe_allow_html=True)
        
#         # Key metrics
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#             st.metric("Total Startups", len(df))
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         with col2:
#             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#             total_funding = df['Total Funding Raised (USD)'].sum()
#             st.metric("Total Funding", f"${total_funding/1e9:.2f}B")
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         with col3:
#             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#             avg_age = df['Age (Years)'].mean()
#             st.metric("Average Age", f"{avg_age:.1f} years")
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         with col4:
#             st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#             unique_industries = len(df['Primary Industry'].unique())
#             st.metric("Industries", unique_industries)
#             st.markdown('</div>', unsafe_allow_html=True)
        
#         # Industry word cloud
#         st.markdown('<div class="sub-header">🏷️ Industry Landscape</div>', unsafe_allow_html=True)
#         wordcloud_fig = generate_industry_wordcloud(df)
#         st.pyplot(wordcloud_fig)
        
#         # Funding by industry
#         st.markdown('<div class="sub-header">💰 Funding Distribution by Industry</div>', unsafe_allow_html=True)
#         funding_fig = plot_funding_by_industry(df)
#         st.plotly_chart(funding_fig, use_container_width=True)
        
#         # Age vs Funding
#         st.markdown('<div class="sub-header">📈 Startup Age vs Funding</div>', unsafe_allow_html=True)
#         age_funding_fig = plot_age_vs_funding(df)
#         st.plotly_chart(age_funding_fig, use_container_width=True)
        
#         # Funding stage distribution
#         st.markdown('<div class="sub-header">🔄 Funding Stage Distribution</div>', unsafe_allow_html=True)
#         stage_fig = plot_funding_stage_distribution(df)
#         st.plotly_chart(stage_fig, use_container_width=True)
        
#     elif page == "Startup Explorer":
#         st.markdown('<div class="sub-header">🔍 Startup Explorer</div>', unsafe_allow_html=True)
        
#         # Startup selector
#         selected_startup = st.selectbox("Select a startup to analyze", df['Company Name'].sort_values())
        
#         if selected_startup:
#             # Get startup profile
#             profile = create_startup_profile(selected_startup, df)
            
#             if profile:
#                 # Display startup info
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.markdown(f"### {profile['name']}")
#                     st.markdown(f"**Founded:** {profile['founding_year']} ({profile['age']} years ago)")
#                     st.markdown(f"**Industry:** {profile['industry']}")
#                     st.markdown(f"**Funding:** {profile['funding']} ({profile['funding_stage']})")
#                     st.markdown(f"**Team Size:** {profile['team_size']}")
#                     st.markdown(f"**Maturity Score:** {profile['maturity_score']}/100 (Top {100-profile['industry_percentile']:.1f}% in industry)")
                
#                 with col2:
#                     # Maturity gauge
#                     fig = go.Figure(go.Indicator(
#                         mode = "gauge+number",
#                         value = profile['maturity_score'],
#                         domain = {'x': [0, 1], 'y': [0, 1]},
#                         title = {'text': "Maturity Score"},
#                         gauge = {
#                             'axis': {'range': [0, 100]},
#                             'bar': {'color': "#1E88E5"},
#                             'steps': [
#                                 {'range': [0, 33], 'color': "#FFCDD2"},
#                                 {'range': [33, 66], 'color': "#FFECB3"},
#                                 {'range': [66, 100], 'color': "#C8E6C9"}
#                             ]
#                         }
#                     ))
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 # Tech and product info
#                 st.markdown("### Technology and Products")
#                 st.markdown(f"**Tech Stack / Core Capabilities:** {profile['tech_stack']}")
#                 st.markdown(f"**Recent Product Launches:** {profile['product_launches']}")
                
#                 # Market position
#                 st.markdown("### Market Position")
#                 st.markdown(f"**Industry Categories:** {profile['all_tags']}")
#                 st.markdown(f"**Customer/User Base:** {profile['customer_base']}")
                
#                 # Similar startups
#                 st.markdown("### Similar Startups")
#                 st.markdown("Companies in the same industry:")
#                 for similar in profile['similar_startups']:
#                     st.markdown(f"- {similar}")
                
#                 # Comparison with other startups
#                 st.markdown("### Compare with Other Startups")
#                 comparison_startups = st.multiselect(
#                     "Select startups to compare with",
#                     df[df['Company Name'] != selected_startup]['Company Name'].sort_values(),
#                     default=profile['similar_startups'][:1] if profile['similar_startups'] else None
#                 )
                
#                 if comparison_startups:
#                     all_selected = [selected_startup] + comparison_startups
#                     radar_fig = create_startup_comparison_radar(all_selected, df)
#                     st.plotly_chart(radar_fig, use_container_width=True)
                
#                 # Generate PDF report
#                 st.markdown("### Generate Report")
#                 if st.button("Generate PDF Report"):
#                     with st.spinner("Generating PDF report..."):
#                         pdf_buffer = generate_pdf_report(selected_startup, df)
#                         if pdf_buffer:
#                             st.markdown(
#                                 get_download_link(pdf_buffer, f"{selected_startup}_analysis.pdf", "Download PDF Report"),
#                                 unsafe_allow_html=True
#                             )
#                         else:
#                             st.error("Failed to generate PDF report.")
        
#     elif page == "Industry Analysis":
#         st.markdown('<div class="sub-header">🏭 Industry Analysis</div>', unsafe_allow_html=True)
        
#         # Industry selector
#         selected_industry = st.selectbox("Select an industry to analyze", sorted(df['Primary Industry'].unique()))
        
#         if selected_industry:
#             # Filter data for selected industry
#             industry_df = df[df['Primary Industry'] == selected_industry]
            
#             # Display industry info
#             st.markdown(f"### {selected_industry} Industry")
#             st.markdown(f"**Number of Startups:** {len(industry_df)}")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 total_funding = industry_df['Total Funding Raised (USD)'].sum()
#                 st.metric("Total Funding", f"${total_funding/1e6:.2f}M")
            
#             with col2:
#                 avg_age = industry_df['Age (Years)'].mean()
#                 st.metric("Average Age", f"{avg_age:.1f} years")
            
#             with col3:
#                 avg_maturity = industry_df['Maturity Score'].mean()
#                 st.metric("Average Maturity", f"{avg_maturity:.1f}/100")
            
#             # Funding stage distribution for this industry
#             st.markdown("### Funding Stage Distribution")
#             stage_counts = industry_df['Funding Stage'].value_counts().reset_index()
#             stage_counts.columns = ['Funding Stage', 'Count']
            
#             # Define stage order
#             stage_order = ['Pre-seed', 'Seed', 'Series A', 'Series B', 'Series C', 'Late Stage']
#             stage_counts['Funding Stage'] = pd.Categorical(stage_counts['Funding Stage'], categories=stage_order, ordered=True)
#             stage_counts = stage_counts.sort_values('Funding Stage')
            
#             fig = px.bar(
#                 stage_counts,
#                 x='Funding Stage',
#                 y='Count',
#                 title=f'Funding Stages in {selected_industry}',
#                 color='Funding Stage',
#                 color_discrete_sequence=px.colors.sequential.Viridis
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Top startups in this industry
#             st.markdown("### Top Startups by Maturity Score")
#             top_startups = industry_df.sort_values('Maturity Score', ascending=False).head(5)
            
#             fig = px.bar(
#                 top_startups,
#                 x='Company Name',
#                 y='Maturity Score',
#                 title=f'Top 5 {selected_industry} Startups by Maturity Score',
#                 color='Maturity Score',
#                 color_continuous_scale='Viridis'
#             )
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Age distribution
#             st.markdown("### Age Distribution")
#             fig = px.histogram(
#                 industry_df,
#                 x='Age (Years)',
#                 nbins=10,
#                 title=f'Age Distribution of {selected_industry} Startups',
#                 color_discrete_sequence=['#1E88E5']
#             )
#             st.plotly_chart(fig, use_container_width=True)
    
#     elif page == "Funding Analysis":
#         st.markdown('<div class="sub-header">💰 Funding Analysis</div>', unsafe_allow_html=True)
        
#         # Funding stage selector
#         selected_stage = st.selectbox("Select a funding stage to analyze", 
#                                      ['All Stages'] + sorted(df['Funding Stage'].unique()))
        
#         # Filter data based on selected stage
#         if selected_stage == 'All Stages':
#             filtered_df = df
#             stage_title = "All Funding Stages"
#         else:
#             filtered_df = df[df['Funding Stage'] == selected_stage]
#             stage_title = selected_stage
        
#         # Display funding info
#         st.markdown(f"### {stage_title} Analysis")
#         st.markdown(f"**Number of Startups:** {len(filtered_df)}")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             total_funding = filtered_df['Total Funding Raised (USD)'].sum()
#             st.metric("Total Funding", f"${total_funding/1e6:.2f}M")
        
#         with col2:
#             avg_funding = filtered_df['Total Funding Raised (USD)'].mean()
#             st.metric("Average Funding", f"${avg_funding/1e6:.2f}M")
        
#         with col3:
#             avg_rounds = filtered_df['Number of Funding Rounds'].mean()
#             st.metric("Average Rounds", f"{avg_rounds:.1f}")
        
#         # Funding by industry for this stage
#         st.markdown("### Funding Distribution by Industry")
#         industry_funding = filtered_df.groupby('Primary Industry')['Total Funding Raised (USD)'].sum().sort_values(ascending=False).head(10)
        
#         fig = px.bar(
#             x=industry_funding.index,
#             y=industry_funding.values,
#             labels={'x': 'Industry', 'y': 'Total Funding (USD)'},
#             title=f'Top 10 Industries by Total Funding ({stage_title})',
#             color=industry_funding.values,
#             color_continuous_scale='Viridis'
#         )
        
#         fig.update_layout(xaxis_tickangle=-45)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Top funded startups
#         st.markdown("### Top Funded Startups")
#         top_funded = filtered_df.sort_values('Total Funding Raised (USD)', ascending=False).head(10)
        
#         fig = px.bar(
#             top_funded,
#             x='Company Name',
#             y='Total Funding Raised (USD)',
#             title=f'Top 10 Funded Startups ({stage_title})',
#             color='Primary Industry',
#             log_y=True
#         )
#         fig.update_layout(xaxis_tickangle=-45)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Funding vs team size
#         st.markdown("### Funding vs Team Size")
#         fig = px.scatter(
#             filtered_df,
#             x='Team Size (Numeric)',
#             y='Total Funding Raised (USD)',
#             color='Primary Industry',
#             size='Age (Years)',
#             hover_name='Company Name',
#             log_y=True,
#             title=f'Funding vs Team Size ({stage_title})'
#         )
#         st.plotly_chart(fig, use_container_width=True)
    
#     elif page == "Raw Data":
#         st.markdown('<div class="sub-header">📋 Raw Data</div>', unsafe_allow_html=True)
        
#         # Display raw data with filters
#         st.markdown("### Filter Data")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             industry_filter = st.multiselect(
#                 "Filter by Industry",
#                 sorted(df['Primary Industry'].unique()),
#                 default=[]
#             )
        
#         with col2:
#             stage_filter = st.multiselect(
#                 "Filter by Funding Stage",
#                 sorted(df['Funding Stage'].unique()),
#                 default=[]
#             )
        
#         # Apply filters
#         filtered_df = df.copy()
        
#         if industry_filter:
#             filtered_df = filtered_df[filtered_df['Primary Industry'].isin(industry_filter)]
        
#         if stage_filter:
#             filtered_df = filtered_df[filtered_df['Funding Stage'].isin(stage_filter)]
        
#         # Display filtered data
#         st.markdown(f"### Showing {len(filtered_df)} startups")
        
#         # Select columns to display
#         display_cols = st.multiselect(
#             "Select columns to display",
#             df.columns,
#             default=['Company Name', 'Primary Industry', 'Founding Year', 'Age (Years)', 
#                     'Total Funding Raised (USD)', 'Funding Stage', 'Maturity Score']
#         )
        
#         if display_cols:
#             st.dataframe(filtered_df[display_cols], use_container_width=True)
#         else:
#             st.dataframe(filtered_df, use_container_width=True)
        
#         # Download options
#         st.markdown("### Download Data")
        
#         if st.button("Download CSV"):
#             csv = filtered_df.to_csv(index=False)
#             b64 = base64.b64encode(csv.encode()).decode()
#             href = f'<a href="data:file/csv;base64,{b64}" download="startup_data.csv">Download CSV File</a>'
#             st.markdown(href, unsafe_allow_html=True)

# if __name__ == "__main__":
#     main()
