import streamlit as st
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tempfile

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Competitor Research Assistant", layout="wide")

st.title("🔍 AI & Auto Company Competitor Research Assistant")
st.markdown("Helping PMs quickly understand how a startup stacks up against the competition.")

# -----------------------------
# Load & Clean Data
# -----------------------------
@st.cache_data
def load_company_data():
    df = pd.read_csv("company_competitor_data.csv")
    key_metrics = [
        "Revenue Growth YoY (%)",
        "Net Profit Margin (%)",
        "Return on Equity (%)",
        "Operating Margin (%)",
        "Market Share (%)"
    ]
    df_clean = df.dropna(subset=key_metrics, how='all')
    return df_clean

# For display purposes
def tidy_table(df):
    cleaned = df.dropna(axis=1, how='all')
    return cleaned.fillna("N/A").astype(str)

# -----------------------------
# Chart Generation
# -----------------------------
def create_metric_chart(data, company_name, metric):
    plt.figure(figsize=(6, 3))
    sns.barplot(x="Company", y=metric, data=data)
    plt.axhline(y=data[data["Company"] == company_name][metric].values[0], color="red", linestyle="--", label="Selected Company")
    plt.title(f"{metric} Comparison")
    plt.xticks(rotation=45)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return buffer

# -----------------------------
# PDF Report Logic
# -----------------------------
class CompetitorPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Company Competitor Analysis Report", ln=True, align="C")
        self.ln(5)

    def add_company_details(self, row):
        self.set_font("Arial", "", 12)
        for col, val in row.items():
            self.multi_cell(0, 8, f"{col}: {val}", align="L")
        self.ln(5)

    def add_metric_image(self, image_data):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(image_data.read())
            temp_path = tmp_img.name
        self.image(temp_path, w=180)

# -----------------------------
# App Logic
# -----------------------------
data = load_company_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

search_input = st.sidebar.text_input("Search for a Company")
selected_company = st.sidebar.selectbox("Choose a Company", ["All"] + sorted(data["Company"].dropna().unique()))
selected_sector = st.sidebar.selectbox("Sector", ["All"] + sorted(data["Sector"].dropna().unique()))

# Apply filters
filtered = data.copy()
if search_input:
    filtered = filtered[filtered["Company"].str.contains(search_input, case=False, na=False)]
if selected_company != "All":
    filtered = filtered[filtered["Company"] == selected_company]
if selected_sector != "All":
    filtered = filtered[filtered["Sector"] == selected_sector]

# Show filtered data
display_data = tidy_table(filtered)
st.subheader(f"📊 {len(display_data)} Company(ies) Found")
st.dataframe(display_data)

if st.checkbox("Show Full Dataset"):
    st.write("### Raw Dataset")
    st.dataframe(data)

# -----------------------------
# Generate PDF if 1 Company is Selected
# -----------------------------
if len(filtered) == 1:
    st.subheader("📄 Generate Company Report")

    if st.button("Generate & Download PDF"):
        picked_company = filtered.iloc[0]
        pdf = CompetitorPDF()
        pdf.add_page()
        pdf.add_company_details(picked_company)

        important_metrics = ["Revenue Growth YoY (%)", "Net Profit Margin (%)", "Return on Equity (%)"]
        for metric in important_metrics:
            if metric in filtered.columns and filtered[metric].notna().sum() > 1:
                chart_img = create_metric_chart(filtered.dropna(subset=[metric]), picked_company["Company"], metric)
                pdf.add_metric_image(chart_img)

        report_bytes = pdf.output(dest='S').encode('latin1')
        st.download_button(
            "📥 Download Report", 
            data=report_bytes, 
            file_name=f"{picked_company['Company']}_competitor_report.pdf", 
            mime="application/pdf"
        )
else:
    st.info("👈 Please narrow down to one specific company to enable PDF report generation.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("💡 *Made for PMs to size up competition in seconds.*")


#streamlit Practise 

# import streamlit as st

# st.title("Hello DUDE 😎 ")
# st.write("This is Sathvik from NST ")

# if st.button('Click me'):
#     st.success("You click the right button 👾 ")
# name = st.text_input("Enter your name here please")
# if name:
#     st.write(f"Hello my brother {name}! 👋🏻")

# option = st.selectbox("Choose your favorite language", ["Python", "JavaScript", "Go", "Rust"])
# st.write(f"You have selected ur fav language as: {option}")

# import pandas as pd
# import numpy as np

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=["A", "B", "C"]
# )

# st.line_chart(chart_data)
# col1, col2 = st.columns(2)

# with col1:
#     st.write("👈 This is the left column")
# with col2:
#     st.write("👉 This is the right column")
# # kpis
# app.py
# Importing necessary libraries

# app.py
# app.py
# app.py
