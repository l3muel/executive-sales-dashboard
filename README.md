

# 📊 Executive Sales Dashboard & AI Consultant

**Capstone Project | Data Analytics & Generative AI**

## 📖 Project Overview

This project is a full-stack analytics application designed to solve the "Analysis Paralysis" problem for executives. It combines interactive **Python-based visualizations** with an **AI Strategy Consultant** (powered by Google Gemini) to deliver instant, actionable business insights from raw sales data.

**Key Features:**

* **Real-Time Analytics:** Interactive filtering by Region and Category.
* **Profitability Engine:** Instantly identifies loss-making sub-categories with red/green visual indicators.
* **AI Strategy Partner:** A Generative AI module that reads the filtered data and writes a professional executive summary.
* **Robust Data Pipeline:** Auto-corrects inconsistent date formats (`dd-mm-yyyy`) and column naming errors.

---

## 🛠️ Quick Start Guide

### 1. Prerequisites

* Python 3.8 or higher installed.
* VS Code (or any code editor).
* A Google AI Studio API Key (Free).

### 2. Installation

```bash
# Clone the repository (if downloading from GitHub)
git clone https://github.com/YOUR_USERNAME/executive-sales-dashboard.git

# Install the required libraries
pip install -r requirements.txt

```

### 3. API Configuration (Crucial!)

To unlock the AI features, you must provide your API key.

* Create a folder named `.streamlit` in the root directory.
* Create a file inside it named `secrets.toml`.
* Paste your key:
```toml
GOOGLE_API_KEY = "AIzaSy..."

```



### 4. Run the App

```bash
streamlit run app.py

```

---

## 📂 Project Files & Code

### 1. `requirements.txt`

*Dependencies required to run the app.*

```text
streamlit
pandas
plotly
numpy
google-generativeai

```

### 2. `app.py`

*The main application code. (Universal Fix Version)*

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Superstore Executive Suite",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SETUP API KEY ---
API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# --- 1. ROBUST DATA LOADER ---
@st.cache_data
def load_data():
    file_path = "Superstore.csv"
    
    if os.path.exists(file_path):
        try:
            # Load with flexible encoding
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            # Intelligent Date Parsing (Handles dd-mm-yyyy vs mm-dd-yyyy automatically)
            for col in ['Order Date', 'Ship Date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            
            # Standardize Column Names
            df.columns = [c.replace(' ', '_').replace('-', '_') for c in df.columns]
            
            # Create Year-Month column for sorting
            df['Month_Year'] = df['Order_Date'].dt.to_period('M').dt.to_timestamp()
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    return None

# --- 2. STATIC INSIGHTS ENGINE (No AI Needed) ---
def generate_static_insights(df):
    """Generates deterministic facts from the data."""
    best_month = df.groupby('Month_Year')['Sales'].sum().idxmax()
    best_month_val = df.groupby('Month_Year')['Sales'].sum().max()
    
    worst_subcat = df.groupby('Sub_Category')['Profit'].sum().idxmin()
    worst_subcat_val = df.groupby('Sub_Category')['Profit'].sum().min()
    
    top_segment = df.groupby('Segment')['Sales'].sum().idxmax()
    
    return {
        "best_month": f"{best_month.strftime('%B %Y')}",
        "best_month_val": best_month_val,
        "worst_subcat": worst_subcat,
        "worst_subcat_val": worst_subcat_val,
        "top_segment": top_segment
    }

# --- 3. AI ANALYSIS ENGINE ---
def get_ai_response(summary_text):
    if not API_KEY:
        return "⚠️ **AI Feature Locked:** Please add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`."
    
    try:
        genai.configure(api_key=API_KEY)
        # Try Flash model first (fastest), then Pro
        models = ['gemini-1.5-flash', 'gemini-pro']
        for m in models:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content(
                    f"You are a Fortune 500 Strategy Consultant. Analyze this summary and provide 3 strategic recommendations:\n{summary_text}"
                )
                return response.text
            except:
                continue
        return "❌ AI Service Unavailable currently."
    except Exception as e:
        return f"❌ Connection Error: {str(e)}"

# --- MAIN DASHBOARD UI ---
def main():
    st.title("🏢 Superstore Executive Suite")
    st.markdown("### Strategic Performance Overview")
    st.markdown("---")
    
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Data file not found. Please upload 'Superstore.csv'.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filter Panel")
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    cats = ['All'] + sorted(df['Category'].unique().tolist())
    selected_cat = st.sidebar.selectbox("Category", cats)
    
    # Filter Logic
    filtered_df = df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_cat != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_cat]

    # --- TOP ROW KPIs ---
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    total_orders = filtered_df['Order_ID'].nunique()
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Total Revenue", f"${total_sales:,.0f}")
    k2.metric("💸 Total Profit", f"${total_profit:,.0f}", delta=f"{profit_margin:.1f}% Margin")
    k3.metric("📦 Total Orders", f"{total_orders:,}")
    k4.metric("📊 Avg Order Value", f"${total_sales/total_orders:,.0f}" if total_orders else 0)

    st.markdown("---")

    # --- TABS ---
    tab_overview, tab_analysis, tab_ai = st.tabs(["📊 Dashboard", "📉 Deep Dive", "🤖 AI Consultant"])
    
    # === TAB 1: OVERVIEW ===
    with tab_overview:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📈 Revenue Trend")
            trend = filtered_df.groupby('Month_Year')['Sales'].sum().reset_index()
            fig_trend = px.area(trend, x='Month_Year', y='Sales', title="Monthly Sales Evolution")
            st.plotly_chart(fig_trend, use_container_width=True)
        with c2:
            st.subheader("🍕 Sales by Segment")
            seg_sales = filtered_df.groupby('Segment')['Sales'].sum().reset_index()
            fig_pie = px.pie(seg_sales, values='Sales', names='Segment', title="Customer Segmentation", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # === TAB 2: DEEP DIVE ===
    with tab_analysis:
        col_prof1, col_prof2 = st.columns(2)
        with col_prof1:
            st.subheader("Profit by Sub-Category")
            sub_prof = filtered_df.groupby('Sub_Category')['Profit'].sum().sort_values().reset_index()
            sub_prof['Color'] = sub_prof['Profit'].apply(lambda x: 'green' if x>0 else 'red')
            fig_sub = px.bar(sub_prof, x='Profit', y='Sub_Category', orientation='h', color='Color', color_discrete_map={'green': 'green', 'red': 'red'})
            st.plotly_chart(fig_sub, use_container_width=True)
        with col_prof2:
            st.subheader("💡 Automated Insights")
            facts = generate_static_insights(filtered_df)
            st.info(f"📅 Best Month: **{facts['best_month']}**")
            st.success(f"🏆 Top Segment: **{facts['top_segment']}**")
            if facts['worst_subcat_val'] < 0:
                st.error(f"⚠️ Loss Alert: **{facts['worst_subcat']}** is losing **${facts['worst_subcat_val']:,.0f}**")

    # === TAB 3: AI CONSULTANT ===
    with tab_ai:
        st.subheader("🤖 Artificial Intelligence Strategy Partner")
        if st.button("Generate Executive Briefing"):
            with st.spinner("Analyzing market data..."):
                summary = f"Sales: ${total_sales}, Profit: ${total_profit}, Margin: {profit_margin:.2f}%"
                response = get_ai_response(summary)
                st.markdown(response)

if __name__ == "__main__":
    main()

```

---

## 🔒 Security & Deployment

### 1. Security First

Before uploading to GitHub, create a file named `.gitignore` in your root folder and add this line:

```text
.streamlit/secrets.toml

```

**This prevents your API key from being stolen.**

### 2. Deploy to Streamlit Cloud

1. Push your code to GitHub.
2. Go to **share.streamlit.io**.
3. Connect your repository.
4. In the deployment settings, click **"Advanced Settings" -> "Secrets"**.
5. Paste your API key there:
```toml
GOOGLE_API_KEY = "AIzaSy..."

```


6. Click **Deploy**.