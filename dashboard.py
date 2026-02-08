import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from openai import OpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Executive AI Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- SETUP & CONSTANTS ---
# Replace with your preferred API setup. 
# For production, utilize st.secrets["OPENAI_API_KEY"]
API_KEY = st.secrets.get("OPENAI_API_KEY", None) 

# --- 1. DATA LOADER ---
@st.cache_data
def load_data():
    """
    Attempts to load 'data/Global_Superstore.csv'.
    If not found, generates a realistic mock dataset for demonstration.
    """
    file_path = "data/Global_Superstore.csv"
    
    if os.path.exists(file_path):
        try:
            # Adjust encoding/dates as necessary for the specific CSV
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            return df
        except Exception as e:
            st.warning(f"Error reading CSV: {e}. Loading demo data instead.")
    
    # --- MOCK DATA GENERATOR (For immediate portfolio display) ---
    dates = pd.date_range(start="2023-01-01", periods=100)
    categories = ["Technology", "Furniture", "Office Supplies"]
    regions = ["North", "South", "East", "West"]
    
    data = {
        "Order Date": np.random.choice(dates, 500),
        "Category": np.random.choice(categories, 500),
        "Region": np.random.choice(regions, 500),
        "Sales": np.random.uniform(100, 5000, 500),
        "Profit": np.random.uniform(-50, 1000, 500)
    }
    return pd.DataFrame(data)

# --- 2. AI ANALYSIS ENGINE ---
def generate_ai_insight(summary_text):
    """
    Sends the data summary to the LLM to get an executive brief.
    """
    if not API_KEY:
        return """
        **⚠️ AI Simulation Mode (No API Key Found)**
        
        ### 📊 Executive Summary
        Performance remains strong with Technology leading revenue generation.
        
        ### 🚀 Key Opportunities
        * **Furniture in the East** is showing a **15%** upward trend.
        * **Office Supplies** require inventory optimization.
        
        *(To enable real AI analysis, add your OpenAI API Key to .streamlit/secrets.toml)*
        """

    client = OpenAI(api_key=API_KEY)
    
    system_prompt = """
    You are a Senior Data Strategy Consultant. 
    Analyze the provided data summary.
    Format your response in Markdown with these headers:
    ### 📊 Executive Summary
    ### 🚀 Key Performance Drivers
    ### ⚠️ Critical Risks
    ### 💡 Strategic Recommendation
    Keep it professional, concise, and actionable.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this dataset summary:\n{summary_text}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {str(e)}"

# --- MAIN APP LOGIC ---
def main():
    st.title("🚀 Enterprise Performance Dashboard")
    st.markdown("### AI-Powered Strategic Insights")
    
    # Load Data
    df = load_data()
    
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    selected_region = st.sidebar.multiselect("Select Region", df['Region'].unique(), default=df['Region'].unique())
    filtered_df = df[df['Region'].isin(selected_region)]

    # Top Level KPIs
    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    margin = (total_profit / total_sales) * 100
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Revenue", f"${total_sales:,.0f}")
    kpi2.metric("Total Profit", f"${total_profit:,.0f}")
    kpi3.metric("Profit Margin", f"{margin:.1f}%")

    st.markdown("---")

    # Layout: Charts on Left, AI on Right
    col_charts, col_ai = st.columns([2, 1])

    with col_charts:
        st.subheader("📈 Revenue Trends")
        # Time Series Chart
        daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        fig_line = px.line(daily_sales, x='Order Date', y='Sales', template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.subheader("🏆 Category Performance")
        # Bar Chart
        cat_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig_bar = px.bar(cat_sales, x='Category', y='Sales', color='Sales', template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_ai:
        st.subheader("🤖 Executive Briefing")
        
        # Prepare data for AI (Summarized to save tokens)
        data_summary = f"""
        Total Revenue: ${total_sales}
        Total Profit: ${total_profit}
        Profit Margin: {margin:.1f}%
        Top Performing Category: {cat_sales.sort_values('Sales', ascending=False).iloc[0]['Category']}
        Region(s) Analyzed: {", ".join(selected_region)}
        """
        
        if st.button("Generate AI Report"):
            with st.spinner("Consulting AI Analyst..."):
                insight = generate_ai_insight(data_summary)
                st.markdown(insight)
        else:
            st.info("Click 'Generate AI Report' to analyze current filters.")

if __name__ == "__main__":
    main()