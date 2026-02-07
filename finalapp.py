import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Executive AI Dashboard", page_icon="📊", layout="wide")

# --- SETUP API KEY ---
API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# --- 1. DATA LOADER (Adapted for your new dataset) ---
@st.cache_data
def load_data():
    file_path = "data/Global_Superstores.csv"
    
    if os.path.exists(file_path):
        try:
            # Load with flexible encoding
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            
            # --- RENAME COLUMNS TO STANDARD NAMES ---
            # We map your specific columns to generic names the app likes
            df = df.rename(columns={
                'InvoiceDate': 'Order_Date',
                'Country': 'Region',  # We'll use Country as our "Region" filter
                'Description': 'Product'
            })
            
            # --- CREATE THE MISSING "SALES" COLUMN ---
            # Sales = Quantity * Price
            df['Sales'] = df['Quantity'] * df['Price']
            
            # --- FIX DATE FORMAT ---
            df['Order_Date'] = pd.to_datetime(df['Order_Date'])
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ Could not read file: {e}. Switching to Mock Data.")
    
    # --- MOCK DATA GENERATOR (Fallback) ---
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        "Order_Date": np.random.choice(dates, 500),
        "Product": np.random.choice(["Laptop", "Chair", "Mug"], 500),
        "Region": np.random.choice(["UK", "France", "Germany"], 500),
        "Sales": np.random.uniform(100, 5000, 500)
    }
    return pd.DataFrame(data)

# --- 2. AI ANALYSIS ENGINE ---
def generate_ai_insight(summary_text):
    if not API_KEY:
        return "⚠️ **Simulation Mode:** Add GOOGLE_API_KEY to .streamlit/secrets.toml to activate AI."

    genai.configure(api_key=API_KEY)
    
    # Try models in order of preference
    models_to_try = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"You are a Senior Data Consultant. Analyze this summary:\n{summary_text}"
            )
            return response.text
        except Exception:
            continue
            
    return "❌ AI Connection Failed. Please run: pip install --upgrade google-generativeai"

# --- MAIN APP ---
def main():
    st.title("🚀 Global Retail Performance Dashboard")
    st.markdown("### AI-Powered Strategic Insights")
    
    df = load_data()
    
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    if 'Region' in df.columns:
        countries = df['Region'].unique()
        selected_countries = st.sidebar.multiselect("Select Country", countries, default=countries[:5]) # Select top 5 by default
        if selected_countries:
            filtered_df = df[df['Region'].isin(selected_countries)]
        else:
            filtered_df = df
    else:
        filtered_df = df

    # --- KPIs (Modified: Only Revenue, No Profit) ---
    total_revenue = filtered_df['Sales'].sum()
    total_orders = filtered_df.shape[0]
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Total Transactions", f"{total_orders:,}")
    c3.metric("Avg Order Value", f"${avg_order_value:.2f}")

    st.divider()

    col_charts, col_ai = st.columns([2, 1])
    
    with col_charts:
        st.subheader("📈 Revenue Trends")
        if 'Order_Date' in filtered_df.columns:
            # --- UNIVERSAL FIX (Works on ALL Pandas versions) ---
            # Create a temporary column for the month to avoid "freq" errors entirely
            filtered_df['Month_Year'] = filtered_df['Order_Date'].dt.to_period('M').dt.to_timestamp()
            
            daily = filtered_df.groupby('Month_Year')['Sales'].sum().reset_index()
            fig_line = px.line(daily, x='Month_Year', y='Sales', title="Monthly Sales Trend")
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.subheader("🏆 Top 10 Best-Selling Products")
        if 'Product' in filtered_df.columns:
            # Group by Product and take Top 10
            top_products = filtered_df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(10).reset_index()
            fig_bar = px.bar(top_products, x='Sales', y='Product', orientation='h', title="Top 10 Products by Revenue")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort bars
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with col_ai:
        st.subheader("🤖 AI Consultant")
        
        # Smart Summary for AI
        top_product_name = "N/A"
        if 'Product' in filtered_df.columns and not filtered_df.empty:
            top_product_name = filtered_df.groupby('Product')['Sales'].sum().idxmax()
            
        summary = (
            f"Total Revenue: ${total_revenue:,.2f}\n"
            f"Total Orders: {total_orders}\n"
            f"Top Selling Product: {top_product_name}\n"
            f"Countries Analyzed: {len(filtered_df['Region'].unique())}"
        )
        
        if st.button("Generate Report"):
            with st.spinner("Analyzing..."):
                st.markdown(generate_ai_insight(summary))

if __name__ == "__main__":
    main()