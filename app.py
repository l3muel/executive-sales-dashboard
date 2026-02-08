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
    file_path = "Superstore.csv"  # Ensure this matches your uploaded file name
    
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
# --- 3. ADVANCED AI ENGINE ---
def get_ai_response(df):
    """
    Constructs a detailed prompt with trend analysis and sends it to Gemini.
    """
    if not API_KEY:
        return "⚠️ **AI Feature Locked:** Please add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`."

    # 1. Calculate Key Metrics
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    # 2. Identify Trends (Month over Month)
    if 'Month_Year' in df.columns:
        monthly_sales = df.groupby('Month_Year')['Sales'].sum().sort_index()
        latest_month = monthly_sales.iloc[-1]
        prev_month = monthly_sales.iloc[-2] if len(monthly_sales) > 1 else latest_month
        mom_growth = ((latest_month - prev_month) / prev_month) * 100
        trend_str = f"{mom_growth:+.1f}% vs previous month"
    else:
        trend_str = "Data unavailable"

    # 3. Identify "Problem Areas" (Sub-Categories losing money)
    loss_makers = df.groupby('Sub_Category')['Profit'].sum()
    loss_makers = loss_makers[loss_makers < 0].sort_values().head(3).index.tolist()
    loss_str = ", ".join(loss_makers) if loss_makers else "None"

    # 4. Construct the "Chief Strategy Officer" Prompt
    system_prompt = f"""
    You are the Chief Strategy Officer (CSO) of a Global Retail Company. 
    Analyze the dashboard data below and provide a high-level executive summary.
    
    **Data Context:**
    - Total Revenue: ${total_sales:,.0f}
    - Net Profit: ${total_profit:,.0f} ({profit_margin:.1f}% Margin)
    - Sales Trend: {trend_str}
    - Critical Loss-Makers (Money losing categories): {loss_str}
    - Top Performing Region: {df.groupby('Region')['Sales'].sum().idxmax()}
    
    **Your Mission:**
    1. Assess the financial health (Is the margin healthy? Is growth positive?).
    2. Identify the single biggest risk based on the loss-making categories.
    3. Propose 2 actionable strategies to improve profitability next quarter.
    
    **Format:**
    Use Markdown with emojis. Keep it concise (bullet points). 
    Do not mention "based on the data provided"—jump straight into the insights.
    """

    # 5. Call the API
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"❌ Analysis Failed: {str(e)}"

# --- MAIN DASHBOARD UI ---
def main():
    # Header
    st.title("🏢 Superstore Executive Suite")
    st.markdown("### Strategic Performance Overview")
    st.markdown("---")
    
    df = load_data()
    
    if df is None:
        st.warning("⚠️ Data file not found. Please upload 'Superstore.csv'.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filter Panel")
    
    # Region Filter
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Category Filter
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
    k1.metric("💰 Total Revenue", f"${total_sales:,.0f}", delta_color="normal")
    k2.metric("💸 Total Profit", f"${total_profit:,.0f}", delta=f"{profit_margin:.1f}% Margin")
    k3.metric("📦 Total Orders", f"{total_orders:,}")
    k4.metric("📊 Avg Order Value", f"${total_sales/total_orders:,.0f}" if total_orders else 0)

    st.markdown("---")

    # --- TABS FOR BETTER UX ---
    tab_overview, tab_analysis, tab_ai = st.tabs(["📊 Dashboard", "📉 Deep Dive", "🤖 AI Consultant"])
    
    # === TAB 1: OVERVIEW ===
    with tab_overview:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 Revenue Trend")
            # Resample by month
            trend = filtered_df.groupby('Month_Year')['Sales'].sum().reset_index()
            fig_trend = px.area(trend, x='Month_Year', y='Sales', title="Monthly Sales Evolution", color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with c2:
            st.subheader("🍕 Sales by Segment")
            seg_sales = filtered_df.groupby('Segment')['Sales'].sum().reset_index()
            fig_pie = px.pie(seg_sales, values='Sales', names='Segment', title="Customer Segmentation", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🌍 Geographic Performance")
        # Simple Bar chart for States/Regions
        state_sales = filtered_df.groupby('State')['Sales'].sum().nlargest(10).reset_index().sort_values('Sales')
        fig_map = px.bar(state_sales, x='Sales', y='State', orientation='h', title="Top 10 States by Revenue", text_auto='.2s', color='Sales', color_continuous_scale='Viridis')
        st.plotly_chart(fig_map, use_container_width=True)

    # === TAB 2: DEEP DIVE ===
    with tab_analysis:
        st.subheader("🔍 Profitability Analysis")
        col_prof1, col_prof2 = st.columns(2)
        
        with col_prof1:
            # Profit by Sub-Category (Good for finding loss makers)
            sub_prof = filtered_df.groupby('Sub_Category')['Profit'].sum().sort_values().reset_index()
            fig_sub = px.bar(sub_prof, x='Profit', y='Sub_Category', orientation='h', 
                             title="Profit by Sub-Category (Red = Loss)",
                             color='Profit', 
                             color_continuous_scale=['red', 'gray', 'green'])
            st.plotly_chart(fig_sub, use_container_width=True)
            
        with col_prof2:
            # Deterministic Insights Section
            facts = generate_static_insights(filtered_df)
            
            st.info(f"📅 **Peak Performance:** The best month was **{facts['best_month']}** with **${facts['best_month_val']:,.0f}** in sales.")
            st.success(f"🏆 **Top Customer Segment:** **{facts['top_segment']}** contributes the most revenue.")
            
            if facts['worst_subcat_val'] < 0:
                st.error(f"⚠️ **Profit Alert:** The **{facts['worst_subcat']}** category is losing money (**${facts['worst_subcat_val']:,.0f}**). Consider reviewing pricing strategy.")
            else:
                st.success("✅ All sub-categories are profitable.")

    # === TAB 3: AI CONSULTANT ===
    with tab_ai:
        st.subheader("🤖 Artificial Intelligence Strategy Partner")
        st.markdown("_This module uses Generative AI to analyze the current filter context and provide executive recommendations._")
        
        if st.button("Generate Executive Briefing"):
            with st.spinner("Analyzing market data..."):
                # Prepare data summary for AI
                summary = f"""
                Total Sales: ${total_sales}
                Total Profit: ${total_profit}
                Margin: {profit_margin:.2f}%
                Top Region: {filtered_df.groupby('Region')['Sales'].sum().idxmax()}
                Worst Performing Sub-Category: {filtered_df.groupby('Sub_Category')['Profit'].sum().idxmin()}
                """
                response = get_ai_response(summary)
                st.markdown(response)

if __name__ == "__main__":
    main()