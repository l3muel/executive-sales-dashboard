import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Aurora Executive Command Center",
    page_icon="🌌",
    layout="wide"
)

# --- API SETUP ---
API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# --- DATA ENGINE ---
@st.cache_data
def load_aurora_data():
    df = pd.read_csv("aurora_full_dataset.csv")
    # Date Conversions
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['Month_Year'] = df['transaction_date'].dt.to_period('M').dt.to_timestamp()
    
    # Financial Cleanup
    financial_cols = ['revenue', 'profit', 'budgeted_revenue', 'actual_revenue', 'variance_revenue']
    for col in financial_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Churn Flag Logic (Convert Yes/No to 1/0)
    df['churn_binary'] = df['churn_flag'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    return df

# --- AI CORE ---
def get_cdo_insights(prompt, context):
    if not API_KEY: return "API Key Missing"
    genai.configure(api_key=API_KEY)
    
    # Using 1.5 Flash as requested for speed and intelligence
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        full_prompt = f"Context: {context}\n\nTask: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {e}"

# --- MAIN UI ---
def main():
    df = load_aurora_data()

    # SIDEBAR FILTERS
    st.sidebar.title("🌌 Aurora Filters")
    region = st.sidebar.multiselect("Select Region", options=df['region'].unique(), default=df['region'].unique())
    segment = st.sidebar.multiselect("Customer Segment", options=df['customer_segment'].unique(), default=df['customer_segment'].unique())
    
    filtered_df = df[(df['region'].isin(region)) & (df['customer_segment'].isin(segment))]

    # HEADER
    st.title("🌌 Aurora Retail: Digital Transformation Dashboard")
    st.markdown("### Strategic Oversight & AI-Powered Analytics")
    
    # --- KPI ROW ---
    total_rev = filtered_df['actual_revenue'].sum()
    total_budget = filtered_df['budgeted_revenue'].sum()
    variance = total_rev - total_budget
    churn_rate = (filtered_df['churn_binary'].sum() / len(filtered_df)) * 100
    avg_sentiment = filtered_df['sentiment_score'].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Revenue", f"${total_rev:,.0f}", delta=f"${variance:,.0f} vs Budget")
    m2.metric("Avg Sentiment", f"{avg_sentiment:.2f}", help="Customer satisfaction score")
    m3.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-1.2%", delta_color="inverse")
    m4.metric("Net Profit", f"${filtered_df['profit'].sum():,.0f}")

    st.divider()

    # --- TABS ---
    tab_performance, tab_churn, tab_ai = st.tabs(["📊 Performance Analysis", "🚨 Churn & Sentiment", "🤖 CDO Strategy AI"])

    with tab_performance:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Revenue vs Budget by Category")
            fig_var = px.bar(filtered_df.groupby('category')[['actual_revenue', 'budgeted_revenue']].sum().reset_index(), 
                             x='category', y=['actual_revenue', 'budgeted_revenue'], 
                             barmode='group', color_discrete_sequence=['#008080', '#FF6347'])
            st.plotly_chart(fig_var, use_container_width=True)
        
        with c2:
            st.subheader("Profit Trends by Region")
            fig_trend = px.area(filtered_df.groupby(['Month_Year', 'region'])['profit'].sum().reset_index(), 
                                x='Month_Year', y='profit', color='region', line_group='region')
            st.plotly_chart(fig_trend, use_container_width=True)

    with tab_churn:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Churn Probability by Segment")
            fig_churn = px.sunburst(filtered_df, path=['region', 'customer_segment'], values='churn_binary',
                                   color='churn_binary', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with c4:
            st.subheader("Sentiment vs. Profitability")
            fig_scat = px.scatter(filtered_df, x='sentiment_score', y='profit', color='customer_segment', 
                                  size='actual_revenue', hover_name='customer_name')
            st.plotly_chart(fig_scat, use_container_width=True)

    with tab_ai:
        st.subheader("🤖 Chief Digital Officer: Strategic Briefing")
        if st.button("Generate AI Strategy Report"):
            with st.spinner("Analyzing Aurora metrics..."):
                stats_summary = f"""
                Revenue: ${total_rev}, Budget: ${total_budget}, Variance: ${variance}.
                Churn Rate: {churn_rate:.2f}%, Avg Sentiment: {avg_sentiment:.2f}.
                Top Region: {filtered_df.groupby('region')['actual_revenue'].sum().idxmax()}.
                """
                prompt = "Act as a CDO. Analyze these retail metrics. Identify the biggest risk and suggest 3 high-impact digital strategies to improve profit."
                report = get_cdo_insights(prompt, stats_summary)
                st.markdown(report)

if __name__ == "__main__":
    main()