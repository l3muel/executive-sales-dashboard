import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Universal Business Analytics",
    page_icon="🚀",
    layout="wide"
)

# --- SETUP API KEY ---
API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# --- HELPER: SIMPLE FORECASTING ---
def predict_sales(df, months=3):
    """Simple Linear Regression to forecast future sales."""
    # Group by month index (0, 1, 2...)
    monthly_data = df.groupby('Month_Year')['Sales'].sum().reset_index()
    monthly_data = monthly_data.sort_values('Month_Year')
    
    # Create numeric time index
    monthly_data['Time_Index'] = np.arange(len(monthly_data))
    
    # Fit Line (y = mx + c)
    if len(monthly_data) > 1:
        slope, intercept = np.polyfit(monthly_data['Time_Index'], monthly_data['Sales'], 1)
        
        # Generate Future Data
        last_index = monthly_data['Time_Index'].max()
        future_indices = np.arange(last_index + 1, last_index + 1 + months)
        future_sales = slope * future_indices + intercept
        
        # Create Future Dates
        last_date = monthly_data['Month_Year'].max()
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months)]
        
        forecast_df = pd.DataFrame({
            'Month_Year': future_dates,
            'Sales': future_sales,
            'Type': ['Forecast'] * months
        })
        
        history_df = monthly_data[['Month_Year', 'Sales']].copy()
        history_df['Type'] = 'History'
        
        return pd.concat([history_df, forecast_df])
    return monthly_data

# --- AI: TALK TO DATA ---
def ask_ai_about_data(df, question):
    if not API_KEY:
        return "⚠️ Please configure your Google API Key."
    
    # Create a small summary to send to AI (sending whole DB is too big)
    sample = df.head(5).to_markdown()
    columns = list(df.columns)
    stats = df.describe().to_markdown()
    
    prompt = f"""
    You are a Data Analyst Python Assistant.
    Here is the dataset schema:
    Columns: {columns}
    Sample Data:
    {sample}
    Key Statistics:
    {stats}
    
    User Question: "{question}"
    
    Answer the question based on the data summary provided. 
    If the answer requires calculation you cannot do, explain how you would calculate it.
    Keep it short and professional.
    """
    
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI Error: {e}"

# --- MAIN APP ---
def main():
    st.title("🚀 Universal Business Analytics Platform")
    st.markdown("---")

    # --- 1. SIDEBAR: DATA UPLOAD ---
    st.sidebar.header("📂 Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=['csv'])
    
    df = None
    
    # Load logic
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Custom File Loaded")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    elif os.path.exists("Superstore.csv"):
        df = pd.read_csv("Superstore.csv", encoding='ISO-8859-1')
        st.sidebar.info("Using Default Superstore Data")
    else:
        st.error("⚠️ No data found. Upload a CSV or ensure 'Superstore.csv' is in the folder.")
        st.stop()

    # --- 2. DYNAMIC COLUMN MAPPING (The Magic) ---
    # We need to know which columns are which. Ask the user!
    st.sidebar.subheader("⚙️ Column Mapping")
    all_cols = list(df.columns)
    
    # Try to guess the columns automatically
    default_date = next((c for c in all_cols if 'date' in c.lower()), all_cols[0])
    default_sales = next((c for c in all_cols if 'sales' in c.lower() or 'rev' in c.lower()), all_cols[1] if len(all_cols)>1 else all_cols[0])
    default_profit = next((c for c in all_cols if 'profit' in c.lower()), all_cols[2] if len(all_cols)>2 else all_cols[0])
    default_cat = next((c for c in all_cols if 'cat' in c.lower() or 'seg' in c.lower()), all_cols[3] if len(all_cols)>3 else all_cols[0])

    col_date = st.sidebar.selectbox("Date Column", all_cols, index=all_cols.index(default_date))
    col_sales = st.sidebar.selectbox("Sales/Revenue Column", all_cols, index=all_cols.index(default_sales))
    col_profit = st.sidebar.selectbox("Profit Column", all_cols, index=all_cols.index(default_profit))
    col_cat = st.sidebar.selectbox("Category/Segment Column", all_cols, index=all_cols.index(default_cat))

    # --- 3. DATA CLEANING & PREP ---
    try:
        # Standardize our internal names based on user selection
        clean_df = df.rename(columns={
            col_date: 'Order_Date',
            col_sales: 'Sales',
            col_profit: 'Profit',
            col_cat: 'Category'
        })
        
        # Convert Date
        clean_df['Order_Date'] = pd.to_datetime(clean_df['Order_Date'], dayfirst=True, errors='coerce')
        clean_df['Month_Year'] = clean_df['Order_Date'].dt.to_period('M').dt.to_timestamp()
        
        # Force Numerics
        clean_df['Sales'] = pd.to_numeric(clean_df['Sales'], errors='coerce').fillna(0)
        clean_df['Profit'] = pd.to_numeric(clean_df['Profit'], errors='coerce').fillna(0)
        
    except Exception as e:
        st.error(f"❌ Data Processing Failed. Check your column mapping. Error: {e}")
        st.stop()

    # --- 4. DASHBOARD TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Forecast", "🗣️ Talk to Data", "🤖 Strategy"])

    # === TAB 1: OVERVIEW ===
    with tab1:
        # KPIs
        total_sales = clean_df['Sales'].sum()
        total_profit = clean_df['Profit'].sum()
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Revenue", f"${total_sales:,.0f}")
        k2.metric("Total Profit", f"${total_profit:,.0f}")
        k3.download_button(
            label="📥 Download Clean Data",
            data=clean_df.to_csv(index=False).encode('utf-8'),
            file_name='analyzed_data.csv',
            mime='text/csv',
        )
        
        # Chart
        st.subheader("Performance by Category")
        cat_perf = clean_df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
        fig = px.bar(cat_perf, x='Category', y=['Sales', 'Profit'], barmode='group', title="Sales vs Profit")
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: FORECAST ===
    with tab2:
        st.subheader("🔮 3-Month Sales Forecast")
        st.info("Using Linear Regression Trend Analysis")
        
        forecast_data = predict_sales(clean_df)
        
        fig_cast = px.line(forecast_data, x='Month_Year', y='Sales', color='Type', 
                           title="Projected Sales Trend", 
                           color_discrete_map={'History': 'blue', 'Forecast': 'red'})
        # Add markers
        fig_cast.update_traces(mode='lines+markers')
        st.plotly_chart(fig_cast, use_container_width=True)
        
        # Show the numbers
        st.write("Predicted Numbers:", forecast_data[forecast_data['Type']=='Forecast'])

    # === TAB 3: TALK TO DATA (NLP) ===
    with tab3:
        st.subheader("🗣️ Ask Questions in Plain English")
        user_q = st.text_input("Example: What is the highest selling category? Or, Which month had the lowest profit?")
        
        if user_q:
            with st.spinner("Thinking..."):
                answer = ask_ai_about_data(clean_df, user_q)
                st.markdown(f"**Answer:** {answer}")

    # === TAB 4: STRATEGY (Your Old Logic) ===
    with tab4:
        st.subheader("🤖 Executive Briefing")
        if st.button("Generate Strategic Report"):
            with st.spinner("Consulting Strategy Engine..."):
                # Simple summary for the strategy engine
                summary_stats = f"Sales: ${total_sales}, Profit: ${total_profit}. Top Category: {cat_perf.sort_values('Sales').iloc[-1]['Category']}"
                
                try:
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel('gemini-pro')
                    res = model.generate_content(f"Act as a CSO. Provide strategic advice based on: {summary_stats}")
                    st.markdown(res.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

if __name__ == "__main__":
    main()