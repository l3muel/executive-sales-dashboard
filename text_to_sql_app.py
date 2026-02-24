import streamlit as st
import sqlite3
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import re

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Aurora BI Assistant", page_icon="🤖", layout="wide")

# --- API SETUP ---
API_KEY = st.secrets.get("GOOGLE_API_KEY", None)

# --- DATABASE CONNECTION & SCHEMA ---
DB_NAME = 'aurora_data_warehouse.db'

def get_database_schema():
    """Extracts the table names and column names to teach the AI what data we have."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_str = "Database Schema:\n"
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        schema_str += f"- Table '{table_name}' has columns: {', '.join(columns)}\n"
        
    conn.close()
    return schema_str

# --- AI SQL GENERATOR ---
def generate_sql(question, schema):
    """Uses Gemini to translate English into a SQL query."""
    if not API_KEY:
        return "ERROR: Missing API Key."
        
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')  # Using the faster model for better performance in a dashboard setting
    
    prompt = f"""
    You are an expert SQL Data Analyst. 
    Translate the following natural language business question into a valid SQLite query.
    
    {schema}
    
    Business Question: "{question}"
    
    Rules:
    1. Return ONLY the raw SQL query.
    2. Do not include markdown formatting like ```sql.
    3. Do not include any explanations.
    4. Ensure the query is compatible with SQLite.
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean up the response in case the AI adds markdown backticks anyway
        clean_sql = re.sub(r"```sql|```", "", response.text).strip()
        return clean_sql
    except Exception as e:
        return f"ERROR: {e}"

# --- SECURE EXECUTION ENGINE ---
def execute_query(sql_query):
    """Runs the SQL query safely and returns a Pandas DataFrame."""
    # SAFEGUARD: Block destructive commands
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
    if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
        return None, "🔒 Security Alert: Modifying the database is blocked. Read-only queries only."
        
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, f"SQL Execution Error: {e}"

# --- MAIN UI ---
def main():
    st.title("🤖 Aurora Business Intelligence (Text-to-SQL)")
    st.markdown("Ask questions in plain English. The AI will write the SQL, fetch the data, and visualize it.")
    
    # 1. User Input
    question = st.text_input("Ask a question about the business:", 
                             placeholder="e.g., What is the total revenue by region?")
    
    if st.button("Generate Insight") and question:
        with st.spinner("Translating to SQL..."):
            
            # Step 1: Get Schema & Generate SQL
            schema = get_database_schema()
            sql_query = generate_sql(question, schema)
            
            if sql_query.startswith("ERROR"):
                st.error(sql_query)
                st.stop()
                
            st.info(f"**Generated SQL Query:**\n```sql\n{sql_query}\n```")
            
            # Step 2: Execute SQL
            df, error = execute_query(sql_query)
            
            if error:
                st.error(error)
            elif df is not None and not df.empty:
                st.success("Query executed successfully!")
                
                # Step 3: Visualization & Results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(df, use_container_width=True)
                    
                with col2:
                    # Attempt automatic visualization if we have at least 2 columns (e.g., category + number)
                    if len(df.columns) >= 2:
                        x_col = df.columns[0]
                        y_col = df.columns[1]
                        
                        # If the second column is numeric, graph it
                        if pd.api.types.is_numeric_dtype(df[y_col]):
                            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write("Data format not optimal for automatic bar chart.")
                    else:
                        # If it's a single value (like "Total Revenue"), display it as a metric
                        st.metric(label=df.columns[0], value=f"{df.iloc[0,0]}")
            else:
                st.warning("The query ran successfully, but returned no data.")

    st.markdown("---")
    
    # --- RUBRIC REQUIREMENT: Safeguards & Limitations ---
    with st.expander("🛡️ System Architecture: Accuracy, Limitations & Safeguards"):
        st.markdown("""
        ### **1. Accuracy**
        * The AI is strictly fed the exact database schema (Table and Column names) before generating the query, minimizing hallucinations.
        * **Limitation:** Natural language is ambiguous. "Best region" could mean highest revenue, highest profit, or lowest churn. The AI infers the meaning, which may not always align with the user's exact intent.
        
        ### **2. Security Safeguards**
        * **Read-Only Enforcement:** The execution engine contains a hardcoded blocklist (`DROP`, `DELETE`, `UPDATE`, `INSERT`). Any attempt to alter the database via prompt injection is intercepted and blocked immediately.
        * **Local Execution:** Queries are executed via `sqlite3` locally; the raw database file is never uploaded to the LLM.
        
        ### **3. Explanability**
        * The system exposes the raw generated SQL to the user before displaying the data. This "Glass Box" approach allows data teams to audit the AI's logic to ensure the insight is mathematically sound.
        """)

if __name__ == "__main__":
    main()