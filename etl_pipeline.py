import pandas as pd
import sqlite3
import logging

# Set up logging to track the pipeline's health (Good for your CompTIA/Security background)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_aurora_pipeline():
    logging.info("🚀 Starting Aurora ETL Pipeline...")

    # --- 1. EXTRACT ---
    logging.info("📥 Phase 1: Extracting raw data...")
    try:
        raw_df = pd.read_csv("aurora_full_dataset.csv")
        logging.info(f"Loaded {len(raw_df)} rows successfully.")
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return

    # --- 2. TRANSFORM ---
    logging.info("⚙️ Phase 2: Transforming data...")
    
    # Clean Dates
    raw_df['transaction_date'] = pd.to_datetime(raw_df['transaction_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Ensure Financials are Numeric (handling potential dirty data)
    financials = ['revenue', 'profit', 'cost', 'actual_revenue', 'budgeted_revenue', 'sentiment_score']
    for col in financials:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)

    # Standardize Churn Flag for machine learning readiness
    raw_df['churn_binary'] = raw_df['churn_flag'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    # --- BUILD STAR SCHEMA (Dimensional Modeling) ---
    logging.info("🌟 Structuring Star Schema...")
    
    # Table 1: Dim_Customer (Unique customers and their traits)
    dim_customer = raw_df[['customer_id', 'customer_name', 'customer_segment', 'region']].drop_duplicates(subset=['customer_id'])
    
    # Table 2: Dim_Product (Unique products and categories)
    dim_product = raw_df[['product_id', 'product_name', 'category', 'brand']].drop_duplicates(subset=['product_id'])
    
    # Table 3: Fact_Sales (The core transaction data)
    fact_sales = raw_df[['transaction_id', 'transaction_date', 'customer_id', 'product_id', 
                         'channel', 'quantity', 'revenue', 'profit', 'budgeted_revenue', 
                         'actual_revenue', 'churn_binary', 'sentiment_score']]

    # --- 3. LOAD ---
    logging.info("💾 Phase 3: Loading into Data Warehouse...")
    try:
        # This creates a local database file automatically!
        conn = sqlite3.connect('aurora_data_warehouse.db')
        
        # Write tables to the database
        dim_customer.to_sql('dim_customer', conn, if_exists='replace', index=False)
        dim_product.to_sql('dim_product', conn, if_exists='replace', index=False)
        fact_sales.to_sql('fact_sales', conn, if_exists='replace', index=False)
        
        conn.close()
        logging.info("✅ Pipeline Complete! Database 'aurora_data_warehouse.db' is ready.")
        
    except Exception as e:
        logging.error(f"Load failed: {e}")

if __name__ == "__main__":
    run_aurora_pipeline()