import argparse
import sys
import os

# Add project root to path so we can import 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import ETL functions from app.etl
try:
    from app.etl import (
        update_company_summaries, 
        update_daily_prices, 
        update_financial_statements, 
        update_news_sentiment
    )
except ImportError as e:
    print(f"Error importing app.etl: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run ETL jobs manually.")
    parser.add_argument(
        '--jobs', 
        nargs='+', 
        type=int, 
        help="List of job IDs to run (1=Summaries, 2=Prices, 3=Financials, 4=News)",
        required=True
    )
    
    args = parser.parse_args()
    
    jobs = args.jobs
    print(f"Received request to run jobs: {jobs}")
    
    if 1 in jobs:
        print("\n--- Starting Job 1: Company Summaries ---")
        try:
            update_company_summaries()
            print("--- Job 1 Completed ---")
        except Exception as e:
            print(f"Job 1 Failed: {e}")
        
    if 2 in jobs:
        print("\n--- Starting Job 2: Daily Prices ---")
        try:
            update_daily_prices()
            print("--- Job 2 Completed ---")
        except Exception as e:
             print(f"Job 2 Failed: {e}")
        
    if 3 in jobs:
         print("\n--- Starting Job 3: Financial Statements ---")
         try:
            update_financial_statements()
            print("--- Job 3 Completed ---")
         except Exception as e:
             print(f"Job 3 Failed: {e}")
        
    if 4 in jobs:
        print("\n--- Starting Job 4: News & Sentiment ---")
        try:
            update_news_sentiment()
            print("--- Job 4 Completed ---")
        except Exception as e:
            print(f"Job 4 Failed: {e}")

if __name__ == "__main__":
    main()
