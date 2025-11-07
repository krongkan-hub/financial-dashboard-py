# scripts/update_constants.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import pandas as pd
import plotly.express as px
from pprint import pformat

# --- Import from project files ---
try:
    # --- [MODIFIED] เปลี่ยนไปใช้ get_cagr_data_for_sorting สำหรับ Sort (รวม DB+Live) ---
    from app.data_handler import get_cagr_data_for_sorting 
except ImportError:
    print("Error: Could not import get_cagr_data_for_sorting from data_handler.py. (Check if data_handler.py is correct.)")
    exit()
    
try:
    from app.constants import SECTORS, INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING
except ImportError:
    print("Error: Could not import from constants.py")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sort_tickers_by_growth(): # <<< Keep function name for simplicity, but update internal logic
    logging.info("--- [START] 1. Starting Ticker Sorting Process (by Revenue CAGR (3Y) with Live Fallback) ---")
    
    # 1. รวบรวม Tickers ทั้งหมด
    all_tickers = list(set(t for tickers in SECTORS.values() for t in tickers))
    logging.info(f"--- Step 1/7: Found {len(all_tickers)} unique tickers across all sectors.")

    # 2. ดึงข้อมูล Revenue CAGR (3Y) ด้วย Fallback Logic
    DISPLAY_SORT_COLUMN = 'Revenue CAGR (3Y)' 
    logging.info(f"--- Step 2/7: Fetching {DISPLAY_SORT_COLUMN} data from DB (with Live Fallback)...")
    
    # [MODIFIED] Call the robust fallback function
    df = get_cagr_data_for_sorting(tuple(all_tickers)) 
    
    SORT_COLUMN = DISPLAY_SORT_COLUMN
    if df.empty or SORT_COLUMN not in df.columns:
        # [MODIFIED] Change the error message
        logging.error(f"--- [FATAL] Could not obtain any valid sorting data for '{SORT_COLUMN}' from DB or live fetch. Aborting update. ---")
        return

    logging.info(f"--- Step 2/7: Successfully fetched combined sorting data for {len(df)} valid tickers.")
    
    valid_ticker_set = set(df['Ticker'])

    # 3. สร้าง map สำหรับการค้นหา Growth
    growth_map = df.set_index('Ticker')[SORT_COLUMN].to_dict() 

    # 4. หา 5 อันดับแรก (Top 5 Default Tickers) 
    logging.info("--- Step 3/7: Calculating Top 5 Tickers (FIXED) ---")
    # [MODIFIED] Hardcode the list as requested
    FIXED_TOP_5_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META']
    top_5_tickers = FIXED_TOP_5_TICKERS
    logging.info(f"--- Step 3/7: Fixed Top 5 Tickers used: {top_5_tickers}")


    # 5. สร้างลิสต์ Ticker "ทั้งหมด" ที่เรียงตาม Growth
    logging.info(f"--- Step 4/7: Creating master list of all tickers, sorted by {SORT_COLUMN} (from {len(df)} data points)...")
    # [MODIFIED] Sort by CAGR column
    all_tickers_df = df.nlargest(len(df), SORT_COLUMN)
    all_tickers_sorted = all_tickers_df['Ticker'].tolist()
    logging.info(f"--- Step 4/7: Master list created with {len(all_tickers_sorted)} tickers.")

    # 6. สร้าง SECTORS dictionary ใหม่ที่เรียงลำดับแล้ว
    logging.info(f"--- Step 5/7: Re-sorting sectors internally by {SORT_COLUMN}...")
    sorted_sectors = {}
    total_removed = 0
    for sector, tickers in SECTORS.items():
        unique_tickers = list(set(tickers))
        valid_unique_tickers = [t for t in unique_tickers if t in valid_ticker_set]
        # Sort by Growth (reverse=True for highest growth first)
        # [MODIFIED] Sort by CAGR (stored in growth_map)
        sorted_list = sorted(valid_unique_tickers, key=lambda t: growth_map.get(t, -float('inf')), reverse=True) 
        sorted_sectors[sector] = sorted_list
        removed_count = len(unique_tickers) - len(valid_unique_tickers)
        if removed_count > 0:
            logging.warning(f"Sector '{sector}': Removed {removed_count} invalid/unfetched tickers during re-sort.")
            total_removed += removed_count
    logging.info(f"--- Step 5/7: Completed re-sorting {len(SECTORS)} sectors. Total {total_removed} invalid tickers removed.")

    # 7. เขียนไฟล์ constants.py ใหม่ทั้งหมด (เปลี่ยนชื่อ List)
    logging.info("--- Step 6/7: Writing new constants.py file to include ALL_TICKERS_SORTED_BY_GROWTH...")
    try:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'constants.py'))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# constants.py (UPDATED BY SCRIPT)\n")
            f.write("# This file centralizes all static configuration data for the application.\n\n")
            f.write("import plotly.express as px\n")
            f.write("from datetime import date, datetime\n\n") 
            f.write("HISTORICAL_START_DATE = datetime(2010, 1, 1)\n\n")
            
            f.write("# --- [NEW] Top 5 Tickers for default view ---\n")
            f.write(f"TOP_5_DEFAULT_TICKERS = {pformat(top_5_tickers)}\n\n")

            f.write(f"# --- [NEW] Sorted list of ALL tickers (by {SORT_COLUMN}) ---\n")
            f.write(f"ALL_TICKERS_SORTED_BY_GROWTH = {pformat(all_tickers_sorted)}\n\n")

            f.write("# Dictionary mapping sectors to their respective stock tickers.\n")
            f.write(f"# (De-duplicated, filtered for valid tickers, and sorted by {SORT_COLUMN} as of last script run)\n")
            f.write(f"SECTORS = {pformat(sorted_sectors, indent=4)}\n\n")
            
            f.write("# Dictionary mapping index tickers to their full names for display purposes.\n")
            f.write(f"INDEX_TICKER_TO_NAME = {pformat(INDEX_TICKER_TO_NAME, indent=4)}\n\n")
            
            f.write("# Dictionary mapping sectors to relevant benchmark indices.\n")
            f.write(f"SECTOR_TO_INDEX_MAPPING = {pformat(SECTOR_TO_INDEX_MAPPING, indent=4)}\n\n")

            f.write("# Generate a list of all possible symbols for color mapping\n")
            f.write("all_possible_symbols = list(INDEX_TICKER_TO_NAME.keys())\n")
            f.write("for sector_tickers in SECTORS.values():\n")
            f.write("    all_possible_symbols.extend(sector_tickers)\n")
            f.write("all_possible_symbols = sorted(list(set(all_possible_symbols)))\n\n")
            
            f.write("# Pre-defined color map for consistent chart coloring across the app.\n")
            f.write("COLOR_DISCRETE_MAP = {\n")
            f.write("    symbol: color for symbol, color in zip(\n")
            f.write("        all_possible_symbols,\n")
            f.write("        px.colors.qualitative.Plotly * (len(all_possible_symbols) // 10 + 1)\n")
            f.write("    )\n")
            f.write("}\n")

        logging.info(f"--- Step 6/7: Successfully updated and saved file to {output_path}!")

    except Exception as e:
        logging.error(f"--- [FATAL] Failed to write new constants.py file at {output_path}: {e} ---")
        return

    logging.info("--- [END] 7. Ticker Sorting Process Finished Successfully. ---")
    
if __name__ == "__main__":
    sort_tickers_by_growth()