# update_constants.py
import logging
import pandas as pd
import yfinance as yf
import plotly.express as px
from pprint import pformat # ใช้ pprint เพื่อจัด format dict ให้อ่านง่าย

# --- Import from project files ---
# ตรวจสอบให้แน่ใจว่า data_handler.py อยู่ใน path เดียวกัน
try:
    from data_handler import get_competitor_data
except ImportError:
    print("Error: Could not import get_competitor_data from data_handler.py")
    print("Please run this script from the root directory of the project.")
    exit()
    
# Import ข้อมูลเดิมจาก constants.py
try:
    from constants import SECTORS, INDEX_TICKER_TO_NAME, SECTOR_TO_INDEX_MAPPING
except ImportError:
    print("Error: Could not import from constants.py")
    print("Make sure constants.py exists in the same directory.")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sort_tickers_by_market_cap():
    logging.info("Starting ticker sorting process...")
    
    # 1. รวบรวม Tickers ทั้งหมด
    all_tickers = list(set(t for tickers in SECTORS.values() for t in tickers))
    logging.info(f"Found {len(all_tickers)} unique tickers to fetch.")

    # 2. ดึงข้อมูล Market Cap
    # นี่คือส่วนที่ช้าที่สุด (แต่ทำแค่ครั้งเดียว)
    df = get_competitor_data(tuple(all_tickers))
    
    if df.empty or 'Market Cap' not in df.columns:
        logging.error("Could not fetch market cap data. Aborting update.")
        return

    logging.info("Successfully fetched market cap data.")

    # 3. สร้าง map สำหรับการค้นหา Market Cap
    market_cap_map = df.set_index('Ticker')['Market Cap'].to_dict()

    # 4. หา 5 อันดับแรก (Top 5 Default Tickers)
    top_5_df = df.nlargest(5, 'Market Cap')
    top_5_tickers = top_5_df['Ticker'].tolist()
    logging.info(f"Top 5 Tickers: {top_5_tickers}")

    # --- [เพิ่มส่วนนี้] ---
    # 4.5. สร้างลิสต์ Ticker "ทั้งหมด" ที่เรียงตาม Market Cap
    all_tickers_df = df.nlargest(len(df), 'Market Cap')
    all_tickers_sorted = all_tickers_df['Ticker'].tolist()
    logging.info(f"Created a sorted list of all {len(all_tickers_sorted)} tickers.")
    # --- [จบส่วนที่เพิ่ม] ---

    # 5. สร้าง SECTORS dictionary ใหม่ที่เรียงลำดับแล้ว
    sorted_sectors = {}
    for sector, tickers in SECTORS.items():
        # เรียง Ticker ตาม Market Cap (จากมากไปน้อย)
        # ถ้า Ticker ไหนไม่มีข้อมูล Market Cap (ได้ 0) จะถูกย้ายไปอยู่ท้ายสุด
        sorted_list = sorted(tickers, key=lambda t: market_cap_map.get(t, 0), reverse=True)
        sorted_sectors[sector] = sorted_list
    
    logging.info("All sector lists have been sorted by market cap.")
    
    # 6. สร้าง constants ที่ต้องอ้างอิง SECTORS ใหม่
    all_possible_symbols = list(INDEX_TICKER_TO_NAME.keys())
    for sector_tickers in sorted_sectors.values():
        all_possible_symbols.extend(sector_tickers)
    all_possible_symbols = sorted(list(set(all_possible_symbols)))

    color_discrete_map = {
        symbol: color for symbol, color in zip(
            all_possible_symbols,
            px.colors.qualitative.Plotly * (len(all_possible_symbols) // 10 + 1)
        )
    }
    
    # 7. เขียนไฟล์ constants.py ใหม่ทั้งหมด
    try:
        with open('constants.py', 'w', encoding='utf-8') as f:
            f.write("# constants.py (UPDATED BY SCRIPT)\n")
            f.write("# This file centralizes all static configuration data for the application.\n\n")
            f.write("import plotly.express as px\n\n")
            
            f.write("# --- [NEW] Top 5 Tickers for default view ---\n")
            f.write(f"TOP_5_DEFAULT_TICKERS = {pformat(top_5_tickers)}\n\n")

            # --- [เพิ่มส่วนนี้] ---
            f.write("# --- [NEW] Sorted list of ALL tickers ---\n")
            f.write(f"ALL_TICKERS_SORTED_BY_MC = {pformat(all_tickers_sorted)}\n\n")
            # --- [จบส่วนที่เพิ่ม] ---

            f.write("# Dictionary mapping sectors to their respective stock tickers.\n")
            f.write("# (Sorted by Market Cap as of last script run)\n")
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
            # ใช้ pformat(color_discrete_map, indent=4) อาจจะยาวไป
            f.write("COLOR_DISCRETE_MAP = {\n")
            f.write("    symbol: color for symbol, color in zip(\n")
            f.write("        all_possible_symbols,\n")
            f.write("        px.colors.qualitative.Plotly * (len(all_possible_symbols) // 10 + 1)\n")
            f.write("    )\n")
            f.write("}\n")

        logging.info("Successfully updated and saved constants.py!")

    except Exception as e:
        logging.error(f"Failed to write new constants.py file: {e}")

if __name__ == "__main__":
    sort_tickers_by_market_cap()