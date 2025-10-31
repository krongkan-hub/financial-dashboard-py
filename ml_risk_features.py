import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

# --- Helper Functions (คัดลอกจาก run_ml_risk.py) ---

def calculate_ttm(series):
    """Calculates Trailing Twelve Months (TTM) sum."""
    return series.rolling(window=4, min_periods=4).sum()

def calculate_change(series, periods):
    """Calculates change over a specified number of periods."""
    return series.diff(periods=periods)

def get_column(df, possible_names):
    """Safely gets a column name from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    log.debug(f"Could not find any of {possible_names} in DataFrame.")
    return None

# --- Feature Contract (คัดลอกจาก run_ml_risk.py V5, line 343) ---
# นี่คือ "สัญญา" 22 Base Features ที่โมเดลคาดหวัง (ก่อนการ Impute)
ML_RISK_BASE_FEATURES = [
    'Current Ratio', 'Operating Margin', 'Total Assets (ln)', 
    'Total Revenue_YoY_Growth', 'Net Income_YoY_Growth', 'Operating Cash Flow_YoY_Growth', 
    'Change in Operating Margin', 'Change in Current Ratio', 
    'MA_Net Income', 'MA_ROE',
    'Interest Coverage Ratio', 'Quick Ratio', 'Debt to EBITDA', 'NWC_to_Total_Assets', 
    'CV_Operating_Margin', 'Count_Negative_Net_Income_12Q', 'Sales_Volatility',
    'RETA_Proxy', 'EBITTA_Proxy', 'SATA_Proxy',
    'WCT', 
    'CCC' 
]

def engineer_features_for_prediction(df_raw):
    """
    สร้าง 22 Base Features ดิบ (Raw Features) จากข้อมูลล่าสุด
    (ดัดแปลงจาก logic ใน engineer_features() ของ run_ml_risk.py)
    
    คืนค่า:
    - X_features (DataFrame ที่มี 22 คอลัมน์ ตรงตาม ML_RISK_BASE_FEATURES)
    - tickers (List ของ Ticker ที่เรียงลำดับตรงกัน)
    """
    log.info("Engineering 22 base features for prediction...")
    
    df_eng = df_raw.copy()
    df_eng = df_eng.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)
    periods_per_year = 4
    
    # --- Robust Column Selection ---
    ca_col = get_column(df_eng, ['Current Assets'])
    cl_col = get_column(df_eng, ['Current Liabilities'])
    ta_col = get_column(df_eng, ['Total Assets'])
    inv_col = get_column(df_eng, ['Inventory'])
    liab_col = get_column(df_eng, ['Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest'])
    equity_col = get_column(df_eng, ['Stockholders Equity', 'Total Stockholders Equity'])
    ebit_col = get_column(df_eng, ['EBIT', 'Operating Income'])
    rev_col = get_column(df_eng, ['Total Revenue', 'Sales'])
    ni_col = get_column(df_eng, ['Net Income', 'Net Income Common Stockholders'])
    ocf_col = get_column(df_eng, ['Operating Cash Flow', 'Total Cash Flow From Operating Activities'])
    int_exp_col = get_column(df_eng, ['Interest Expense']) 
    ar_col = get_column(df_eng, ['Accounts Receivable'])
    ap_col = get_column(df_eng, ['Accounts Payable'])
    cor_col = get_column(df_eng, ['Cost Of Revenue', 'Cost Of Goods Sold'])

    # --- 1. Base Ratios / Size Features ---
    if ca_col and cl_col: df_eng['Current Ratio'] = df_eng[ca_col] / df_eng[cl_col].replace(0, np.nan)
    else: df_eng['Current Ratio'] = np.nan
    if ta_col: df_eng['Total Assets (ln)'] = np.log(df_eng[ta_col].replace(0, np.nan))
    else: df_eng['Total Assets (ln)'] = np.nan
    if ebit_col and rev_col: df_eng['Operating Margin'] = df_eng[ebit_col] / df_eng[rev_col].replace(0, np.nan)
    else: df_eng['Operating Margin'] = np.nan
    if ni_col and equity_col:
        df_eng['Net Income_TTM'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['Avg Equity_TTM'] = df_eng.groupby('ticker')[equity_col].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['ROE'] = df_eng['Net Income_TTM'] / df_eng['Avg Equity_TTM'].replace(0, np.nan)
    else: df_eng['ROE'] = np.nan

    # --- 2. YoY Growth Features ---
    growth_col_map = {'Total Revenue': rev_col, 'Net Income': ni_col, 'Operating Cash Flow': ocf_col}
    for display_name, raw_col in growth_col_map.items():
        if raw_col: df_eng[f'{display_name}_YoY_Growth'] = df_eng.groupby('ticker')[raw_col].pct_change(periods=periods_per_year, fill_method=None)
        else: df_eng[f'{display_name}_YoY_Growth'] = np.nan

    # --- 3. Change in Key Ratios (Over 1 Year) ---
    change_cols_map = {'Operating Margin': 'Operating Margin', 'Current Ratio': 'Current Ratio'}    
    for col_raw, col_display in change_cols_map.items():
        if col_raw in df_eng.columns: df_eng[f'Change in {col_display}'] = df_eng.groupby('ticker')[col_raw].transform(calculate_change, periods=periods_per_year)
        else: df_eng[f'Change in {col_display}'] = np.nan

    # --- 4. Moving Averages (4 Quarters) ---
    ma_cols = ['Net Income', 'ROE']    
    for col in ma_cols:
        if col in df_eng.columns: df_eng[f'MA_{col}'] = df_eng.groupby('ticker')[col].transform(lambda x: x.rolling(window=4, min_periods=1).mean()) # ใช้อย่างน้อย 1 period
        else: df_eng[f'MA_{col}'] = np.nan

    # --- 5. Advanced Credit Risk Features (7 Features) ---
    if ebit_col and int_exp_col: df_eng['Interest Coverage Ratio'] = df_eng[ebit_col] / df_eng[int_exp_col].replace(0, np.nan)
    else: df_eng['Interest Coverage Ratio'] = np.nan
    if ca_col and inv_col and cl_col: df_eng['Quick Ratio'] = (df_eng[ca_col] - df_eng[inv_col]) / df_eng[cl_col].replace(0, np.nan)
    else: df_eng['Quick Ratio'] = np.nan
    if liab_col and ebit_col:
        df_eng['Total Liabilities_TTM'] = df_eng.groupby('ticker')[liab_col].transform(calculate_ttm)
        df_eng['EBIT_TTM_4Q'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm) 
        df_eng['Debt to EBITDA'] = df_eng['Total Liabilities_TTM'] / df_eng['EBIT_TTM_4Q'].replace(0, np.nan)
    else: df_eng['Debt to EBITDA'] = np.nan
    if ca_col and cl_col and ta_col:
        df_eng['Net Working Capital'] = df_eng[ca_col] - df_eng[cl_col]
        df_eng['NWC_to_Total_Assets'] = df_eng['Net Working Capital'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['NWC_to_Total_Assets'] = np.nan
    if 'Operating Margin' in df_eng.columns:
        df_eng['Op_Margin_Mean_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
        df_eng['Op_Margin_Std_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=1).std())
        df_eng['CV_Operating_Margin'] = df_eng['Op_Margin_Std_4Q'] / df_eng['Op_Margin_Mean_4Q'].replace(0, np.nan)
    else: df_eng['CV_Operating_Margin'] = np.nan
    if ni_col:
        df_eng['is_negative_net_income'] = (df_eng[ni_col] < 0).astype(int)
        df_eng['Count_Negative_Net_Income_12Q'] = df_eng.groupby('ticker')['is_negative_net_income'].transform(lambda x: x.rolling(window=12, min_periods=1).sum())
    else: df_eng['Count_Negative_Net_Income_12Q'] = np.nan
    if 'Total Revenue_YoY_Growth' in df_eng.columns:
        df_eng['Sales_Volatility'] = df_eng.groupby('ticker')['Total Revenue_YoY_Growth'].transform(lambda x: x.rolling(window=4, min_periods=1).std())
    else: df_eng['Sales_Volatility'] = np.nan
    
    # --- 6. Altman Z-Score Components Proxies (3 Features) ---
    if ni_col and ta_col:
        df_eng['Net Income_TTM_RETA'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['RETA_Proxy'] = df_eng['Net Income_TTM_RETA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['RETA_Proxy'] = np.nan
    if ebit_col and ta_col:
        df_eng['EBIT_TTM_EBITTA'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df_eng['EBITTA_Proxy'] = df_eng['EBIT_TTM_EBITTA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['EBITTA_Proxy'] = np.nan
    if rev_col and ta_col:
        df_eng['Total Revenue_TTM_SATA'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['SATA_Proxy'] = df_eng['Total Revenue_TTM_SATA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['SATA_Proxy'] = np.nan
    
    # --- 7. Working Capital Turnover (WCT) ---
    if 'Net Working Capital' in df_eng.columns and rev_col:
        df_eng['Total Revenue_TTM'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['WCT'] = df_eng['Total Revenue_TTM'] / (df_eng['Net Working Capital'].replace(0, np.nan) + 1e-6)
    else:
        df_eng['WCT'] = np.nan

    # --- 8. Cash Conversion Cycle (CCC) ---
    if cor_col and inv_col:
        df_eng['COR_TTM'] = df_eng.groupby('ticker')[cor_col].transform(calculate_ttm)
        df_eng['Avg_Inventory'] = df_eng.groupby('ticker')[inv_col].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        df_eng['DIO'] = (df_eng['Avg_Inventory'] / df_eng['COR_TTM'].replace(0, np.nan)) * 365
    else: df_eng['DIO'] = np.nan
    if ar_col and rev_col:
        df_eng['REV_TTM'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['Avg_AR'] = df_eng.groupby('ticker')[ar_col].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        df_eng['DSO'] = (df_eng['Avg_AR'] / df_eng['REV_TTM'].replace(0, np.nan)) * 365
    else: df_eng['DSO'] = np.nan
    if ap_col and cor_col:
        df_eng['COR_TTM_AP'] = df_eng.groupby('ticker')[cor_col].transform(calculate_ttm)
        df_eng['Avg_AP'] = df_eng.groupby('ticker')[ap_col].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        df_eng['DPO'] = (df_eng['Avg_AP'] / df_eng['COR_TTM_AP'].replace(0, np.nan)) * 365
    else: df_eng['DPO'] = np.nan
    
    df_eng['CCC'] = df_eng['DSO'] + df_eng['DIO'] - df_eng['DPO']
    
    # --- Final Data Prep ---
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
    
    # ดึง Tickers และ Tickers List ออกมาก่อน
    tickers = df_eng['ticker'].tolist()
    
    # ตรวจสอบว่ามีคอลัมน์ทั้งหมด 22 features หรือไม่
    X_features = pd.DataFrame(index=df_eng.index)
    for col in ML_RISK_BASE_FEATURES:
        if col in df_eng.columns:
            X_features[col] = df_eng[col]
        else:
            log.warning(f"Feature '{col}' not found during prediction engineering. Filling with NaN.")
            X_features[col] = np.nan # เติม NaN ถ้าไม่มี (Imputer จะจัดการต่อ)

    # คืนค่าเฉพาะ 22 features ที่เรียงลำดับถูกต้อง และ list ของ tickers
    return X_features[ML_RISK_BASE_FEATURES], tickers