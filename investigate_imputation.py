# investigate_imputation.py (FINAL Indexing Fix: Using iloc for robust positional alignment)

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- WARNING: Assuming required functions can be imported or are available ---
try:
    # Import necessary utility functions
    # NOTE: These functions must be defined in run_ml_risk.py and accessible
    from run_ml_risk import fetch_data, create_target_variable, get_column
except ImportError:
    logging.error("Could not import necessary functions from run_ml_risk.py. Ensure both files are in the same directory.")
    raise SystemExit("Aborting script due to failed import.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ttm(series):
    """Calculates Trailing Twelve Months sum for a pandas Series grouped by ticker."""
    return series.rolling(window=4, min_periods=4).sum()

def analyze_imputation_source(df_ffs, df_fcs):
    """
    Performs initial feature engineering to identify where missing data originates.
    df_ffs here is the WIDE format DataFrame returned by fetch_data.
    """
    logging.info("1. Creating Target Variable and Base Features...")
    df_with_y = create_target_variable(df_ffs, df_fcs)

    if df_with_y.empty:
        logging.error("DataFrame after creating Y is empty.")
        return

    logging.info("2. Engineering Features to identify Imputation Candidates (Partial Rerun)...")
    
    # --- Rerun Feature Engineering (Partial) to recreate df_eng ---
    
    df_eng = df_with_y.copy()
    # เนื่องจาก df_with_y ถูก reset index แล้วใน create_target_variable, 
    # การ sort/reset index ที่นี่อีกครั้งจะไม่ถูกต้อง 
    # เราจะใช้ index ที่มาจากการ reset index ครั้งแรกใน create_target_variable
    df_eng = df_eng.sort_values(by=['ticker', 'report_date'])
    
    periods_per_year = 4
    
    # Define source columns for investigation (same as before)
    ca_col = get_column(df_eng, ['Current Assets'])
    cl_col = get_column(df_eng, ['Current Liabilities'])
    ta_col = get_column(df_eng, ['Total Assets'])
    inv_col = get_column(df_eng, ['Inventory'])
    liab_col = get_column(df_eng, ['Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest'])
    equity_col = get_column(df_eng, ['Stockholders Equity', 'Total Stockholders Equity'])
    ebit_col = get_column(df_eng, ['EBIT', 'Operating Income'])
    rev_col = get_column(df_eng, ['Total Revenue'])
    ni_col = get_column(df_eng, ['Net Income', 'Net Income Common Stockholders'])
    ocf_col = get_column(df_eng, ['Operating Cash Flow', 'Total Cash Flow From Operating Activities'])
    int_exp_col = get_column(df_eng, ['Interest Expense']) 

    # 1. Base Ratios / Size Features
    if ca_col and cl_col:
        df_eng['Current Ratio'] = df_eng[ca_col] / df_eng[cl_col].replace(0, np.nan)
    if ta_col:
        df_eng['Total Assets (ln)'] = np.log(df_eng[ta_col].replace(0, np.nan))
    if ebit_col and rev_col:
        df_eng['Operating Margin'] = df_eng[ebit_col] / df_eng[rev_col].replace(0, np.nan)
    # Calculate ROE for MA
    if ni_col and equity_col:
        df_eng['Net Income_TTM'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['Avg Equity_TTM'] = df_eng.groupby('ticker')[equity_col].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['ROE'] = df_eng['Net Income_TTM'] / df_eng['Avg Equity_TTM'].replace(0, np.nan)

    # 2. YoY Growth Features
    growth_col_map = {'Total Revenue': rev_col, 'Net Income': ni_col, 'Operating Cash Flow': ocf_col}
    for display_name, raw_col in growth_col_map.items():
        if raw_col:
            df_eng[f'{display_name}_YoY_Growth'] = df_eng.groupby('ticker')[raw_col].pct_change(periods=periods_per_year, fill_method=None)

    # 3. Change in Key Ratios (Over 1 Year)
    change_cols_map = {'de_ratio': 'D/E Ratio', 'Operating Margin': 'Operating Margin', 'Current Ratio': 'Current Ratio'}    
    for col_raw, col_display in change_cols_map.items():
        if col_raw in df_eng.columns: 
            df_eng[f'Change in {col_display}'] = df_eng.groupby('ticker')[col_raw].diff(periods=periods_per_year)

    # 4. Moving Averages (4 Quarters)
    ma_cols = ['Net Income', 'ROE']    
    for col in ma_cols:
        if col in df_eng.columns: 
            df_eng[f'MA_{col}'] = df_eng.groupby('ticker')[col].transform(lambda x: x.rolling(window=4, min_periods=4).mean())

    # 5. Advanced Credit Risk Features
    if ebit_col and int_exp_col:
        df_eng['Interest Coverage Ratio'] = df_eng[ebit_col] / df_eng[int_exp_col].replace(0, np.nan)
    if ca_col and inv_col and cl_col:
        df_eng['Quick Ratio'] = (df_eng[ca_col] - df_eng[inv_col]) / df_eng[cl_col].replace(0, np.nan)
    if liab_col and ebit_col:
        df_eng['Total Liabilities_TTM'] = df_eng.groupby('ticker')[liab_col].transform(calculate_ttm)
        df_eng['EBIT_TTM_4Q'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df_eng['Debt to EBITDA'] = df_eng['Total Liabilities_TTM'] / df_eng['EBIT_TTM_4Q'].replace(0, np.nan)
    if ca_col and cl_col and ta_col:
        df_eng['Net Working Capital'] = df_eng[ca_col] - df_eng[cl_col]
        df_eng['NWC_to_Total_Assets'] = df_eng['Net Working Capital'] / df_eng[ta_col].replace(0, np.nan)
    if 'Operating Margin' in df_eng.columns:
        df_eng['Op_Margin_Mean_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['Op_Margin_Std_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
        df_eng['CV_Operating_Margin'] = df_eng['Op_Margin_Std_4Q'] / df_eng['Op_Margin_Mean_4Q'].replace(0, np.nan)
    if ni_col:
        df_eng['is_negative_net_income'] = (df_eng[ni_col] < 0).astype(int)
        df_eng['Count_Negative_Net_Income_12Q'] = df_eng.groupby('ticker')['is_negative_net_income'].transform(lambda x: x.rolling(window=12, min_periods=12).sum())
    if 'Total Revenue_YoY_Growth' in df_eng.columns:
        df_eng['Sales_Volatility'] = df_eng.groupby('ticker')['Total Revenue_YoY_Growth'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
    
    # 6. Z-Score Proxies
    if ni_col and ta_col:
        df_eng['Net Income_TTM_RETA'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['RETA_Proxy'] = df_eng['Net Income_TTM_RETA'] / df_eng[ta_col].replace(0, np.nan)
    if ebit_col and ta_col:
        df_eng['EBIT_TTM_EBITTA'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df_eng['EBITTA_Proxy'] = df_eng['EBIT_TTM_EBITTA'] / df_eng[ta_col].replace(0, np.nan)
    if rev_col and ta_col:
        df_eng['Total Revenue_TTM_SATA'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['SATA_Proxy'] = df_eng['Total Revenue_TTM_SATA'] / df_eng[ta_col].replace(0, np.nan)
        
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
    
    
    feature_cols = [
        'de_ratio', 'Current Ratio', 'Operating Margin', 'Total Assets (ln)', 
        'Total Revenue_YoY_Growth', 'Net Income_YoY_Growth', 'Operating Cash Flow_YoY_Growth', 
        'Change in D/E Ratio', 'Change in Operating Margin', 'Change in Current Ratio', 
        'MA_Net Income', 'MA_ROE',
        'Interest Coverage Ratio', 'Quick Ratio', 'Debt to EBITDA', 'NWC_to_Total_Assets', 
        'CV_Operating_Margin', 'Count_Negative_Net_Income_12Q', 'Sales_Volatility',
        'RETA_Proxy', 'EBITTA_Proxy', 'SATA_Proxy' 
    ]
    
    X = df_eng[feature_cols].copy()
    
    # Filter only to rows present in the final dataset (where Y is not NaN)
    # CRITICAL FIX: Since df_with_y was reset index, we must use iloc for positional filtering.
    df_final_y_only = df_with_y.dropna(subset=['Y']).reset_index(drop=True)
    
    # Reset index of X to align with df_final_y_only's new sequential index
    X_aligned = X.reset_index(drop=True)
    
    # Filter X_aligned by the index positions (0 to 809)
    # The indices of df_final_y_only are sequential (0, 1, ..., 809)
    X_final = X_aligned.iloc[df_final_y_only.index]

    logging.info("3. Analyzing All-NaN Columns...")
    
    # Identify which columns are ALL NaN in the final filtered dataset (X_final)
    all_nan_cols = X_final.columns[X_final.isnull().all()].tolist()
    
    if not all_nan_cols:
        logging.info("No features were found to be ALL NaN. Imputation should not have been forced.")
        return

    logging.warning(f"Found {len(all_nan_cols)} ALL-NaN features which were forced to 0: {all_nan_cols}")
    
    print("\n" + "="*80)
    print(f"ALL-NAN FEATURE INVESTIGATION REPORT (Based on {len(X_final)} final rows)")
    print("="*80)
    
    # The investigation logic now runs on X_final
    for col in all_nan_cols:
        print(f"\n--- Feature: {col} ---")
        
        cause = "Unknown or Complex Calculation"
        source_metrics = []
        
        # Determine the cause and base metrics
        if 'YoY_Growth' in col or 'Change in ' in col:
            cause = f"Requires 4 periods lookback (TTM/Growth/Change) or base metric missing."
            source_metrics.append(col.replace('_YoY_Growth', '').replace('Change in ', '').replace('D/E Ratio', 'de_ratio'))
            
        elif 'MA_' in col:
            cause = "Requires 4 periods lookback (Moving Average) or base metric missing."
            source_metrics.append(col.replace('MA_', ''))

        elif 'Debt to EBITDA' == col:
            cause = "Requires 4 quarters of Total Liabilities and EBIT TTM."
            source_metrics = ['Total Liabilities', 'EBIT']

        elif 'CV_Operating_Margin' == col or 'Sales_Volatility' == col:
            cause = "Requires 4 periods of prior calculated features (Operating Margin/YoY Growth)."
            
        elif 'Count_Negative_Net_Income_12Q' == col:
             cause = "Requires 12 quarters of Net Income data."
             source_metrics = ['Net Income']

        elif 'RETA_Proxy' in col or 'EBITTA_Proxy' in col or 'SATA_Proxy' in col:
            cause = "Requires 4 quarters of Net Income/EBIT/Revenue TTM and Total Assets."
            source_metrics = ['Net Income', 'EBIT', 'Total Revenue', 'Total Assets']
            
        elif col == 'de_ratio':
            cause = "D/E Ratio data is missing entirely from FactCompanySummary for all 810 points."
            source_metrics = ['de_ratio']
            
        print(f"  Primary Cause: {cause}")
        
        # --- Check missingness in the WIDE format DF (df_ffs) ---
        if source_metrics:
            for metric in source_metrics:
                # D/E Ratio is stored as 'de_ratio' in the wide df_eng/df_ffs
                if metric == 'de_ratio':
                    raw_col_name = 'de_ratio'
                else:
                    # Find the actual raw column name used 
                    raw_col_name = get_column(df_ffs, [metric]) 
                
                # Check the raw metric availability across all rows used in the final training set (810 rows)
                if raw_col_name and raw_col_name in X_final.columns:
                    raw_na_count = X_final[raw_col_name].isnull().sum()
                    raw_total_count = len(X_final)
                    
                    if raw_na_count > 0:
                        missing_pct = raw_na_count / raw_total_count * 100
                        print(f"  -> WARNING: Base Metric '{metric}' ({raw_col_name}) is {raw_na_count}/{raw_total_count} ({missing_pct:.2f}%) missing in *Final* data rows.")
                        if raw_na_count == raw_total_count:
                             print("  -> CRITICAL: This base metric is also 100% missing in the final data rows.")
                    else:
                        print(f"  -> OK: Base Metric '{metric}' ({raw_col_name}) is fully available (0% missing).")
                else:
                    print(f"  -> CRITICAL WARNING: Base Metric '{metric}' column not found in data frame.")

    print("\n" + "="*80)
    print("RECOMMENDATION: If base metrics are 100% missing, fix the ETL process. Otherwise, the ALL-NaN is caused by the lack of historical periods (TTM/Growth/12Q).")
    print("="*80)


# --- Main Execution for Investigation ---
if __name__ == "__main__":
    logging.info("===== Starting Feature Imputation Investigation (Rerunning) =====")
    start_time = datetime.now()

    # 1. Fetch Data
    df_financials, df_summary_de = fetch_data(start_year=2010) 

    if df_financials.empty:
        logging.error("Stopping script because no financial data could be fetched.")
    else:
        # 2. Run Analysis
        analyze_imputation_source(df_financials, df_summary_de)

    end_time = datetime.now()
    logging.info(f"===== Investigation Finished in {end_time - start_time} =====")