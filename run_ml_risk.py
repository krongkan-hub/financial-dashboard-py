# run_ml_risk.py (เวอร์ชัน FINAL-FIX V5: Stricter Precision & Y Definition)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import joblib 
import shap 
from xgboost import XGBClassifier 
from data_handler import get_ml_risk_raw_data

# --- Database Interaction ---
# NOTE: ต้องมั่นใจว่า app.py, FactFinancialStatements, FactCompanySummary ได้ถูก import อย่างถูกต้องในสภาพแวดล้อมของคุณ
from app import db, server, FactFinancialStatements, FactCompanySummary
from sqlalchemy import func, and_
from sqlalchemy.orm import aliased

# --- Machine Learning ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer, fbeta_score, precision_recall_curve 
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for Target Variable Definition (***EDITED: Stricter Y Definition***)
DE_RATIO_THRESHOLD = 7.0 # ปรับจาก 5.0 -> 7.0
ICR_THRESHOLD = 1.0
CONSECUTIVE_LOSS_PERIODS = 8 # ปรับจาก 6 -> 8

# --- V2.0 RATING MAP ---
# Rating Mapping (Simplified S&P/Fitch Scale) - Higher number = Higher Risk (Worse Rating)
RATING_MAP = {
    'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
    'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10,
    'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16,
    'CCC+': 17, 'CCC': 18, 'CCC-': 19,
    'CC': 20, 'C': 21, 'D': 22, # Default/Worst Rating
    'NR': 99, 'N/A': 99, np.nan: 99, 'None': 99, '': 99 # Not Rated / Missing
}
# --- END V2.0 RATING MAP ---

# Feature Engineering Constants
MA_PERIODS = 4 

# Model Saving Configuration
MODEL_DIR = "models"
MODEL_FILENAME = "trained_risk_model_xgb.joblib" 
SCALER_FILENAME = "scaler.joblib" 
IMPUTER_FILENAME = "imputer.joblib" 

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper Functions (Unchanged) ---
def calculate_ttm(series):
    return series.rolling(window=4, min_periods=4).sum()

def calculate_change(series, periods):
    return series.diff(periods=periods)

def get_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None 

# --- Core Functions ---

def create_target_variable(df_ffs, df_fcs): # df_fcs is now largely unused for V2.0 target logic
    """Creates the binary target variable (Y) based on defined V2.0 risk criteria: 
    Y_Rating (Downgrade) OR Y_Solvency (Accumulated Deficit > 0 AND Current Ratio < 1.0)."""
    logging.info("2. Creating Target Variable (Y) using V2.0 Solvency and Rating Downgrade criteria...")
    if df_ffs.empty:
        logging.warning("Financial statement data is empty, cannot create target variable.")
        return pd.DataFrame()

    df = df_ffs.copy()
    # เนื่องจาก `df_ffs` ใน V2.0 มาจาก get_ml_risk_raw_data ซึ่งควรมีข้อมูลดิบทั้งหมดแล้ว
    df = df.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)

    # --- V2.0 Target 1: Y_Rating (Downgrade in next 4 quarters) ---
    if 'credit_rating' in df.columns:
        # 1. Convert rating string to numerical risk score (ใช้ RATING_MAP ที่เพิ่มด้านบน)
        df['rating_score'] = df['credit_rating'].astype(str).str.upper().map(RATING_MAP).fillna(99)
        
        # 2. Shift score back 4 periods to get the future rating score for the current date
        df['future_rating_score'] = df.groupby('ticker')['rating_score'].shift(-4)
        
        # 3. Y_Rating = 1 ถ้ามีการ Downgrade (future_score > current_score) และไม่ใช่ "Not Rated"
        # The higher the score, the worse the rating (e.g., A (6) -> A- (7) is a downgrade)
        rule_rating_downgrade = (df['future_rating_score'] > df['rating_score']) & (df['rating_score'] < 99)
        df['Y_Rating_raw'] = rule_rating_downgrade.astype(int)
    else:
        logging.warning("Missing 'credit_rating' column. Y_Rating cannot be calculated.")
        df['Y_Rating_raw'] = 0

    # --- V2.0 Target 2: Y_Solvency (Accumulated Deficit > 0 AND Current Ratio < 1.0) ---
    # ใช้ชื่อคอลัมน์ที่ถูก Pivot แล้วใน data_handler.py
    accumulated_deficit_col = get_column(df, ['Accumulated Deficit', 'Accumulated_Deficit'])
    current_ratio_col = get_column(df, ['Current Ratio', 'Current_Ratio'])
    
    if accumulated_deficit_col and current_ratio_col:
        # Rule Solvency: (Accumulated Deficit > 0) AND (Current Ratio < 1.0)
        rule_solvency = (df[accumulated_deficit_col] > 0) & (df[current_ratio_col] < 1.0)
        df['Y_Solvency_raw'] = rule_solvency.astype(int)
    else:
        logging.warning("Missing 'Accumulated Deficit' or 'Current Ratio' columns. Y_Solvency cannot be calculated.")
        df['Y_Solvency_raw'] = 0

    # --- Final Target (Y) using OR logic ---
    df['Y_raw_v2'] = ((df['Y_Rating_raw'] == 1) | (df['Y_Solvency_raw'] == 1)).astype(int)

    # Introduce Time Lag (4 quarters) - Y_raw_v2 is the combined target
    lag_periods = 4
    df['Y'] = df.groupby('ticker')['Y_raw_v2'].shift(-lag_periods) 
    logging.info(f"Target variable Y (V2.0) created with a lag of {lag_periods} quarters.")

    df_final = df.dropna(subset=['Y']).copy()
    df_final['Y'] = df_final['Y'].astype(int)

    logging.info(f"Created Y for {len(df_final)} data points.")
    logging.info(f"Distribution of Y: \n{df_final['Y'].value_counts(normalize=True)}")

    # Cleanup temporary columns and ensure correct columns are passed to the next stage (engineer_features)
    explicit_cols = ['ticker', 'report_date', 'Y']
    # Keep all raw financial/market data columns from the original input
    cols_to_keep_from_original = [col for col in df_ffs.columns if col not in ['ticker', 'report_date']]
    
    cols_to_keep = explicit_cols + cols_to_keep_from_original
    cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]

    df_final = df_final[cols_to_keep]
    # Drop temporary columns used only for calculation
    df_final = df_final.drop(columns=['rating_score', 'future_rating_score', 'Y_Rating_raw', 'Y_Solvency_raw', 'Y_raw_v2'], errors='ignore') 

    # Note: If 'de_ratio' exists in df_ffs (from the raw data merge), it will be kept, which is fine for feature engineering in the next step.
    
    return df_final


def engineer_features(df):
    """
    Calculates 20+ features, handles infinities/NaNs, and applies scaling/imputation 
    WITH INDICATOR FEATURE FIX. (Unchanged logic)
    """
    logging.info("3. Engineering Features (X)...")
    if df.empty or 'Y' not in df.columns:
        logging.warning("Input DataFrame is empty or missing Y column.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []

    df_eng = df.copy()
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
    
    # --- [NEW V2.0 Market Data Columns] ---
    price_col = get_column(df_eng, ['closing_price']) 
    mc_col = get_column(df_eng, ['market_cap'])
    so_col = get_column(df_eng, ['shares_outstanding'])
    # --- [END NEW V2.0 Market Data Columns] ---

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
        if col in df_eng.columns: df_eng[f'MA_{col}'] = df_eng.groupby('ticker')[col].transform(lambda x: x.rolling(window=MA_PERIODS, min_periods=MA_PERIODS).mean())
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
        df_eng['Op_Margin_Mean_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['Op_Margin_Std_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
        df_eng['CV_Operating_Margin'] = df_eng['Op_Margin_Std_4Q'] / df_eng['Op_Margin_Mean_4Q'].replace(0, np.nan)
    else: df_eng['CV_Operating_Margin'] = np.nan
    if ni_col:
        df_eng['is_negative_net_income'] = (df_eng[ni_col] < 0).astype(int)
        df_eng['Count_Negative_Net_Income_12Q'] = df_eng.groupby('ticker')['is_negative_net_income'].transform(lambda x: x.rolling(window=12, min_periods=12).sum())
    else: df_eng['Count_Negative_Net_Income_12Q'] = np.nan
    if 'Total Revenue_YoY_Growth' in df_eng.columns:
        df_eng['Sales_Volatility'] = df_eng.groupby('ticker')['Total Revenue_YoY_Growth'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
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
    
    # --- 7. Working Capital Turnover (WCT) (Included in previous log) ---
    if 'Net Working Capital' in df_eng.columns and rev_col:
        df_eng['Total Revenue_TTM'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['WCT'] = df_eng['Total Revenue_TTM'] / (df_eng['Net Working Capital'].replace(0, np.nan) + 1e-6)
    else:
        df_eng['WCT'] = np.nan

    # --- 8. NEW: Cash Conversion Cycle (CCC) ---
    # Need 4 quarters of Cost of Revenue (COR_TTM) and Revenue (REV_TTM)
    if cor_col and inv_col:
        df_eng['COR_TTM'] = df_eng.groupby('ticker')[cor_col].transform(calculate_ttm)
        df_eng['Avg_Inventory'] = df_eng.groupby('ticker')[inv_col].transform(lambda x: x.rolling(window=2, min_periods=2).mean())
        df_eng['DIO'] = (df_eng['Avg_Inventory'] / df_eng['COR_TTM'].replace(0, np.nan)) * 365
    else: df_eng['DIO'] = np.nan

    if ar_col and rev_col:
        df_eng['REV_TTM'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['Avg_AR'] = df_eng.groupby('ticker')[ar_col].transform(lambda x: x.rolling(window=2, min_periods=2).mean())
        df_eng['DSO'] = (df_eng['Avg_AR'] / df_eng['REV_TTM'].replace(0, np.nan)) * 365
    else: df_eng['DSO'] = np.nan

    if ap_col and cor_col:
        df_eng['COR_TTM_AP'] = df_eng.groupby('ticker')[cor_col].transform(calculate_ttm)
        df_eng['Avg_AP'] = df_eng.groupby('ticker')[ap_col].transform(lambda x: x.rolling(window=2, min_periods=2).mean())
        df_eng['DPO'] = (df_eng['Avg_AP'] / df_eng['COR_TTM_AP'].replace(0, np.nan)) * 365
    else: df_eng['DPO'] = np.nan
    
    df_eng['CCC'] = df_eng['DSO'] + df_eng['DIO'] - df_eng['DPO']

    # --- 9. NEW V2.0: Market-Based Features (4 Features) ---
    # Market Cap (ln)
    if mc_col:
        df_eng['Market Cap (ln)'] = np.log(df_eng[mc_col].replace(0, np.nan))
    else:
        df_eng['Market Cap (ln)'] = np.nan
        
    # P/B Ratio (Market Cap / Total Equity)
    if mc_col and equity_col:
        df_eng['P/B Ratio'] = df_eng[mc_col] / df_eng[equity_col].replace(0, np.nan)
    else:
        df_eng['P/B Ratio'] = np.nan
        
    # Stock Return (3-Month) - ใช้ closing_price และ pct_change ย้อนหลัง 1 ไตรมาส (3 เดือน/1 period)
    if price_col:
        df_eng['Stock_Return_3M'] = df_eng.groupby('ticker')[price_col].pct_change(periods=1, fill_method=None)
    else:
        df_eng['Stock_Return_3M'] = np.nan
        
    # Return Volatility (Std Dev of Stock Return over 4 Quarters)
    if 'Stock_Return_3M' in df_eng.columns:
        df_eng['Return_Volatility_4Q'] = df_eng.groupby('ticker')['Stock_Return_3M'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
    else:
        df_eng['Return_Volatility_4Q'] = np.nan
        
    # --- END NEW V2.0 Market-Based Features ---

    # --- Final Feature Selection ---
    feature_cols = [
        'Current Ratio', 'Operating Margin', 'Total Assets (ln)', 
        'Total Revenue_YoY_Growth', 'Net Income_YoY_Growth', 'Operating Cash Flow_YoY_Growth', 
        'Change in Operating Margin', 'Change in Current Ratio', 
        'MA_Net Income', 'MA_ROE',
        'Interest Coverage Ratio', 'Quick Ratio', 'Debt to EBITDA', 'NWC_to_Total_Assets', 
        'CV_Operating_Margin', 'Count_Negative_Net_Income_12Q', 'Sales_Volatility',
        'RETA_Proxy', 'EBITTA_Proxy', 'SATA_Proxy',
        'WCT', 
        'CCC' # <-- NEW FEATURE (22 base features)
    ]
    
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

    X = df_eng[feature_cols].copy()
    y = df_eng['Y'].copy()
    identifiers = df_eng[['ticker', 'report_date']].copy()
    
    X.dropna(axis=0, how='all', inplace=True)
    y = y.loc[X.index] 
    identifiers = identifiers.loc[X.index] 
    
    if X.empty:
        logging.error("No valid features remain after dropping all-NaN rows.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []
    
    feature_cols_final = [col for col in feature_cols if col in X.columns]
    X = X[feature_cols_final]
    

    # --- Imputation and Scaling (FIX: Add Indicator) ---
    X_to_impute = X.copy()
    
    all_nan_cols = X_to_impute.columns[X_to_impute.isnull().all()].tolist()
    if all_nan_cols:
        logging.warning(f"Forcing imputation for all-NaN features by filling with 0: {all_nan_cols}")
        X_to_impute[all_nan_cols] = X_to_impute[all_nan_cols].fillna(0)  

    # FIX 1: Implement Median Imputation with Indicator Feature
    imputer = SimpleImputer(strategy='median', add_indicator=True) 
    X_imputed_array = imputer.fit_transform(X_to_impute)
    
    feature_names_out = list(feature_cols_final)
    missing_features_indices = imputer.indicator_.features_
    indicator_feature_names = [f'Missing_{feature_names_out[i]}' for i in missing_features_indices]
    final_feature_names = feature_names_out + indicator_feature_names

    X_imputed_df = pd.DataFrame(X_imputed_array, columns=final_feature_names, index=X_to_impute.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=final_feature_names, index=X_imputed_df.index) 

    logging.info(f"Selected {len(final_feature_names)} features (including indicators) for model training.")
    logging.info(f"Feature names: {list(final_feature_names)}")

    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))
    logging.info(f"Imputer saved to {os.path.join(MODEL_DIR, IMPUTER_FILENAME)}")
    logging.info(f"Scaler saved to {os.path.join(MODEL_DIR, SCALER_FILENAME)}")

    return X_scaled_df, y, identifiers, final_feature_names


def split_data(X, y, identifiers):
    """Splits data into train, validation, and test sets based on time. (Unchanged logic)"""
    logging.info("4. Splitting data into Train, Validation, Test sets (Time-Based)...")
    
    TEST_RATIO = 0.20  
    VAL_RATIO = 0.10  
    MIN_TRAIN_SAMPLES = 50 

    data_full = pd.concat([identifiers.reset_index(drop=True),
                           X.reset_index(drop=True),
                           y.reset_index(drop=True)], axis=1)
    data_full = data_full.sort_values(by='report_date').reset_index(drop=True)

    if data_full.empty:
        raise ValueError("Cannot split empty DataFrame.")

    total_samples = len(data_full)
    
    test_size = int(total_samples * TEST_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    
    if total_samples - test_size - val_size < MIN_TRAIN_SAMPLES or total_samples < 100:
        logging.warning("Time-Based split failed. Falling back to random split.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42, stratify=y)
        X_val, y_val = pd.DataFrame(), pd.Series()
        
        train_indices, test_indices = train_test_split(identifiers.index, test_size=TEST_RATIO, random_state=42, stratify=y.values)
        train_ids = identifiers.loc[train_indices]
        test_ids = identifiers.loc[test_indices]
        val_ids = pd.DataFrame()
        
    else:
        train_end_index = total_samples - test_size - val_size
        val_end_index = total_samples - test_size
        
        train_data = data_full.iloc[:train_end_index]
        val_data = data_full.iloc[train_end_index:val_end_index]
        test_data = data_full.iloc[val_end_index:]
        
        feature_cols = X.columns
        X_train = train_data[feature_cols]
        y_train = train_data['Y']
        train_ids = train_data[['ticker', 'report_date']]

        X_val = val_data[feature_cols]
        y_val = val_data['Y']
        val_ids = val_data[['ticker', 'report_date']]

        X_test = test_data[feature_cols]
        y_test = test_data['Y']
        test_ids = test_data[['ticker', 'report_date']]

    logging.info(f"Data split:")
    logging.info(f"  Train: {len(X_train)} samples (Dates: {train_ids['report_date'].min().date()} - {train_ids['report_date'].max().date()})")
    if not X_val.empty:
        logging.info(f"  Validation: {len(X_val)} samples (Dates: {val_ids['report_date'].min().date()} - {val_ids['report_date'].max().date()})")
    logging.info(f"  Test: {len(X_test)} samples (Dates: {test_ids['report_date'].min().date()} - {test_ids['report_date'].max().date()})")

    logging.info(f"  Train Y distribution:\n{y_train.value_counts(normalize=True)}")
    if not y_val.empty:
        logging.info(f"  Validation Y distribution:\n{y_val.value_counts(normalize=True)}")
    logging.info(f"  Test Y distribution:\n{y_test.value_counts(normalize=True)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train):
    """Trains the XGBoost Classifier using calculated scale_pos_weight (Reduced FN Penalty)"""
    logging.info("4.1 Calculating scale_pos_weight...")
    
    count_neg = y_train.value_counts().get(0, 0)
    count_pos = y_train.value_counts().get(1, 1) 
    scale_pos_weight_value = count_neg / count_pos
    
    logging.info(f"  Calculated scale_pos_weight: {scale_pos_weight_value:.2f} (Count 0 / Count 1)")
    
    X_train_final, y_train_final = X_train, y_train
        
    logging.info("5. Training XGBoost model using RandomizedSearchCV...")

    # ***EDITED: Base scale_pos_weight multiplier reduced to 1.0x (Less aggressive FN penalty)***
    base_scale_pos_weight = scale_pos_weight_value * 1.0 
    
    xgb_base = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        scale_pos_weight=base_scale_pos_weight, 
    )
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.2],
        'gamma': [0, 0.1, 0.5], 
        'subsample': [0.7, 0.8, 0.9],
        'reg_lambda': [0.1, 1, 10],   
        'reg_alpha': [0, 0.1, 1],     
    }
    
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_grid,
        scoring=f2_scorer,
        cv=skf,
        n_iter=20, 
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train_final, y_train_final) 
    
    best_model = random_search.best_estimator_
    
    logging.info("Model training complete. Hyperparameter tuning finished.")
    logging.info(f"Best F2-Score found: {random_search.best_score_:.4f}")
    logging.info(f"Best Hyperparameters: {random_search.best_params_}")
    
    return best_model 

def optimize_threshold(model, X_val, y_val, beta=2.0):
    """
    FIX 3: Calculates the optimal threshold from the Validation set by explicitly maximizing F-beta score (F2-Score)
    with a stricter MIN_PRECISION constraint.
    """
    if X_val.empty or y_val.empty:
        logging.warning("Validation set is empty, skipping threshold optimization.")
        return 0.5, 0.0, 0.0 

    logging.info("Optimizing Classification Threshold using Validation Set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calculate F-beta scores (F2-Score) for all thresholds
    fbeta_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-6)
    
    # Constraint: Only consider thresholds where Precision is at least 0.50 (Our new minimum acceptable level) (***EDITED: Stricter MIN_PRECISION***)
    MIN_PRECISION = 0.30 
    constrained_indices = np.where(precision[:-1] >= MIN_PRECISION)[0]
    
    if len(constrained_indices) > 0:
        # Among those that meet min Precision, maximize F2-Score
        optimal_idx = constrained_indices[np.argmax(fbeta_scores[constrained_indices])]
        
        optimal_threshold = thresholds[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_fbeta = fbeta_scores[optimal_idx]
        
    else:
        # Fallback to pure F2-Score maximization (if no point meets MIN_PRECISION)
        optimal_idx = np.argmax(fbeta_scores[:-1])
        
        optimal_threshold = thresholds[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_fbeta = fbeta_scores[optimal_idx]
        logging.warning(f"  Cannot meet minimum Precision target of {MIN_PRECISION:.2f}. Falling back to pure F{beta}-Score maximization.")


    logging.info(f"  Optimization Goal: Maximize F{beta}-Score (Min Precision {MIN_PRECISION:.2f}).")
    logging.info(f"  Optimal Threshold found: {optimal_threshold:.4f}")
    logging.info(f"  Resulting Precision: {optimal_precision:.4f}")
    logging.info(f"  Resulting Recall: {optimal_recall:.4f}")
    logging.info(f"  Resulting F{beta}-Score: {optimal_fbeta:.4f}")
    
    return optimal_threshold, optimal_recall, optimal_precision


def evaluate_model(model, X_test, y_test, optimal_threshold=0.5):
    """Evaluates the model on the test set and prints metrics. (FIXED: Removed zero_division from accuracy_score)"""
    logging.info("6. Evaluating model on Test Set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] 

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # ***FIX: Remove zero_division=0 from accuracy_score***
    accuracy = accuracy_score(y_test, y_pred) 
    
    # Keep zero_division=0 for other metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
        logging.warning("Cannot calculate AUC: Only one class present in y_test.")

    cm = confusion_matrix(y_test, y_pred)

    logging.info(f"--- Evaluation Metrics (Threshold {optimal_threshold:.4f}) ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"AUC-ROC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "AUC-ROC:   N/A")
    logging.info("--- Confusion Matrix ---")
    logging.info(f"\n{cm}")
    
    logging.info("--- Classification Report ---")
    logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    
    logging.info("--- End Evaluation ---")

def explain_model(model, X_train, feature_names):
    """
    FIX 4: Uses SHAP to explain the model and prints the analysis hint for Missing Indicators. (Unchanged logic)
    """
    logging.info("7. Explaining model using SHAP...")
    
    # Use a small sample of training data for SHAP background/speed
    X_train_sample = X_train.sample(min(200, len(X_train)), random_state=42)

    try:
        # Primary method: Use TreeExplainer (will likely fail, but we try first)
        explainer = shap.TreeExplainer(model, X_train_sample) 
        shap_values = explainer.shap_values(X_train_sample)
        
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_class1 = shap_values[1] 
        else:
            shap_values_class1 = shap_values
        
        logging.info(f"SHAP explanation calculated for {len(feature_names)} features using TreeExplainer.")
        
        # --- SHAP Analysis Hint ---
        if isinstance(shap_values_class1, np.ndarray):
            X_train_sample_df = pd.DataFrame(X_train_sample, columns=feature_names, index=X_train_sample.index)
            missing_feature_cols = [col for col in X_train_sample_df.columns if col.startswith('Missing_')]
            
            if missing_feature_cols:
                missing_indices = [X_train_sample_df.columns.get_loc(col) for col in missing_feature_cols]

                mean_abs_shap_missing = np.mean(np.abs(shap_values_class1[:, missing_indices]))
                mean_abs_shap_all = np.mean(np.abs(shap_values_class1))
                
                logging.info(f"SHAP Analysis Hint: Mean Absolute SHAP for Missing Indicators: {mean_abs_shap_missing:.4f}")
                logging.info(f"SHAP Analysis Hint: Mean Absolute SHAP for All Features: {mean_abs_shap_all:.4f}")
                if mean_abs_shap_all > 1e-6:
                    logging.info(f"SHAP Analysis Hint: Missing Indicators are {(mean_abs_shap_missing/mean_abs_shap_all):.1%} of the total feature importance.")
                else:
                    logging.info("SHAP Analysis Hint: Total importance too small to calculate ratio.")
        # --- End SHAP Analysis Hint ---

    except Exception as e:
        # Secondary method: Fallback to KernelExplainer (will capture the hint via the outer logic on next run)
        logging.error(f"Error during TreeExplainer: {e}. Falling back to KernelExplainer with predict_proba wrapper...", exc_info=True)
        try:
             explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], X_train_sample)
             shap_values_kernel = explainer(X_train_sample)
             
             logging.info("SHAP explanation calculated successfully using secondary (KernelExplainer) method.")
             
             # Re-try printing the hint using KernelExplainer results
             if hasattr(shap_values_kernel, 'values') and isinstance(shap_values_kernel.values, np.ndarray):
                shap_values_class1 = shap_values_kernel.values
                X_train_sample_df = pd.DataFrame(X_train_sample, columns=feature_names, index=X_train_sample.index)
                missing_feature_cols = [col for col in X_train_sample_df.columns if col.startswith('Missing_')]
            
                if missing_feature_cols:
                    missing_indices = [X_train_sample_df.columns.get_loc(col) for col in missing_feature_cols]

                    mean_abs_shap_missing = np.mean(np.abs(shap_values_class1[:, missing_indices]))
                    mean_abs_shap_all = np.mean(np.abs(shap_values_class1))
                    
                    logging.info(f"SHAP Analysis Hint (Fallback): Mean Absolute SHAP for Missing Indicators: {mean_abs_shap_missing:.4f}")
                    logging.info(f"SHAP Analysis Hint (Fallback): Mean Absolute SHAP for All Features: {mean_abs_shap_all:.4f}")
                    if mean_abs_shap_all > 1e-6:
                        logging.info(f"SHAP Analysis Hint (Fallback): Missing Indicators are {(mean_abs_shap_missing/mean_abs_shap_all):.1%} of the total feature importance.")
                    else:
                        logging.info("SHAP Analysis Hint (Fallback): Total importance too small to calculate ratio.")


        except Exception as e2:
            logging.error(f"Error during SHAP explanation (All attempts failed): {e2}", exc_info=True)
            return


def save_model(model):
    """Saves the trained model to a file using joblib. (Unchanged logic)"""
    logging.info("8. Saving the trained model...")
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model successfully saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("===== Starting ML Risk Prediction Script (XGBoost + Tuned + Balanced FN Penalty + Max F2 Threshold + Stricter Y & Precision) =====")
    start_time = datetime.now()

    # 1. Fetch Data
    logging.info("1. Fetching comprehensive ML raw data from database starting from year 2010...")
    
    # ❌ ลบ/คอมเมนต์ บรรทัดเก่า: df_financials, df_summary_de = fetch_data(start_year=2010)
    # ✅ เรียกใช้ฟังก์ชันใหม่
    df_raw_data = get_ml_risk_raw_data(start_year=2010) 

    if df_raw_data.empty:
        logging.error("Stopping script because no financial data could be fetched.")
    else:
        # 2. Create Target Variable (Uses Stricter Constants)
        # ปรับการเรียกใช้: ใช้ df_raw_data.reset_index() และส่ง DataFrame ว่างไปแทน df_summary_de
        df_financials_reset = df_raw_data.reset_index()
        df_with_y = create_target_variable(df_financials_reset, pd.DataFrame()) 

        if not df_with_y.empty:
            # 3. Engineer Features (includes CCC and fixed indicator feature)
            X, y, ids, feature_names_final = engineer_features(df_with_y)

            if not X.empty:
                # 4. Split Data (Time-Based)
                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, ids)

                    # 5. Train Model (uses calculated scale_pos_weight * 1.0)
                    trained_model = train_model(X_train, y_train)

                    # **FIXED STEP: Optimize Threshold for F2-Score (Maximized Recall) with MIN_PRECISION 0.30**
                    if not X_val.empty:
                        # Maximize F2-Score (Beta=2.0) with Min Precision 0.50 constraint
                        optimized_threshold, _, _ = optimize_threshold(trained_model, X_val, y_val, beta=2.0)
                    else:
                        optimized_threshold = 0.5 
                        logging.warning("No Validation set available. Using default threshold 0.5 for evaluation.")

                    # 6. Evaluate Model (on Test set) using optimized threshold
                    evaluate_model(trained_model, X_test, y_test, optimized_threshold)

                    # 7. Explain Model (with SHAP analysis hint)
                    explain_model(trained_model, X_train, feature_names_final)

                    # 8. Save Model
                    save_model(trained_model)
                    
                    logging.info("Placeholder: Prediction for latest data would happen here.")

                except ValueError as ve:
                    logging.error(f"Error during data splitting or processing: {ve}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred during ML pipeline: {e}", exc_info=True)
            else:
                logging.error("Stopping script because feature engineering resulted in empty data.")
        else:
            logging.error("Stopping script because target variable creation failed or resulted in empty data.")

    end_time = datetime.now()
    logging.info(f"===== ML Risk Prediction Script Finished in {end_time - start_time} =====")