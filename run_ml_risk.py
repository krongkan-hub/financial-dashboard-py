# run_ml_risk.py (เวอร์ชัน FINAL-FIX: Fixed NameError, scale_pos_weight, Time Split, Z-Score, SHAP base_score error)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import joblib 
import shap 
# from imblearn.over_sampling import SMOTE # นำออก, ใช้ scale_pos_weight แทน
from xgboost import XGBClassifier 

# --- Database Interaction ---
from app import db, server, FactFinancialStatements, FactCompanySummary
from sqlalchemy import func, and_
from sqlalchemy.orm import aliased

# --- Machine Learning ---
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for Target Variable Definition
DE_RATIO_THRESHOLD = 5.0
ICR_THRESHOLD = 1.0
CONSECUTIVE_LOSS_PERIODS = 6 

# Feature Engineering Constants
MA_PERIODS = 4 

# Model Saving Configuration
MODEL_DIR = "models"
MODEL_FILENAME = "trained_risk_model_xgb.joblib" 
SCALER_FILENAME = "scaler.joblib" 
IMPUTER_FILENAME = "imputer.joblib" 

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper Functions ---
def calculate_ttm(series):
    """Calculates Trailing Twelve Months sum for a pandas Series grouped by ticker."""
    return series.rolling(window=4, min_periods=4).sum()

def calculate_change(series, periods):
    """Calculates the change over a number of periods."""
    return series.diff(periods=periods)

def get_column(df, possible_names):
    """NEW HELPER: Finds the first existing column from a list of possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None 

# --- Core Functions ---

def fetch_data(start_year=2015):
    """
    Fetches required financial data from the database.
    """
    logging.info(f"1. Fetching data from database starting from year {start_year}...")
    start_date = date(start_year, 1, 1)

    # UPDATED: รวมชื่อ Metric ที่เป็นไปได้ทั้งหมดจากการสังเกตข้อมูล YFinance/ETL
    required_metrics = [
        # Base/Target Metrics (High Ambiguity)
        'Total Stockholders Equity', 'Stockholders Equity', 'Total Assets',
        'Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest',
        'EBIT', 'Operating Income', 
        
        # Base/Target Metrics (Low Ambiguity)
        'Interest Expense', 'Net Income', 'Total Revenue', 'Operating Cash Flow',
        'Current Assets', 'Current Liabilities', 'Inventory', 
    ]

    try:
        with server.app_context():
            stmt_ffs = db.session.query(
                FactFinancialStatements.ticker,
                FactFinancialStatements.report_date,
                FactFinancialStatements.metric_name,
                FactFinancialStatements.metric_value
            ).filter(
                FactFinancialStatements.report_date >= start_date,
                FactFinancialStatements.metric_name.in_(required_metrics)
            ).statement

            df_ffs_long = pd.read_sql(stmt_ffs, db.engine)
            df_ffs_long['report_date'] = pd.to_datetime(df_ffs_long['report_date'])

            if df_ffs_long.empty:
                logging.error("No data found in FactFinancialStatements for the specified period.")
                return pd.DataFrame(), pd.DataFrame()

            # Pivot to wide format
            df_ffs_wide = df_ffs_long.pivot_table(
                index=['ticker', 'report_date'],
                columns='metric_name',
                values='metric_value'
            ).reset_index()
            df_ffs_wide = df_ffs_wide.sort_values(by=['ticker', 'report_date'])

            # Query FactCompanySummary for D/E Ratio (using original logic)
            FCS = aliased(FactCompanySummary)
            
            # ... (unchanged logic for querying FactCompanySummary) ...
            quarter_group = (func.strftime('%Y', FCS.date_updated) + '-' +
                             ((func.strftime('%m', FCS.date_updated).cast(db.Integer) + 2) / 3).cast(db.String)
                             ).label('year_quarter')

            subquery = db.session.query(
                FCS.ticker,
                quarter_group, 
                func.max(FCS.date_updated).label('max_summary_date')
            ).group_by(FCS.ticker, quarter_group).subquery() 

            stmt_fcs = db.session.query(
                FactCompanySummary.ticker,
                FactCompanySummary.date_updated.label('summary_date'),
                (func.strftime('%Y', FactCompanySummary.date_updated) + '-' +
                 ((func.strftime('%m', FactCompanySummary.date_updated).cast(db.Integer) + 2) / 3).cast(db.String)
                 ).label('year_quarter'), 
                FactCompanySummary.de_ratio
            ).join(
                subquery,
                and_(
                    FactCompanySummary.ticker == subquery.c.ticker,
                    FactCompanySummary.date_updated == subquery.c.max_summary_date 
                )
            ).filter(FactCompanySummary.date_updated >= start_date).statement 
            
            df_fcs = pd.read_sql(stmt_fcs, db.engine) 
            
            # Approximate date calculation (original logic)
            df_fcs[['Year', 'Quarter']] = df_fcs['year_quarter'].str.split('-', expand=True)
            df_fcs['Quarter_Start_Month'] = (df_fcs['Quarter'].astype(float).astype(int) - 1) * 3 + 1
            df_fcs['Quarter_Start_Date'] = pd.to_datetime(
                df_fcs['Year'].astype(str) + '-' + df_fcs['Quarter_Start_Month'].astype(str) + '-01',
                errors='coerce'
            )
            df_fcs['report_date_approx'] = df_fcs['Quarter_Start_Date'] + pd.offsets.QuarterEnd(0)
            df_fcs = df_fcs.drop(columns=['Year', 'Quarter', 'Quarter_Start_Month', 'Quarter_Start_Date', 'year_quarter', 'summary_date'])
            df_fcs = df_fcs.sort_values(by=['ticker', 'report_date_approx'])
            
            logging.info(f"Fetched {len(df_ffs_wide)} wide financial statement records.")
            logging.info(f"Fetched {len(df_fcs)} D/E ratio records.")
            return df_ffs_wide, df_fcs

    except Exception as e:
        logging.error(f"Error fetching data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()


def create_target_variable(df_ffs, df_fcs):
    """
    Creates the binary target variable (Y) based on defined risk criteria.
    """
    logging.info("2. Creating Target Variable (Y)...")
    if df_ffs.empty:
        logging.warning("Financial statement data is empty, cannot create target variable.")
        return pd.DataFrame()

    df = df_ffs.copy()
    df = df.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)

    # --- Use Robust Column Selection for Ambiguous Metrics ---
    
    equity_col = get_column(df, ['Stockholders Equity', 'Total Stockholders Equity'])
    ebit_col = get_column(df, ['EBIT', 'Operating Income'])
    interest_col = get_column(df, ['Interest Expense'])
    net_income_col = get_column(df, ['Net Income', 'Net Income Common Stockholders'])

    # --- Calculate necessary components for Y ---
    
    # TTM ICR
    if ebit_col and interest_col:
        df['EBIT_TTM'] = df.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df['Interest Expense_TTM'] = df.groupby('ticker')[interest_col].transform(lambda x: x.rolling(window=4, min_periods=4).sum().replace(0, np.nan))
        df['TTM_ICR'] = df['EBIT_TTM'] / df['Interest Expense_TTM'].replace(0, np.nan)
        df['TTM_ICR'] = df['TTM_ICR'].replace([np.inf, -np.inf], np.nan)
    else:
        df['TTM_ICR'] = np.nan
        
    # Consecutive Net Losses
    if net_income_col:
        df['is_loss'] = (df.get(net_income_col, pd.Series()) < 0).astype(int)
        df['consecutive_losses'] = df.groupby('ticker')['is_loss'].transform(
            lambda x: x.rolling(window=CONSECUTIVE_LOSS_PERIODS, min_periods=CONSECUTIVE_LOSS_PERIODS).sum()
        )
    else:
        df['consecutive_losses'] = np.nan

    # Negative Equity
    if equity_col:
        df['is_negative_equity'] = (df.get(equity_col, pd.Series()) < 0).astype(int)
    else:
        df['is_negative_equity'] = 0 

    # Merge D/E Ratio (using original merge logic)
    df['report_date_approx'] = df['report_date'] + pd.offsets.QuarterEnd(0)
    df = pd.merge_asof(df.sort_values('report_date_approx'),
                       df_fcs[['ticker', 'report_date_approx', 'de_ratio']].sort_values('report_date_approx'),
                       on='report_date_approx',
                       by='ticker',
                       direction='backward', 
                       tolerance=pd.Timedelta('365 days')) 

    # --- Apply Risk Rules ---
    rule1 = df['is_negative_equity'] == 1
    rule2 = (df['de_ratio'] > DE_RATIO_THRESHOLD) & (df['TTM_ICR'] < ICR_THRESHOLD)
    rule3 = df['consecutive_losses'] >= CONSECUTIVE_LOSS_PERIODS

    df['Y_raw'] = ((rule1) | (rule2) | (rule3)).astype(int)

    # --- Introduce Time Lag (4 quarters) ---
    lag_periods = 4
    df['Y'] = df.groupby('ticker')['Y_raw'].shift(-lag_periods)
    logging.info(f"Target variable Y created with a lag of {lag_periods} quarters.")

    df_final = df.dropna(subset=['Y']).copy()
    df_final['Y'] = df_final['Y'].astype(int)

    logging.info(f"Created Y for {len(df_final)} data points.")
    logging.info(f"Distribution of Y: \n{df_final['Y'].value_counts(normalize=True)}")

    # --- Robust column selection: Keep all financial metrics and calculated Y/de_ratio ---
    explicit_cols = ['ticker', 'report_date', 'Y', 'de_ratio']
    cols_to_keep_from_original = [col for col in df_ffs.columns if col not in ['ticker', 'report_date']]
    
    cols_to_keep = explicit_cols + cols_to_keep_from_original
    cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]

    df_final = df_final[cols_to_keep]

    return df_final


def engineer_features(df):
    """
    Calculates 22 features (including 3 Z-Score Proxies), handles infinities/NaNs, and applies scaling/imputation.
    """
    logging.info("3. Engineering Features (X)...")
    if df.empty or 'Y' not in df.columns:
        logging.warning("Input DataFrame is empty or missing Y column.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []

    df_eng = df.copy()
    df_eng = df_eng.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)
    periods_per_year = 4
    
    # --- Use Robust Column Selection for all raw metrics ---
    
    # Financial/Balance Sheet Components
    ca_col = get_column(df_eng, ['Current Assets'])
    cl_col = get_column(df_eng, ['Current Liabilities'])
    ta_col = get_column(df_eng, ['Total Assets'])
    inv_col = get_column(df_eng, ['Inventory'])
    liab_col = get_column(df_eng, ['Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest'])
    equity_col = get_column(df_eng, ['Stockholders Equity', 'Total Stockholders Equity'])
    
    # Income/Cash Flow Components
    ebit_col = get_column(df_eng, ['EBIT', 'Operating Income'])
    rev_col = get_column(df_eng, ['Total Revenue'])
    ni_col = get_column(df_eng, ['Net Income', 'Net Income Common Stockholders'])
    ocf_col = get_column(df_eng, ['Operating Cash Flow', 'Total Cash Flow From Operating Activities'])
    int_exp_col = get_column(df_eng, ['Interest Expense']) 
    
    # --- 1. Base Ratios / Size Features ---
    if ca_col and cl_col:
        df_eng['Current Ratio'] = df_eng[ca_col] / df_eng[cl_col].replace(0, np.nan)
    else: df_eng['Current Ratio'] = np.nan

    if ta_col:
        df_eng['Total Assets (ln)'] = np.log(df_eng[ta_col].replace(0, np.nan))
    else: df_eng['Total Assets (ln)'] = np.nan

    if ebit_col and rev_col:
        df_eng['Operating Margin'] = df_eng[ebit_col] / df_eng[rev_col].replace(0, np.nan)
    else: df_eng['Operating Margin'] = np.nan

    # Calculate ROE for MA
    if ni_col and equity_col:
        df_eng['Net Income_TTM'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['Avg Equity_TTM'] = df_eng.groupby('ticker')[equity_col].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['ROE'] = df_eng['Net Income_TTM'] / df_eng['Avg Equity_TTM'].replace(0, np.nan)
    else: df_eng['ROE'] = np.nan


    # --- 2. YoY Growth Features ---
    growth_col_map = {'Total Revenue': rev_col, 'Net Income': ni_col, 'Operating Cash Flow': ocf_col}
    for display_name, raw_col in growth_col_map.items():
        if raw_col:
            # FIX: Added fill_method=None to remove FutureWarning
            df_eng[f'{display_name}_YoY_Growth'] = df_eng.groupby('ticker')[raw_col].pct_change(periods=periods_per_year, fill_method=None)
        else:
             df_eng[f'{display_name}_YoY_Growth'] = np.nan

    # --- 3. Change in Key Ratios (Over 1 Year) ---
    change_cols_map = {'de_ratio': 'D/E Ratio', 'Operating Margin': 'Operating Margin', 'Current Ratio': 'Current Ratio'}    
    for col_raw, col_display in change_cols_map.items():
        if col_raw in df_eng.columns: 
            df_eng[f'Change in {col_display}'] = df_eng.groupby('ticker')[col_raw].transform(calculate_change, periods=periods_per_year)
        else:
            df_eng[f'Change in {col_display}'] = np.nan

    # --- 4. Moving Averages (4 Quarters) ---
    ma_cols = ['Net Income', 'ROE']    
    for col in ma_cols:
        if col in df_eng.columns: 
            df_eng[f'MA_{col}'] = df_eng.groupby('ticker')[col].transform(lambda x: x.rolling(window=MA_PERIODS, min_periods=MA_PERIODS).mean())
        else:
             df_eng[f'MA_{col}'] = np.nan

    # ------------------------------------------------------------------
    # --- 5. Advanced Credit Risk Features (7 Features) ---
    # ------------------------------------------------------------------
    
    # 1. Interest Coverage Ratio (EBIT / Interest Expense)
    if ebit_col and int_exp_col:
        df_eng['Interest Coverage Ratio'] = df_eng[ebit_col] / df_eng[int_exp_col].replace(0, np.nan)
    else: df_eng['Interest Coverage Ratio'] = np.nan

    # 2. Quick Ratio (Acid-Test)
    if ca_col and inv_col and cl_col:
        df_eng['Quick Ratio'] = (df_eng[ca_col] - df_eng[inv_col]) / df_eng[cl_col].replace(0, np.nan)
    else: df_eng['Quick Ratio'] = np.nan

    # 3. Debt to EBITDA (TTM)
    if liab_col and ebit_col:
        df_eng['Total Liabilities_TTM'] = df_eng.groupby('ticker')[liab_col].transform(calculate_ttm)
        df_eng['EBIT_TTM_4Q'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm) 
        df_eng['Debt to EBITDA'] = df_eng['Total Liabilities_TTM'] / df_eng['EBIT_TTM_4Q'].replace(0, np.nan)
    else: 
        df_eng['Debt to EBITDA'] = np.nan

    # 4. Net Working Capital / Total Assets
    if ca_col and cl_col and ta_col:
        df_eng['Net Working Capital'] = df_eng[ca_col] - df_eng[cl_col]
        df_eng['NWC_to_Total_Assets'] = df_eng['Net Working Capital'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['NWC_to_Total_Assets'] = np.nan

    # 5. Coefficient of Variation (CV) of Operating Margin (4Q)
    if 'Operating Margin' in df_eng.columns:
        df_eng['Op_Margin_Mean_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
        df_eng['Op_Margin_Std_4Q'] = df_eng.groupby('ticker')['Operating Margin'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
        df_eng['CV_Operating_Margin'] = df_eng['Op_Margin_Std_4Q'] / df_eng['Op_Margin_Mean_4Q'].replace(0, np.nan)
    else: df_eng['CV_Operating_Margin'] = np.nan

    # 6. Number of Negative Net Income Quarters (12Q)
    if ni_col:
        df_eng['is_negative_net_income'] = (df_eng[ni_col] < 0).astype(int)
        df_eng['Count_Negative_Net_Income_12Q'] = df_eng.groupby('ticker')['is_negative_net_income'].transform(lambda x: x.rolling(window=12, min_periods=12).sum())
    else: df_eng['Count_Negative_Net_Income_12Q'] = np.nan

    # 7. Sales Volatility (Standard Deviation of YoY Revenue Growth - 4Q)
    if 'Total Revenue_YoY_Growth' in df_eng.columns:
        df_eng['Sales_Volatility'] = df_eng.groupby('ticker')['Total Revenue_YoY_Growth'].transform(lambda x: x.rolling(window=4, min_periods=4).std())
    else: df_eng['Sales_Volatility'] = np.nan
    
    # ------------------------------------------------------------------
    # --- 6. NEW: Altman Z-Score Components Proxies (3 Features) ---
    # ------------------------------------------------------------------
    
    # 1. Retained Earnings / Total Assets (RETA) Proxy: Cumulative TTM Net Income / Total Assets
    if ni_col and ta_col:
        # TTM Net Income (proxy for incremental retained earnings, often used when historical RE isn't easily available)
        df_eng['Net Income_TTM_RETA'] = df_eng.groupby('ticker')[ni_col].transform(calculate_ttm)
        df_eng['RETA_Proxy'] = df_eng['Net Income_TTM_RETA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['RETA_Proxy'] = np.nan
    
    # 2. EBIT / Total Assets (EBITTA) Proxy: TTM EBIT / Total Assets
    if ebit_col and ta_col:
        df_eng['EBIT_TTM_EBITTA'] = df_eng.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df_eng['EBITTA_Proxy'] = df_eng['EBIT_TTM_EBITTA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['EBITTA_Proxy'] = np.nan
        
    # 3. Sales / Total Assets (SATA) Proxy: TTM Revenue / Total Assets
    if rev_col and ta_col:
        df_eng['Total Revenue_TTM_SATA'] = df_eng.groupby('ticker')[rev_col].transform(calculate_ttm)
        df_eng['SATA_Proxy'] = df_eng['Total Revenue_TTM_SATA'] / df_eng[ta_col].replace(0, np.nan)
    else: df_eng['SATA_Proxy'] = np.nan

    # --- Final Feature Selection (Now 22 Features) ---
    feature_cols = [
        'de_ratio', 'Current Ratio', 'Operating Margin', 'Total Assets (ln)', 
        'Total Revenue_YoY_Growth', 'Net Income_YoY_Growth', 'Operating Cash Flow_YoY_Growth', 
        'Change in D/E Ratio', 'Change in Operating Margin', 'Change in Current Ratio', 
        'MA_Net Income', 'MA_ROE',
        # 7 Advanced Credit Risk Features
        'Interest Coverage Ratio', 'Quick Ratio', 'Debt to EBITDA', 'NWC_to_Total_Assets', 
        'CV_Operating_Margin', 'Count_Negative_Net_Income_12Q', 'Sales_Volatility',
        # 3 New Z-Score Proxies
        'RETA_Proxy', 'EBITTA_Proxy', 'SATA_Proxy' 
    ]
    
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

    # --- CRITICAL FIX: Ensure X and y rows are aligned after feature engineering ---
    X = df_eng[feature_cols].copy()
    y = df_eng['Y'].copy()
    identifiers = df_eng[['ticker', 'report_date']].copy()
    
    # Filter rows where ALL selected features are NaN (highly unlikely if Y exists, but safe)
    X.dropna(axis=0, how='all', inplace=True)
    y = y.loc[X.index] # Ensure y matches filtered X
    identifiers = identifiers.loc[X.index] # Ensure identifiers match filtered X
    
    if X.empty:
        logging.error("No valid features remain after dropping all-NaN rows.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []
    
    # Ensure only existing columns are selected
    feature_cols_final = [col for col in feature_cols if col in X.columns]
    X = X[feature_cols_final]
    # --------------------------------------------------------------------------------------------------

    # --- Imputation and Scaling ---
    X_to_impute = X.copy()
    
    all_nan_cols = X_to_impute.columns[X_to_impute.isnull().all()].tolist()
    if all_nan_cols:
        logging.warning(f"Forcing imputation for all-NaN features by filling with 0: {all_nan_cols}")
        X_to_impute[all_nan_cols] = X_to_impute[all_nan_cols].fillna(0)  

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_to_impute)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_cols_final, index=X_to_impute.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)
    # FIX: Corrected NameError here
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols_final, index=X_imputed_df.index) 

    logging.info(f"Selected {len(feature_cols_final)} features for model training.")
    logging.info(f"Feature names: {list(feature_cols_final)}")

    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))
    logging.info(f"Imputer saved to {os.path.join(MODEL_DIR, IMPUTER_FILENAME)}")
    logging.info(f"Scaler saved to {os.path.join(MODEL_DIR, SCALER_FILENAME)}")

    return X_scaled_df, y, identifiers, feature_cols_final


def split_data(X, y, identifiers):
    """
    Splits data into train, validation, and test sets based on time (Time-Based Split).
    Uses index-based split after sorting to ensure non-overlapping, time-ordered partitions.
    """
    logging.info("4. Splitting data into Train, Validation, Test sets (Time-Based)...")
    
    # --- FIXED TIME SPLIT CONFIGURATION ---
    TEST_RATIO = 0.20  
    VAL_RATIO = 0.20
    MIN_TRAIN_SAMPLES = 50 
    # ------------------------------------------

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
        logging.warning("Time-Based split failed (too few samples, size < 50, or empty). Falling back to random split.")
        
        # Fallback to random split (80/20) with stratification
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42, stratify=y)
        X_val, y_val = pd.DataFrame(), pd.Series()
        
        # Get the corresponding IDs for logging
        train_indices, test_indices = train_test_split(identifiers.index, test_size=TEST_RATIO, random_state=42, stratify=y.values)
        train_ids = identifiers.loc[train_indices]
        test_ids = identifiers.loc[test_indices]
        val_ids = pd.DataFrame()
        
    else:
        # Standard 60/20/20 time-based split (index-based after sorting by report_date)
        train_end_index = total_samples - test_size - val_size
        val_end_index = total_samples - test_size
        
        train_data = data_full.iloc[:train_end_index]
        val_data = data_full.iloc[train_end_index:val_end_index]
        test_data = data_full.iloc[val_end_index:]
        
        # Assign data
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
    """
    Trains the XGBoost Classifier model using scale_pos_weight for class imbalance.
    """
    logging.info("4.1 Calculating scale_pos_weight and skipping SMOTE...")
    
    # Calculate scale_pos_weight
    count_neg = y_train.value_counts().get(0, 0)
    count_pos = y_train.value_counts().get(1, 1) # Set minimum to 1 to avoid ZeroDivisionError
    scale_pos_weight_value = count_neg / count_pos
    
    logging.info(f"  Calculated scale_pos_weight: {scale_pos_weight_value:.2f} (Count 0 / Count 1)")
    
    # Use original training data (no SMOTE)
    X_train_res, y_train_res = X_train, y_train
        
    logging.info("5. Training XGBoost model...")
    
    # Use calculated scale_pos_weight
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight_value, # CRITICAL CHANGE: Use scale_pos_weight instead of SMOTE
        # base_score=0.5 # <--- CRITICAL FIX: บรรทัดนี้ถูกนำออกเพื่อแก้ปัญหา SHAP Error
    )

    model.fit(X_train_res, y_train_res)
    logging.info("Model training complete. scale_pos_weight was used.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints metrics.
    """
    logging.info("6. Evaluating model on Test Set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] 

    # --- Evaluation at Threshold 0.5 (Default) ---
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Check if there are positive samples in y_test for AUC calculation
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = np.nan
        logging.warning("Cannot calculate AUC: Only one class present in y_test.")

    cm = confusion_matrix(y_test, y_pred)

    logging.info("--- Evaluation Metrics (Threshold 0.5) ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"AUC-ROC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "AUC-ROC:   N/A")
    logging.info("--- Confusion Matrix ---")
    logging.info(f"\n{cm}")
    
    logging.info("--- Classification Report (Threshold 0.5) ---")
    logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # --- Evaluation at a Lower Threshold (0.15 for higher Recall) ---
    low_threshold = 0.15 
    y_pred_low = (y_pred_proba >= low_threshold).astype(int)
    
    recall_low = recall_score(y_test, y_pred_low, zero_division=0)
    precision_low = precision_score(y_test, y_pred_low, zero_division=0)
    f1_low = f1_score(y_test, y_pred_low, zero_division=0)
        
    cm_low = confusion_matrix(y_test, y_pred_low)
    
    logging.info(f"--- Evaluation Metrics (Threshold {low_threshold}) ---")
    logging.info(f"Recall:    {recall_low:.4f}")
    logging.info(f"Precision: {precision_low:.4f}")
    logging.info(f"F1-Score:  {f1_low:.4f}")
    logging.info(f"Confusion Matrix:\n{cm_low}")
    
    logging.info("--- End Evaluation ---")


def explain_model(model, X_train, feature_names):
    """
    Uses SHAP to explain the model.
    """
    logging.info("7. Explaining model using SHAP...")
    try:
        explainer = shap.TreeExplainer(model) 
        shap_values = explainer.shap_values(X_train)
        
        # If explainer.shap_values returns a list of two arrays
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_class1 = shap_values[1] # Use values for class 1 (risk=1)
        else:
            shap_values_class1 = shap_values
        
        logging.info("Generating SHAP Summary Plot...")
        logging.info(f"SHAP explanation calculated for {len(feature_names)} features.")

    except Exception as e:
        logging.error(f"Error during SHAP explanation: {e}", exc_info=True)


def save_model(model):
    """
    Saves the trained model to a file using joblib.
    """
    logging.info("8. Saving the trained model...")
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model successfully saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("===== Starting ML Risk Prediction Script (XGBoost) =====")
    start_time = datetime.now()

    # 1. Fetch Data
    df_financials, df_summary_de = fetch_data(start_year=2010) 

    if df_financials.empty:
        logging.error("Stopping script because no financial data could be fetched.")
    else:
        # 2. Create Target Variable
        df_with_y = create_target_variable(df_financials, df_summary_de)

        if not df_with_y.empty:
            # 3. Engineer Features
            X, y, ids, feature_names_final = engineer_features(df_with_y)

            if not X.empty:
                # 4. Split Data (Time-Based)
                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, ids)

                    # 5. Train Model (XGBoost + scale_pos_weight)
                    trained_model = train_model(X_train, y_train)

                    # 6. Evaluate Model (on Test set)
                    evaluate_model(trained_model, X_test, y_test)

                    # 7. Explain Model (using Training set for background)
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