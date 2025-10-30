# run_ml_risk.py (เวอร์ชัน FINAL-FIX V4: Recall Boost & SHAP Fix)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import joblib 
import shap 
from xgboost import XGBClassifier 

# --- Database Interaction ---
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

def fetch_data(start_year=2015):
    """Fetches required financial data from the database. (Unchanged logic)"""
    logging.info(f"1. Fetching data from database starting from year {start_year}...")
    start_date = date(start_year, 1, 1)

    required_metrics = [
        'Total Stockholders Equity', 'Stockholders Equity', 'Total Assets',
        'Total Liabilities', 'Total Liab', 'Total Liabilities Net Minority Interest',
        'EBIT', 'Operating Income', 
        'Interest Expense', 'Net Income', 'Total Revenue', 'Operating Cash Flow',
        'Current Assets', 'Current Liabilities', 'Inventory', 
        # New Metrics for CCC
        'Accounts Receivable', 'Accounts Payable', 'Cost Of Revenue', 'Sales',
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

            df_ffs_wide = df_ffs_long.pivot_table(
                index=['ticker', 'report_date'],
                columns='metric_name',
                values='metric_value'
            ).reset_index()
            df_ffs_wide = df_ffs_wide.sort_values(by=['ticker', 'report_date'])

            FCS = aliased(FactCompanySummary)
            
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
    """Creates the binary target variable (Y) based on defined risk criteria. (Unchanged logic)"""
    logging.info("2. Creating Target Variable (Y)...")
    if df_ffs.empty:
        logging.warning("Financial statement data is empty, cannot create target variable.")
        return pd.DataFrame()

    df = df_ffs.copy()
    df = df.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)

    equity_col = get_column(df, ['Stockholders Equity', 'Total Stockholders Equity'])
    ebit_col = get_column(df, ['EBIT', 'Operating Income'])
    interest_col = get_column(df, ['Interest Expense'])
    net_income_col = get_column(df, ['Net Income', 'Net Income Common Stockholders'])

    # --- Calculate necessary components for Y ---
    if ebit_col and interest_col:
        df['EBIT_TTM'] = df.groupby('ticker')[ebit_col].transform(calculate_ttm)
        df['Interest Expense_TTM'] = df.groupby('ticker')[interest_col].transform(lambda x: x.rolling(window=4, min_periods=4).sum().replace(0, np.nan))
        df['TTM_ICR'] = df['EBIT_TTM'] / df['Interest Expense_TTM'].replace(0, np.nan)
        df['TTM_ICR'] = df['TTM_ICR'].replace([np.inf, -np.inf], np.nan)
    else:
        df['TTM_ICR'] = np.nan
        
    if net_income_col:
        df['is_loss'] = (df.get(net_income_col, pd.Series()) < 0).astype(int)
        df['consecutive_losses'] = df.groupby('ticker')['is_loss'].transform(
            lambda x: x.rolling(window=CONSECUTIVE_LOSS_PERIODS, min_periods=CONSECUTIVE_LOSS_PERIODS).sum()
        )
    else:
        df['consecutive_losses'] = np.nan

    if equity_col:
        df['is_negative_equity'] = (df.get(equity_col, pd.Series()) < 0).astype(int)
    else:
        df['is_negative_equity'] = 0 

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

    explicit_cols = ['ticker', 'report_date', 'Y', 'de_ratio']
    cols_to_keep_from_original = [col for col in df_ffs.columns if col not in ['ticker', 'report_date']]
    
    cols_to_keep = explicit_cols + cols_to_keep_from_original
    cols_to_keep = [col for col in cols_to_keep if col in df_final.columns]

    df_final = df_final[cols_to_keep]

    return df_final


def engineer_features(df):
    """
    Calculates 20+ features, handles infinities/NaNs, and applies scaling/imputation 
    WITH INDICATOR FEATURE FIX.
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
    """Trains the XGBoost Classifier using calculated scale_pos_weight (No SMOTE). (Unchanged logic)"""
    logging.info("4.1 Calculating scale_pos_weight...")
    
    count_neg = y_train.value_counts().get(0, 0)
    count_pos = y_train.value_counts().get(1, 1) 
    scale_pos_weight_value = count_neg / count_pos
    
    logging.info(f"  Calculated scale_pos_weight: {scale_pos_weight_value:.2f} (Count 0 / Count 1)")
    
    X_train_final, y_train_final = X_train, y_train
        
    logging.info("5. Training XGBoost model using RandomizedSearchCV...")

    # INCREASE BASE scale_pos_weight by 1.5x for stronger False Negative penalty
    base_scale_pos_weight = scale_pos_weight_value * 1.5
    
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

def optimize_threshold(model, X_val, y_val, beta=2.0): # FIX 3: Changed default beta back to 2.0
    """
    FIX 3: Calculates the optimal threshold from the Validation set by explicitly maximizing F-beta score (F2-Score).
    """
    if X_val.empty or y_val.empty:
        logging.warning("Validation set is empty, skipping threshold optimization.")
        return 0.5, 0.0, 0.0 

    logging.info("Optimizing Classification Threshold using Validation Set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calculate F-beta scores (F2-Score) for all thresholds
    fbeta_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-6)
    
    # Constraint: Only consider thresholds where Precision is at least 0.15 (Our new minimum acceptable level)
    MIN_PRECISION = 0.15
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


    logging.info(f"  Optimization Goal: Maximize F{beta}-Score (Min Precision 0.15).")
    logging.info(f"  Optimal Threshold found: {optimal_threshold:.4f}")
    logging.info(f"  Resulting Precision: {optimal_precision:.4f}")
    logging.info(f"  Resulting Recall: {optimal_recall:.4f}")
    logging.info(f"  Resulting F{beta}-Score: {optimal_fbeta:.4f}")
    
    return optimal_threshold, optimal_recall, optimal_precision


def evaluate_model(model, X_test, y_test, optimal_threshold=0.5):
    """Evaluates the model on the test set and prints metrics. (Unchanged logic)"""
    logging.info("6. Evaluating model on Test Set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1] 

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
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
    FIX 4: Uses SHAP to explain the model and prints the analysis hint for Missing Indicators.
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
    logging.info("===== Starting ML Risk Prediction Script (XGBoost + Tuned + Aggressive FN Penalty + Max F2 Threshold) =====")
    start_time = datetime.now()

    # 1. Fetch Data
    df_financials, df_summary_de = fetch_data(start_year=2010) 

    if df_financials.empty:
        logging.error("Stopping script because no financial data could be fetched.")
    else:
        # 2. Create Target Variable
        df_with_y = create_target_variable(df_financials, df_summary_de)

        if not df_with_y.empty:
            # 3. Engineer Features (includes CCC and fixed indicator feature)
            X, y, ids, feature_names_final = engineer_features(df_with_y)

            if not X.empty:
                # 4. Split Data (Time-Based)
                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, ids)

                    # 5. Train Model (uses calculated scale_pos_weight * 1.5)
                    trained_model = train_model(X_train, y_train)

                    # **FIXED STEP: Optimize Threshold for F2-Score**
                    if not X_val.empty:
                        # Maximize F2-Score (Beta=2.0) with Min Precision 0.15 constraint
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