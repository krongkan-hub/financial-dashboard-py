# run_ml_risk.py (เวอร์ชันแก้ไขสมบูรณ์: ป้องกัน SimpleImputer ตัด Features ทิ้ง)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import joblib # For saving the model
import shap # For model explanation
# NOTE: ต้องติดตั้ง imblearn (pip install imbalanced-learn) หากต้องการใช้ SMOTE
from imblearn.over_sampling import SMOTE # For handling imbalanced data

# --- Database Interaction ---
from app import db, server, FactFinancialStatements, FactCompanySummary
from sqlalchemy import func, distinct, and_, text
from sqlalchemy.orm import aliased

# --- Machine Learning ---
from sklearn.model_selection import train_test_split # Initially, then maybe TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# --- Configuration ---
# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for Target Variable Definition
DE_RATIO_THRESHOLD = 5.0
ICR_THRESHOLD = 1.0
CONSECUTIVE_LOSS_PERIODS = 6 # Number of quarters for consecutive net loss

# Feature Engineering Constants
YEARS_FOR_CHANGE = 1 # Look back 1 year for change calculations
MA_PERIODS = 4 # Moving average over 4 quarters

# Data Splitting Configuration (Example: Time-based split)
# ใช้หน่วยเป็นเดือนแทนปีเศษส่วน เพื่อป้องกัน ValueError
VALIDATION_PERIOD_MONTHS = 6 # Use 6 months for validation
TEST_PERIOD_MONTHS = 12      # Use 1 most recent year for testing (12 months)

# Model Saving Configuration
MODEL_DIR = "models"
MODEL_FILENAME = "trained_risk_model.joblib"
SCALER_FILENAME = "scaler.joblib" # Save the scaler too!
IMPUTER_FILENAME = "imputer.joblib" # Save the imputer too!

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper Functions ---
def calculate_ttm(series):
    """Calculates Trailing Twelve Months sum for a pandas Series grouped by ticker."""
    return series.rolling(window=4, min_periods=4).sum()

def calculate_change(series, periods):
    """Calculates the change over a number of periods."""
    return series.diff(periods=periods)

# --- Core Functions ---

def fetch_data(start_year=2015):
    """
    Fetches required financial data from the database.
    """
    logging.info(f"1. Fetching data from database starting from year {start_year}...")
    start_date = date(start_year, 1, 1)

    # Metrics needed from FactFinancialStatements
    required_metrics = [
        # For Target Variable 
        'Stockholders Equity', 'EBIT', 'Interest Expense', 'Net Income',
        # For Features 
        'Total Revenue', 'Operating Cash Flow', 'Current Assets', 
        'Current Liabilities', 'Inventory', 'Total Assets'
    ]

    try:
        with server.app_context():
            # Query FactFinancialStatements
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

            # Query FactCompanySummary for D/E Ratio (use latest available per quarter end)
            FCS = aliased(FactCompanySummary)
            
            quarter_group = (func.strftime('%Y', FCS.date_updated) + '-' +
                             ((func.strftime('%m', FCS.date_updated).cast(db.Integer) + 2) / 3).cast(db.String)
                            ).label('year_quarter') # Label for clarity

            subquery = db.session.query(
                FCS.ticker,
                quarter_group, # Group by the calculated year-quarter string
                func.max(FCS.date_updated).label('max_summary_date')
            ).group_by(FCS.ticker, quarter_group).subquery() # Group by ticker and year_quarter

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
                    FactCompanySummary.date_updated == subquery.c.max_summary_date # Join condition remains the same
                )
            ).filter(FactCompanySummary.date_updated >= start_date).statement 
            
            df_fcs = pd.read_sql(stmt_fcs, db.engine) 
            
            # Calculate report_date_approx from the 'year_quarter' column
            df_fcs[['Year', 'Quarter']] = df_fcs['year_quarter'].str.split('-', expand=True)
            df_fcs['Year'] = df_fcs['Year'].astype(int)
            df_fcs['Quarter'] = df_fcs['Quarter'].astype(float).astype(int)
            # Calculate the first month of the quarter: Q1->1, Q2->4, Q3->7, Q4->10
            df_fcs['Quarter_Start_Month'] = (df_fcs['Quarter'] - 1) * 3 + 1
            # Create a date for the first day of the quarter's starting month
            df_fcs['Quarter_Start_Date'] = pd.to_datetime(
                df_fcs['Year'].astype(str) + '-' + df_fcs['Quarter_Start_Month'].astype(str) + '-01',
                errors='coerce'
            )
            # Calculate the Quarter End date
            df_fcs['report_date_approx'] = df_fcs['Quarter_Start_Date'] + pd.offsets.QuarterEnd(0)

            # Drop intermediate columns if no longer needed
            df_fcs = df_fcs.drop(columns=['Year', 'Quarter', 'Quarter_Start_Month', 'Quarter_Start_Date', 'year_quarter'])
            
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
    Includes a time lag: Y at time T+lag is predicted using X at time T.
    """
    logging.info("2. Creating Target Variable (Y)...")
    if df_ffs.empty:
        logging.warning("Financial statement data is empty, cannot create target variable.")
        return pd.DataFrame()

    df = df_ffs.copy()
    # Ensure dataframe is sorted
    df = df.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)

    # --- Calculate necessary components ---
    # TTM ICR
    df['EBIT_TTM'] = df.groupby('ticker')['EBIT'].transform(calculate_ttm)
    # Handle zero or negative interest expense carefully
    df['Interest Expense_TTM'] = df.groupby('ticker')['Interest Expense'].transform(lambda x: x.rolling(window=4, min_periods=4).sum().replace(0, np.nan)) # Replace 0 with NaN temporarily
    df['TTM_ICR'] = df['EBIT_TTM'] / df['Interest Expense_TTM']
    # If Interest Expense was originally 0 or positive EBIT / 0 -> inf, treat as non-risky for ICR threshold
    # If EBIT is negative and Interest Expense > 0 -> ICR is negative (risky)
    # If both EBIT and Interest Expense are 0 or NaN, ICR will be NaN
    df['TTM_ICR'] = df['TTM_ICR'].replace([np.inf, -np.inf], np.nan) # Clean up potential infs

    # Consecutive Net Losses
    df['is_loss'] = (df['Net Income'] < 0).astype(int)
    # Calculate rolling sum of losses over the specified window
    df['consecutive_losses'] = df.groupby('ticker')['is_loss'].transform(
        lambda x: x.rolling(window=CONSECUTIVE_LOSS_PERIODS, min_periods=CONSECUTIVE_LOSS_PERIODS).sum()
    )

    # Negative Equity
    df['is_negative_equity'] = (df['Stockholders Equity'] < 0).astype(int)

    # Merge D/E Ratio (approximate merge based on quarter end)
    df['report_date_approx'] = df['report_date'] + pd.offsets.QuarterEnd(0)
    
    # *** แก้ไข: เพิ่ม tolerance เป็น 1 ปี เพื่อให้ Merge D/E Ratio สำเร็จมากขึ้น ***
    df = pd.merge_asof(df.sort_values('report_date_approx'),
                       df_fcs[['ticker', 'report_date_approx', 'de_ratio']].sort_values('report_date_approx'),
                       on='report_date_approx',
                       by='ticker',
                       direction='backward', # Find latest D/E on or before the report date
                       tolerance=pd.Timedelta('365 days')) # เพิ่มเป็น 1 ปี
    # *** สิ้นสุดแก้ไข ***

    # --- Apply Risk Rules ---
    rule1 = df['is_negative_equity'] == 1
    # D/E ratio อาจเป็น NaN หาก Merge ไม่สำเร็จ
    rule2 = (df['de_ratio'] > DE_RATIO_THRESHOLD) & (df['TTM_ICR'] < ICR_THRESHOLD)
    rule3 = df['consecutive_losses'] >= CONSECUTIVE_LOSS_PERIODS

    # Combine rules: If any rule is met, Y_raw = 1
    df['Y_raw'] = ((rule1) | (rule2) | (rule3)).astype(int)

    # --- Introduce Time Lag ---
    # We want to predict risk *in the future* (e.g., 1 year = 4 quarters ahead)
    lag_periods = 4
    df['Y'] = df.groupby('ticker')['Y_raw'].shift(-lag_periods)
    logging.info(f"Target variable Y created with a lag of {lag_periods} quarters.")

    # Drop rows where the lagged Y is NaN (i.e., the last 'lag_periods' for each ticker)
    df_final = df.dropna(subset=['Y']).copy()
    df_final['Y'] = df_final['Y'].astype(int)

    logging.info(f"Created Y for {len(df_final)} data points.")
    logging.info(f"Distribution of Y: \n{df_final['Y'].value_counts(normalize=True)}")

    # Select relevant columns for features and the final Y
    cols_to_keep = ['ticker', 'report_date', 'Y'] + list(df_ffs.columns.drop(['ticker', 'report_date'])) + ['de_ratio'] # Keep base metrics + D/E
    df_final = df_final[cols_to_keep]

    return df_final


def engineer_features(df):
    """
    Calculates trend/change features, handles missing values, and scales data.
    """
    logging.info("3. Engineering Features (X)...")
    if df.empty or 'Y' not in df.columns:
        logging.warning("Input DataFrame is empty or missing Y column.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []

    df_eng = df.copy()
    df_eng = df_eng.sort_values(by=['ticker', 'report_date']).reset_index(drop=True)
    periods_per_year = 4

    # --- Calculate Base Ratios / Size Features ---
    df_eng['Current Ratio'] = df_eng['Current Assets'] / df_eng['Current Liabilities'].replace(0, np.nan)
    
    # เพิ่มการคำนวณ Total Assets (ln) และ Operating Margin
    df_eng['Total Assets (ln)'] = np.log(df_eng['Total Assets'].replace(0, np.nan))
    if 'Operating Margin' not in df_eng.columns and 'EBIT' in df_eng.columns and 'Total Revenue' in df_eng.columns:
        df_eng['Operating Margin'] = df_eng['EBIT'] / df_eng['Total Revenue'].replace(0, np.nan)

    # --- Calculate YoY Growth Features ---
    growth_cols = ['Total Revenue', 'Net Income', 'Operating Cash Flow']
    for col in growth_cols:
        if col in df_eng.columns:
            # เพิ่ม fill_method=None เพื่อหลีกเลี่ยง FutureWarning ในอนาคต
            df_eng[f'{col}_YoY_Growth'] = df_eng.groupby('ticker')[col].pct_change(periods=periods_per_year, fill_method=None)

    # --- Calculate Change in Key Ratios (Over 1 Year) ---
    change_cols_map = {'de_ratio': 'D/E Ratio', 'Operating Margin': 'Operating Margin', 'Current Ratio': 'Current Ratio'} 
    
    for col_raw, col_display in change_cols_map.items():
         if col_raw in df_eng.columns:
            df_eng[f'Change in {col_display}'] = df_eng.groupby('ticker')[col_raw].transform(calculate_change, periods=periods_per_year)

    # --- Calculate Moving Averages ---
    ma_cols = ['Net Income', 'ROE'] 
    
    # Calculate ROE if needed
    if 'ROE' not in df_eng.columns and 'Net Income' in df_eng.columns and 'Stockholders Equity' in df_eng.columns:
         # Use TTM Net Income and Average Equity over 4 quarters for a more stable ROE
         df_eng['Net Income_TTM'] = df_eng.groupby('ticker')['Net Income'].transform(calculate_ttm)
         df_eng['Avg Equity_TTM'] = df_eng.groupby('ticker')['Stockholders Equity'].transform(lambda x: x.rolling(window=4, min_periods=4).mean())
         df_eng['ROE'] = df_eng['Net Income_TTM'] / df_eng['Avg Equity_TTM'].replace(0, np.nan)

    for col in ma_cols:
         if col in df_eng.columns:
            df_eng[f'MA_{col}'] = df_eng.groupby('ticker')[col].transform(lambda x: x.rolling(window=MA_PERIODS, min_periods=MA_PERIODS).mean())

    # --- Final Feature Selection ---
    feature_cols = [
        'de_ratio', 'Current Ratio', 'Operating Margin', 'Total Assets (ln)', # Base Ratios / Size
        'Total Revenue_YoY_Growth', 'Net Income_YoY_Growth', 'Operating Cash Flow_YoY_Growth', # Growth
        'Change in D/E Ratio', 'Change in Operating Margin', 'Change in Current Ratio', # Changes
        'MA_Net Income', 'MA_ROE' # Moving Averages
    ]
    
    # Ensure only existing columns are selected
    feature_cols = [col for col in feature_cols if col in df_eng.columns]

    X = df_eng[feature_cols].copy()
    y = df_eng['Y'].copy()

    # Store identifiers and dates before dropping NaNs/imputing
    identifiers = df_eng[['ticker', 'report_date']].copy()

    # --- Handle Infinities and NaNs ---
    X = X.replace([np.inf, -np.inf], np.nan)

    # --- CRITICAL FIX: Ensure all-NaN features are imputed with 0 to prevent SimpleImputer from failing ---
    X_to_impute = X.copy()

    # 1. Identify features that are entirely NaN in the current subset
    all_nan_cols = X_to_impute.columns[X_to_impute.isnull().all()].tolist()
    
    if all_nan_cols:
        logging.warning(f"Forcing imputation for all-NaN features by filling with 0: {all_nan_cols}")
        # Fill all-NaN columns with 0. This gives SimpleImputer a non-missing value to work with 
        # and prevents it from skipping the column and reducing the dimensionality.
        X_to_impute[all_nan_cols] = X_to_impute[all_nan_cols].fillna(0) 

    X_partial = X_to_impute
    feature_names_imputed = X_partial.columns
    
    if X_partial.empty:
        logging.error("No valid features remain after removing all-NaN columns.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), []

    # 2. Impute remaining NaNs (for the columns that only had some missing values)
    imputer = SimpleImputer(strategy='median')
    X_imputed_partial = imputer.fit_transform(X_partial)
    
    # 3. Convert back to DataFrame (The dimension will now match 810, 12)
    X_imputed_df = pd.DataFrame(X_imputed_partial, columns=feature_names_imputed, index=X_partial.index)

    # 4. Update feature names and X for scaling
    X = X_imputed_df 
    
    # --- Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names_imputed)

    logging.info(f"Selected {len(feature_names_imputed)} features for model training.")
    logging.info(f"Feature names: {list(feature_names_imputed)}")

    # Save the imputer and scaler
    joblib.dump(imputer, os.path.join(MODEL_DIR, IMPUTER_FILENAME))
    joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_FILENAME))
    logging.info(f"Imputer saved to {os.path.join(MODEL_DIR, IMPUTER_FILENAME)}")
    logging.info(f"Scaler saved to {os.path.join(MODEL_DIR, SCALER_FILENAME)}")

    return X_scaled_df, y, identifiers, feature_names_imputed


def split_data(X, y, identifiers):
    """
    Splits data into train, validation, and test sets based on time.
    """
    logging.info("4. Splitting data into Train, Validation, Test sets (Time-Based)...")

    # Combine X, y, and identifiers for easy splitting based on date
    data_full = pd.concat([identifiers.reset_index(drop=True),
                           X.reset_index(drop=True),
                           y.reset_index(drop=True)], axis=1)
    data_full = data_full.sort_values(by='report_date')

    if data_full.empty:
        raise ValueError("Cannot split empty DataFrame.")

    # Determine split dates
    max_date = data_full['report_date'].max()
    # ใช้ DateOffset(months=...) 
    test_start_date = max_date - pd.DateOffset(months=TEST_PERIOD_MONTHS) + pd.Timedelta(days=1)
    val_start_date = test_start_date - pd.DateOffset(months=VALIDATION_PERIOD_MONTHS)
    
    # Perform the split
    test_data = data_full[data_full['report_date'] >= test_start_date]
    val_data = data_full[(data_full['report_date'] >= val_start_date) & (data_full['report_date'] < test_start_date)]
    train_data = data_full[data_full['report_date'] < val_start_date]

    if train_data.empty or val_data.empty or test_data.empty:
        logging.warning("One or more data splits are empty. Adjust split years or check data range.")
        # Fallback to random split if time split fails (less ideal)
        logging.warning("Falling back to random train/test split (80/20). No validation set.")
        # ใช้ Stratify เพื่อรักษาอัตราส่วน Y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, y_val = pd.DataFrame(), pd.Series() # No validation set in fallback
        
        # ปรับ identifiers ให้สอดคล้องกับการสุ่ม
        train_indices, test_indices = train_test_split(identifiers.index, test_size=0.2, random_state=42, stratify=y.values)
        train_ids = identifiers.loc[train_indices]
        test_ids = identifiers.loc[test_indices]
        val_ids = pd.DataFrame()
    else:
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

    # Log class distribution in each set
    logging.info(f"  Train Y distribution:\n{y_train.value_counts(normalize=True)}")
    if not y_val.empty:
        logging.info(f"  Validation Y distribution:\n{y_val.value_counts(normalize=True)}")
    logging.info(f"  Test Y distribution:\n{y_test.value_counts(normalize=True)}")


    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model after applying SMOTE for balancing.
    """
    logging.info("4.1 Applying SMOTE to balance training data...")
    try:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"  Train Y distribution (After SMOTE):\n{y_train_res.value_counts(normalize=True)}")
    except NameError:
        logging.warning("SMOTE is not defined (imblearn not imported/installed). Skipping balancing.")
        X_train_res, y_train_res = X_train, y_train
        
    logging.info("5. Training RandomForest model...")
    # Initialize model 
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)

    model.fit(X_train_res, y_train_res)
    logging.info("Model training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints metrics.
    Includes evaluation at different thresholds to optimize for Recall/Precision.
    """
    logging.info("6. Evaluating model on Test Set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1

    # --- Evaluation at Threshold 0.5 (Default) ---
    # ใช้ zero_division=0 เพื่อป้องกัน Warning หากไม่มีการทำนายเป็น Class 1 เลย
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    logging.info("--- Evaluation Metrics (Threshold 0.5) ---")
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info(f"AUC-ROC:   {roc_auc:.4f}")
    logging.info("--- Confusion Matrix ---")
    logging.info(f"\n{cm}")
    
    logging.info("--- Classification Report (Threshold 0.5) ---")
    logging.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # --- Evaluation at a Lower Threshold (Example: 0.15 for higher Recall) ---
    low_threshold = 0.15 
    y_pred_low = (y_pred_proba >= low_threshold).astype(int)
    
    # ใช้ zero_division=0 เพื่อป้องกัน Warning หากไม่มีการทำนายเป็น Class 1 เลย
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
        # Use TreeExplainer for tree-based models like RandomForest
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values - using a subset of training data can speed this up if needed
        # ใช้ X_train เป็น background data
        shap_values = explainer.shap_values(X_train)

        # shap_values might be a list [shap_values_for_class0, shap_values_for_class1]
        # For binary classification, we usually focus on class 1
        shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

        # --- SHAP Summary Plot ---
        logging.info("Generating SHAP Summary Plot...")
        shap.summary_plot(shap_values_class1, X_train, feature_names=feature_names, show=False)
        logging.info("(SHAP plot display/saving would happen here if matplotlib is configured)")

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
    logging.info("===== Starting ML Risk Prediction Script =====")
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
                # 4. Split Data (Note: Train_model now handles SMOTE)
                try:
                     # ใช้ try-except สำหรับ split_data เพื่อจับ ValueError หาก split ล้มเหลว
                     X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, ids)

                     # 5. Train Model (รวมขั้นตอน SMOTE ไว้ใน train_model)
                     trained_model = train_model(X_train, y_train)

                     # 6. Evaluate Model (on Test set)
                     evaluate_model(trained_model, X_test, y_test)

                     # 7. Explain Model (using Training set for background)
                     explain_model(trained_model, X_train, feature_names_final)

                     # 8. Save Model
                     save_model(trained_model)

                     # --- Optional: Predict latest probabilities (Placeholder) ---
                     # predict_latest(trained_model, ...)
                     logging.info("Placeholder: Prediction for latest data would happen here.")
                     # --------------------------------------------------------

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