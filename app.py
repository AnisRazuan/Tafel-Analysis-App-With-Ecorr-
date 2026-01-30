import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# --- HELPER FUNCTION: THE TAFEL SOLVER ---
def fit_tafel(sub_df, method, manual_range=None):
    """
    Fits the Tafel slope for a given subset of data.
    manual_range: A list [log_min, log_max] specific to this branch.
    """
    best_results = None
    best_r2 = -np.inf
    
    # Mode 1: Automatic Sliding Window
    if method == 'Automatic (Recommended)':
        window_size = 6
        r2_thresh = 0.98
        
        num_points = len(sub_df)
        if num_points < window_size:
            return None
            
        for i in range(num_points - window_size + 1):
            subset = sub_df.iloc[i : i + window_size]
            slope, intercept, r_val, p_val, std_err = linregress(subset['log_J'], subset['eta'])
            
            # Filter for reasonable slopes and good R2
            if 0.01 < abs(slope) < 0.5 and (r_val**2) > r2_thresh:
                if (r_val**2) > best_r2:
                    best_r2 = r_val**2
                    best_results = {
                        'slope': slope, 'intercept': intercept, 
                        'r2': best_r2, 'data': subset
                    }
    
    # Mode 2: Manual Range (Log Current)
    else:
        if manual_range is None:
            return None
            
        # manual_range is [log_min, log_max]
        mask = (sub_df['log_J'] >= manual_range[0]) & (sub_df['log_J'] <= manual_range[1])
        subset = sub_df[mask]
        
        if len(subset) > 2:
            slope, intercept, r_val, p_val, std_err = linregress(subset['log_J'], subset['eta'])
            best_results = {
                'slope': slope, 'intercept': intercept, 
                'r2': r_val**2, 'data': subset
            }

    return best_results

# ==========================================
# MAIN APP STARTS HERE
# ==========================================
st.set_page_config(page_title="Advanced Tafel Tool", layout="wide")
st.title("⚡ Dual-Branch Tafel Analysis")
st.markdown("Upload data to automatically detect **Ecorr**, **HER**, and **OER** regions simultaneously.")

# --- 1. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            df_raw = pd.read_excel(uploaded_file, header=None)

        # Standardize Columns
        V_raw_mV = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
        J_raw = pd.to_numeric(df_raw
