import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress

# --- HELPER FUNCTION: THE TAFEL SOLVER ---
def fit_tafel(sub_df, method, manual_range=None):
    """
    Fits the Tafel slope for a given subset of data (Anodic or Cathodic).
    Returns a dictionary of results and the fitted line data.
    """
    best_results = None
    best_r2 = -np.inf
    
    # Mode 1: Automatic Sliding Window
    if method == 'Automatic (Recommended)':
        window_size = 6
        r2_thresh = 0.98  # Robust threshold
        
        # Scan through the data
        num_points = len(sub_df)
        if num_points < window_size:
            return None # Not enough data
            
        for i in range(num_points - window_size + 1):
            subset = sub_df.iloc[i : i + window_size]
            
            # Perform Linear Regression
            slope, intercept, r_val, p_val, std_err = linregress(subset['log_J'], subset['eta'])
            
            # Filter for reasonable slopes (0.01 to 0.5 V/dec usually) and good R2
            if 0.01 < abs(slope) < 0.5 and (r_val**2) > r2_thresh:
                if (r_val**2) > best_r2:
                    best_r2 = r_val**2
                    best_results = {
                        'slope': slope, 'intercept': intercept, 
                        'r2': best_r2, 'data': subset
                    }
    
    # Mode 2: Manual Range (Log Current)
    else:
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

        # Standardize Columns (Col 0 = Voltage, Col 1 = Current)
        V_raw_mV = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce')
        J_raw = pd.to_numeric(df_raw.iloc[:, 1], errors='coerce')
        
        # Clean NaNs
        df = pd.DataFrame({'V_raw_mV': V_raw_mV, 'J_raw': J_raw}).dropna()
        
        # --- 2. PRE-PROCESSING & ECORR ---
        # Convert Units
        df['V'] = df['V_raw_mV'] / 1000.0  # mV -> V
        
        # CALCULATE ECORR (Potential where Current is closest to 0)
        # We find the index where absolute current is minimum
        min_idx = df['J_raw'].abs().idxmin()
        E_corr = df.loc[min_idx, 'V']
        
        # Calculate Overpotential (eta) and Log Current (log_J)
        # Note: We use absolute current for the log plot
        df['J_abs'] = df['J_raw'].abs()
        df['log_J'] = np.log10(df['J_abs'])
        
        # For plot, we use V directly (or you can subtract Ecorr if you prefer overpotential)
        df['eta'] = df['V'] 

        # Sort for clean plotting (Fixes Zigzag issue)
        df = df.sort_values(by='log_J')

        # --- 3. SPLIT BRANCHES ---
        # Anodic = Positive Current | Cathodic = Negative Current
        df_anodic = df[df['J_raw'] > 0].copy()
        df_cathodic = df[df['J_raw'] < 0].copy()

        st.success(f"**Data Loaded:** Ecorr estimated at **{E_corr:.4f} V**")
        
        # --- 4. USER SETTINGS ---
        method = st.radio("Analysis Method", ["Automatic (Recommended)", "Manual (Expert)"])
        manual_range = None
        
        if method == "Manual (Expert)":
            st.info("Enter Log-Current Range (X-axis). Example: 1.5 to 5.0 mA.")
            c1, c2 = st.columns(2)
            j_start = c1.number_input("Start Current", value=1.5)
            j_end = c2.number_input("End Current", value=5.0)
            
            # Convert to Log10 space for the fitter
            if j_start > 0 and j_end > 0:
                l_min = np.log10(min(j_start, j_end))
                l_max = np.log10(max(j_start, j_end))
                manual_range = [l_min, l_max]
            else:
                st.error("Current must be > 0")

        # --- 5. RUN ANALYSIS (THE MAGIC LOOP) ---
        results = {}
        
        # Run fit on Anodic
        if len(df_anodic) > 0:
            results['Anodic'] = fit_tafel(df_anodic, method, manual_range)
            
        # Run fit on Cathodic
        if len(df_cathodic) > 0:
            results['Cathodic'] = fit_tafel(df_cathodic, method, manual_range)

        # --- 6. DISPLAY RESULTS & PLOT ---
        
        # Setup Plot
        fig = go.Figure()
        
        # Plot Raw Data (Grey)
        fig.add_trace(go.Scatter(x=df['log_J'], y=df['V'], mode='markers', 
                                 name='Raw Data', marker=dict(color='lightgrey', size=4)))

        # Display Metrics
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Ecorr (V)", f"{E_corr:.4f}")
        
        # Process & Plot Fits
        for branch_name, res in results.items():
            if res:
                slope = res['slope']
                intercept = res['intercept']
                r2 = res['r2']
                
                # Determine Color (Red for Anodic, Blue for Cathodic)
                color = 'red' if branch_name == 'Anodic' else 'blue'
                
                # Plot the Fit Line
                fit_x = res['data']['log_J']
                fit_y = slope * fit_x + intercept
                
                fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', 
                                         name=f"{branch_name} Fit", line=dict(color=color, width=3)))
                
                # Show Stats
                col_res2.write(f"**{branch_name} Slope:** {slope*1000:.1f} mV/dec")
                col_res3.write(f"**{branch_name} R²:** {r2:.4f}")

        # Final Plot Layout
        fig.update_layout(
            title="Tafel Analysis (Dual Branch)",
            xaxis_title="Log Current Density (log A)",
            yaxis_title="Potential (V)",
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
