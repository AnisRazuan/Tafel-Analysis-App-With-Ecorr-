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
        J_raw = pd.to_numeric(df_raw.iloc[:, 1], errors='coerce')
        
        # Clean NaNs
        df = pd.DataFrame({'V_raw_mV': V_raw_mV, 'J_raw': J_raw}).dropna()
        
        # --- 2. PRE-PROCESSING & ECORR ---
        df['V'] = df['V_raw_mV'] / 1000.0  # mV -> V
        
        # CALCULATE ECORR
        min_idx = df['J_raw'].abs().idxmin()
        E_corr = df.loc[min_idx, 'V']
        
        # Log Current & Overpotential
        df['J_abs'] = df['J_raw'].abs()
        df['log_J'] = np.log10(df['J_abs'])
        df['eta'] = df['V'] 

        # Sort for clean plotting
        df = df.sort_values(by='log_J')

        # --- 3. SPLIT BRANCHES ---
        df_anodic = df[df['J_raw'] > 0].copy()
        df_cathodic = df[df['J_raw'] < 0].copy()

        st.success(f"**Data Loaded:** Ecorr estimated at **{E_corr:.4f} V**")
        
        # --- 4. USER SETTINGS (UPDATED) ---
        method = st.radio("Analysis Method", ["Automatic (Recommended)", "Manual (Expert)"])
        
        manual_anodic_range = None
        manual_cathodic_range = None
        
        if method == "Manual (Expert)":
            st.write("---")
            col_a, col_c = st.columns(2)
            
            # --- ANODIC INPUTS (RED) ---
            with col_a:
                st.markdown("#### 🔴 Anodic (OER) Settings")
                st.info("Positive Currents")
                a_start = st.number_input("Start Current (mA)", value=1.5, key="a_s")
                a_end = st.number_input("End Current (mA)", value=5.0, key="a_e")
                
                if a_start > 0 and a_end > 0:
                    l_min = np.log10(min(a_start, a_end))
                    l_max = np.log10(max(a_start, a_end))
                    manual_anodic_range = [l_min, l_max]

            # --- CATHODIC INPUTS (BLUE) ---
            with col_c:
                st.markdown("#### 🔵 Cathodic (HER) Settings")
                st.info("Negative Currents (enter as positive value)")
                c_start = st.number_input("Start Current (mA)", value=1.5, key="c_s")
                c_end = st.number_input("End Current (mA)", value=5.0, key="c_e")
                
                if c_start > 0 and c_end > 0:
                    l_min = np.log10(min(c_start, c_end))
                    l_max = np.log10(max(c_start, c_end))
                    manual_cathodic_range = [l_min, l_max]

        # --- 5. RUN ANALYSIS ---
        results = {}
        
        # Fit Anodic (using specific Anodic range)
        if len(df_anodic) > 0:
            results['Anodic'] = fit_tafel(df_anodic, method, manual_anodic_range)
            
        # Fit Cathodic (using specific Cathodic range)
        if len(df_cathodic) > 0:
            results['Cathodic'] = fit_tafel(df_cathodic, method, manual_cathodic_range)

        # --- 6. DISPLAY RESULTS & PLOT ---
        fig = go.Figure()
        
        # Plot Raw Data
        fig.add_trace(go.Scatter(x=df['log_J'], y=df['V'], mode='markers', 
                                 name='Raw Data', marker=dict(color='lightgrey', size=4)))

        # Results Display
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Ecorr (V)", f"{E_corr:.4f}")
        
        for branch_name, res in results.items():
            if res:
                slope = res['slope']
                intercept = res['intercept']
                r2 = res['r2']
                color = 'red' if branch_name == 'Anodic' else 'blue'
                
                # Plot Fit
                fit_x = res['data']['log_J']
                fit_y = slope * fit_x + intercept
                fig.add_trace(go.Scatter(x=fit_x, y=fit_y, mode='lines', 
                                         name=f"{branch_name} Fit", line=dict(color=color, width=3)))
                
                # Show Stats
                col_res2.write(f"**{branch_name} Slope:** {slope*1000:.1f} mV/dec")
                col_res3.write(f"**{branch_name} R²:** {r2:.4f}")

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
