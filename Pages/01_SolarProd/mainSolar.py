import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import SolarModel2 as sm2


# ── STREAMLIT INITIAL PERFORMANCE & STATE CONFIGURATION ────────────────────
st.set_page_config(
    page_title="Kaikohe Solar Calibration Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☀️ Kaikohe Solar Generation Calibration & Simulation Engine")
st.markdown(
    "This interactive dashboard fits a physics-informed LightGBM machine learning model "
    "to decouple weather variances from structural panel performance by tracking the plant's **Performance Ratio (PR)**."
)

# ── SIDEBAR CONTROLS & MODEL METADATA OVERRIDES ──────────────────────────────
st.sidebar.header("🛠️ Plant System Parameters")

capacity_mw = st.sidebar.number_input(
    "Installed Capacity (MW)", 
    min_value=0.1, max_value=500.0, value=65.0, step=1.0,
    help="Nameplate maximum DC output capacity of the solar field."
)

eta_stc = st.sidebar.slider(
    "STC Efficiency Factor (η)", 
    min_value=0.50, max_value=1.00, value=0.85, step=0.01,
    help="Efficiency under Standard Test Conditions (1000 W/m², 25°C)."
)

degradation = st.sidebar.slider(
    "Annual Degradation Rate", 
    min_value=0.00, max_value=0.05, value=0.005, step=0.001, format="%.3f"
)

st.sidebar.markdown("---")
'''
st.sidebar.header("🔄 Data Pipeline Options")
force_refresh = st.sidebar.checkbox(
    "Force Refresh NASA Cache", 
    value=False,
    help="Bypasses the local CSV cache and re-requests historical data from the NASA POWER API."
)'''

# ── STEP 1: LOAD METEOROLOGICAL ENVIRONMENT DATA ────────────────────────────
st.subheader("1. Atmospheric Pipeline Extraction")

@st.cache_data(show_spinner="Extracting 15-year historical weather vectors from NASA...")
def load_and_process_weather():
    raw_weather = sm2.fetch_nasa_weather()
    # This automatically uses the fixed timezone safety check implemented previously
    enriched_weather = sm2.build_solar_features(raw_weather)
    return enriched_weather

try:
    weather_df = load_and_process_weather()
    st.success(f"✅ Meteorological matrix successfully loaded. Array shape: {weather_df.shape}")
    
    # Quick KPI Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Weather Records (Hours)", f"{len(weather_df):,}")
    col2.metric("Max Recorded GHI", f"{weather_df['GHI'].max():.1f} W/m²")
    col3.metric("Avg Ambient Temperature", f"{weather_df['T2M'].mean():.2f} °C")

except Exception as e:
    st.error(f"Failed to ingest atmospheric telemetry data: {e}")
    st.stop()

# ── STEP 2: LOAD ACTUAL PRODUCTION TELEMETRY ─────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("📊 Actual Plant Output Ingestion")
uploaded_file = st.sidebar.file_uploader(
    "Upload Production Data (CSV)", 
    type=["csv"],
    help="Must contain a datetime index matching local time and a generation target column."
)

plant_mw_col = st.sidebar.text_input("Production Column Name", value="MW")

# Internal placeholder simulator if no target file is added yet
@st.cache_data
def generate_synthetic_plant_target(df_weather):
    """Fallback generator to allow application interactivity out-of-the-box."""
    from SolarModel2 import _geometry_only_mw
    sim_idx = df_weather.index
    synthetic_mw = _geometry_only_mw(sim_idx, capacity_mw, eta_stc, 0.82, 0.05, degradation)
    # Inject light Gaussian noise to make it realistic for hexbin testing
    noise = np.random.normal(0, 1.5, size=len(synthetic_mw))
    synthetic_mw = np.clip(synthetic_mw + noise, 0, capacity_mw)
    synthetic_mw[df_weather["elevation"] <= 0] = 0.0
    return pd.DataFrame({plant_mw_col: synthetic_mw}, index=sim_idx)

if uploaded_file is not None:
    try:
        plant_df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
        if plant_mw_col not in plant_df.columns:
            st.sidebar.error(f"Column '{plant_mw_col}' not found in uploaded file targets.")
            st.stop()
        st.sidebar.success("✅ Custom generation targets active.")
    except Exception as e:
        st.sidebar.error(f"Error parsing uploaded file: {e}")
        st.stop()
else:
    st.sidebar.warning("⚠️ No plant file uploaded. Using default data fallback.")
    plant_df = sm2.load_default_prod_data()

# ── STEP 3: MODEL CALIBRATION PHASE ──────────────────────────────────────────
st.subheader("2. Performance Ratio Engine Calibration")
if uploaded_file is None:    
    st.success(f"✅ Default Plant production data successfully loaded. Array shape: {plant_df.shape}")
else:    
    st.success(f"✅ User provided Plant production data successfully loaded. Array shape: {plant_df.shape}")

# Construct model with current dashboard parameters
model = sm2.SolarPRModel(capacity_mw=capacity_mw, 
                     eta_stc=eta_stc, 
                     degradation=degradation)

with st.spinner("Aligning datasets via inner localized joins and optimizing LightGBM parameters..."):
    try:
        model.fit(weather_df, 
                  plant_df, 
                  plant_mw_col=plant_mw_col)
        r2_score = model.get_r2(plant_mw_col)
    except Exception as e:
        st.error(f"Model fitting halted due to an alignment error: {e}")
        st.stop()

# Display Model Health Cards
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("Calibration Model Target R² Score", f"{r2_score:.4f}")
m_col2.metric("Residual Pool Size (Daylight Hours)", f"{len(model.residual_pool):,}")
m_col3.metric("Unbiased Mean Residual Drift", f"{model.residual_pool['residual'].mean():.5f}")

# ── STEP 4: INTERACTIVE DIAGNOSTIC WRAPPER VIEWS ──────────────────────────────
st.subheader("3. Comprehensive System Performance Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data description & Reference", "🔍 View Data Matrix Frames", "📊 Diagnostic Matrix Engine Plots", "🎲 Monte Carlo Path Generator"])

with tab1:
    st.markdown("""
    ### 📊 Data Pipeline Column Definitions & Reference

    The ingestion pipeline handles three distinct categories of feature arrays. All data sets are structurally unified and aligned on a clean, **naive local timezone index** (`Pacific/Auckland` wall-clock time) to completely eradicate datetime offset leakage.

    ---

    #### 1. Atmospheric Telemetry (Raw Ingestion Vectors)
    *Source: 15-Year Historical Hourly Records extracted from the NASA POWER API.*

    * **`index`** *(datetime64[ns])*: Naive local clock timestamp (`Pacific/Auckland`). This represents the primary relational anchor used for matrix alignment across all data frames.
    * **`GHI`** *(float64, W/m²)*: Global Horizontal Irradiance. The total amount of shortwave solar radiation received from above by a flat horizontal surface. This is the **primary physical driver** of solar generation output.
    * **`CLRSKY_GHI`** *(float64, W/m²)*: Clear-Sky Global Horizontal Irradiance. The theoretical maximum solar radiation available at the plant's exact coordinates assuming a completely cloudless atmosphere.
    * **`T2M`** *(float64, °C)*: Dry-bulb ambient air temperature measured at 2 meters above ground level. Crucial for modeling PV panel thermal efficiency losses.
    * **`RH2M`** *(float64, %)*: Relative humidity evaluated at 2 meters. Provides a secondary indicator for tracking atmospheric haze and moisture absorption.
    * **`WS2M`** *(float64, m/s)*: Wind speed measured at 2 meters above ground level. Used to capture structural convective cooling across the solar modules, which temporarily boosts cell efficiency.

    ---

    #### 2. Engineered Features (Solar Geometry & Cycles)
    *Source: Deterministic clear-sky spatial metrics computed via `pvlib.location` and cyclical transformations.*

    * **`elevation`** *(float64, Degrees °)*: Angular height of the sun relative to the local horizon. Ranges from negative values at night up to $\sim 78^\circ$ during peak summer. Values $\le 3.0^\circ$ are used to enforce a strict physical zero-production nighttime mask.
    * **`azimuth`** *(float64, Degrees °)*: The clockwise compass heading of the sun from true North ($0^\circ$ to $360^\circ$). Tracks the directional tracking trajectory of the solar field.
    * **`cloud_factor`** *(float64, Ratio 0.0 to 1.0)*: Evaluated dynamically as $\\text{GHI} / \\text{CLRSKY\_GHI}$. A value of $1.0$ indicates absolute atmospheric clarity; values near $0.0$ represent heavy cloud deck attenuation. Automatically forced to $0.0$ at night.
    * **`hour` / `month` / `dayofyear`** *(int64)*: Discrete components isolated directly from the naive datetime index to construct categorical groupings and conditional bootstrap blocks.
    * **`hour_sin` / `hour_cos`** *(float64, -1.0 to 1.0)*: Diurnal cyclical sine/cosine transforms. Eliminates boundary discontinuity gaps so the LightGBM model treats `23:00` and `00:00` as structurally continuous.
    * **`doy_sin` / `doy_cos`** *(float64, -1.0 to 1.0)*: Seasonal sine/cosine transforms mapping the 365.25-day solar orbit to enable smooth, non-linear seasonal transition boundaries.

    ---

    #### 3. Realized Plant Telemetry (Target Vector Ingestion)
    *Source: File-uploader targeting or geometric fallback arrays used to calibrate the machine learning model.*

    * **`index`** *(datetime64[ns])*: Naive local timestamp. **Crucial Constraints:** Must explicitly align with standard New Zealand standard or daylight savings wall-clock time to hit the inner join matrices cleanly without dropping rows.
    * **`MW`** *(float64, Megawatts)*: The dynamic target column representing the actual, grid-injected electrical power generated by the Kaikohe field at that specific timestamp interval.

    ---

    #### 4. Model Class Internal State Data
    *Source: Generated inside the class object space (`self.merged`) during the model `.fit()` loop.*

    * **`physical_baseline_mw`** *(float64, MW)*: The baseline physical power output ceiling assuming the plant operates at $100\%$ nominal efficiency under current conditions: 
    $$\\text{Capacity} \\times \\eta_{\\text{STC}} \\times \\left( \\frac{\\text{GHI}}{1000} \\right)$$
    * **`actual_pr`** *(float64, Ratio)*: The true historical Performance Ratio, calculated as $\\text{Actual MW} / \\text{physical\\_baseline\\_mw}$. Clamped to $[0.0, 1.2]$ to filter out sensor or data alignment anomalies.
    * **`pred_pr`** *(float64, Ratio)*: The structural Performance Ratio predicted by the trained LightGBM regressor based on the active atmospheric vectors.
    * **`pred_mw`** *(float64, MW)*: Final physics-informed point prediction ($\\text{physical\\_baseline\\_mw} \\times \\text{pred\\_pr}$). Hard-clamped to $0.0\\text{ MW}$ whenever $\\text{elevation} \\le 0$.
    * **`residual`** *(float64)*: The localized modeling error vector ($\\text{actual\\_pr} - \\text{pred\\_pr}$). Saved in a multi-dimensional look-up matrix keyed by `(hour, month)` to form the foundational bootstrap array for Monte Carlo simulation profiles.
    """)
    
with tab2:
    st.markdown("### Native Ingestion Data Frames Inspect Element")
    st.markdown("#### `model.merged` Tracking DataFrame State (First 500 rows)")
    if model.merged is not None:
        st.dataframe(model.merged.head(500), use_container_width=True)
    else:
        st.warning("No calibration data frames created yet.")

with tab3:
    st.markdown("### Engineering Validation Summary")
    # Redirect matplotlib plotting target to clean dashboard layout
    fig_path = "streamlit_solar_diagnostics.png"
    
    # Force the plotting script to execute and output to a specific tracking target file
    sm2.plot_solar_diagnostics2(
        model=model,
        weather_df=weather_df,
        residual_pool=model.residual_pool,
        plant_df=plant_df,
        plant_mw_col=plant_mw_col,
        n_sim_paths=st.slider("Monte Carlo Path Density (Panel 5 View)", 2, 15, 5),
        sim_days=st.slider("Simulation Profile Duration (Days)", 3, 30, 14),
        save_path=fig_path
    )
    
    if os.path.exists(fig_path):
        st.image(fig_path, use_container_width=True)
        # Clear figure from background operational memory cache to ensure stability
        plt.close('all')

with tab4:
    st.markdown("### Interactive Future Simulation Controls")
    sim_cols = st.columns(3)
    mc_days = sim_cols[0].number_input("Forecast Window Length (Days)", value=30, min_value=1, max_value=365)
    mc_paths = sim_cols[1].number_input("Number of Simulated Trajectories", value=100, min_value=5, max_value=1000)
    sim_start_date = sim_cols[2].date_input("Simulation Start Anchor (Local TZ)", value=pd.Timestamp("2024-01-15"))

    if st.button("🚀 Run Monte Carlo Production Forecast"):
        n_steps = int(mc_days * 48)  # 30-minute intervals
        start_ts = pd.Timestamp(sim_start_date)
        
        # Array matrix for multi-path trajectories
        simulation_matrix = np.zeros((mc_paths, n_steps))
        
        progress_bar = st.progress(0)
        for i in range(mc_paths):
            path = model.simulate(
                n_steps=n_steps,
                dt_hours=0.5,
                start_dt=start_ts,
                weather_df=weather_df,
                seed=i
            )
            simulation_matrix[i, :] = path
            progress_bar.progress((i + 1) / mc_paths)
            
        # Transform structural metrics into a statistical aggregate DataFrame
        sim_time_index = pd.date_range(start=start_ts, periods=n_steps, freq="30min")
        agg_df = pd.DataFrame(index=sim_time_index)
        agg_df["P10 (Worst Case)"] = np.percentile(simulation_matrix, 10, axis=0)
        agg_df["P50 (Median Base Case)"] = np.percentile(simulation_matrix, 50, axis=0)
        agg_df["P90 (Best Case Case)"] = np.percentile(simulation_matrix, 90, axis=0)
        
        st.markdown(f"#### Statistical Aggregate Profile across {mc_paths} Scenarios")
        st.line_chart(agg_df)
        
        # Display aggregate generation output expectations
        total_p50_gwh = (agg_df["P50 (Median Base Case)"].sum() * 0.5) / 1000.0
        st.info(f"💡 Expected Median Yield over this {mc_days}-day window: **{total_p50_gwh:.2f} GWh**")

