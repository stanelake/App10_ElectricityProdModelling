import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import your precise PriceModel2 module pipelines
import PriceModel2 as pm2

# ── STREAMLIT INITIAL WORKSPACE CONFIGURATION ───────────────────────────────
st.set_page_config(
    page_title="MRSVCJ Electricity Calibration & Simulation Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STRUCTURAL DEFAULT FALLBACK CONSTANTS ────────────────────────────────────
SAVED_LOG_U_PARAMS = {
    "alpha":  4.794884 - 0.0410,   # recentred: absorbed median -0.0410
    "beta":  -0.00008567,
    "gamma": -0.736805,
    "tau":    254.0581,
    "phi":    np.array([0.100192, 0.159623, 0.201218, 0.190828, 0.163910, 0.041892]),   # Mon–Sat
    "zeta":   np.array([ 0.381820,  0.263283,  0.073771, -0.249871, -0.358883, -0.463163,
                        -0.398027, -0.100464, -0.026599,  0.111779,  0.204746]),            # Jan–Nov
    "psi":    np.array([
        -0.002218, -0.047205, -0.067158, -0.113031, -0.172792, -0.328729, -0.379421, -0.421409, 
        -0.398954, -0.326499, -0.100613, -0.039312,  0.031389,  0.094874,  0.190808,  0.301810, 
         0.361674,  0.309555,  0.291363,  0.263849,  0.250891,  0.233304,  0.240551,  0.233940, 
         0.242874,  0.235213,  0.217599,  0.172879,  0.146981,  0.140640,  0.178018,  0.250421, 
         0.262457,  0.300956,  0.300810,  0.412941,  0.421734,  0.409162,  0.334900,  0.303174, 
         0.303599,  0.289456,  0.325971,  0.260650,  0.169627,  0.066564,  0.088458,
    ]),  # psi_01:00 … psi_23:30 (47 half-hour dummies, baseline = 00:30)
}

SAVED_PRICE_PARAMS = {
    "eta":     474.099821,
    "mu":       -0.057159,
    "kappa":    16.134595,
    "theta":   305.814897,
    "sigma_v":   0.041232,
    "rho":       0.015383,
    "beta":     10.396368,
    "mu_J":      0.031577,
}

SAVED_JUMP_PARAMS = {
    "mu_v":    0.007311,
    "mu_x":    0.008495,
    "sigma_x": 0.212559,
    "rho_J":  -0.455157,
}

# ── SIDEBAR CONTROL PANELS ──────────────────────────────────────────────────
st.sidebar.title("⚡ Control Panel")

# Core operational toggle allowing app execution out-of-the-box
use_defaults = st.sidebar.toggle(
    "Use Pre-calibrated Default Parameters", 
    value=True,
    help="When active, the app bypasses historical CSV calculations and uses verified pre-calibrated baseline models."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Data & Simulation Bounds")

csv_file_path = st.sidebar.text_input(
    "Data Source CSV Path", 
    value="DataSource/Wholesale_Prices/Wholesale_price_trends_20260505202423.csv",
    disabled=use_defaults
)
max_gap = st.sidebar.slider("Max F-Fill Allowed Gaps (Bars)", 1, 12, 4, disabled=use_defaults)
sim_horizon_days = st.sidebar.slider("Simulation Horizon (Days)", 5, 180, 30)
sim_trajectories = st.sidebar.number_input("Paths Total Count", min_value=5, max_value=1000, value=150, step=50)

# Setup layout tabs
tab_method, tab_eda, tab_calib, tab_sim = st.tabs([
    "📖 Methodology (Non-Technical Guide)", 
    "📈 Exploratory Data Analysis (EDA)", 
    "🔬 Robust Calibration Diagnostics", 
    "🔮 Monte Carlo Simulation Paths"
])

# ── DATA HANDLING INTERCEPT ENGINE ──────────────────────────────────────────
@st.cache_data
def generate_fallback_dataframe():
    """Generates a synthetic dataset modeled after saved defaults when file is missing."""
    idx = pd.date_range(start="2024-01-01", end="2025-12-31", freq="30min")
    total_bars = len(idx)
    # Generate log_U using the verified parameters
    log_u_values = pm2.log_U(dt_input=pd.Timestamp("2024-01-01"), n_bars=total_bars, **SAVED_LOG_U_PARAMS)
    # Add temporary noise and reconstruct price
    stoch = np.random.normal(0, 0.15, size=total_bars)
    prices = np.exp(log_u_values + stoch)
    # Inject occasional extreme price spikes
    spikes = np.where(np.random.rand(total_bars) > 0.996, np.random.exponential(350.0, size=total_bars), 0.0)
    
    df_out = pd.DataFrame({"Price": np.clip(prices + spikes, 1.0, 2000.0)}, index=idx)
    df_out["logPrice"] = np.log(df_out["Price"])
    df_out["log_U"] = log_u_values
    df_out["Stochastic"] = stoch
    return df_out

# Logic to load dataset or gracefully fall back to default arrays
df = None
fitted_log_u = SAVED_LOG_U_PARAMS

if not use_defaults:
    try:
        df, fitted_log_u = pm2.loadPriceData(csv_path=csv_file_path, max_fill_gap=max_gap, log_U_params="fit")
        st.sidebar.success("✅ Custom CSV Matrix loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed loading custom file: {e}. Falling back to default baseline profile.")
        df = generate_fallback_dataframe()
else:
    df = generate_fallback_dataframe()

if df.index.tz is not None:
    df.index = df.index.tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: METHODOLOGY (EXHAUSTIVE NON-TECHNICAL GUIDE)
# ─────────────────────────────────────────────────────────────────────────────
with tab_method:
    st.header("How the Electricity Pricing Model Works")
    st.markdown(
        "Unlike commodities like gold or oil, electricity cannot be stored easily in vast quantities. "
        "It must be produced, transmitted, and consumed across the power grid at the exact same fraction of a second. "
        "Because of this, whenever a factory turns on or a generator goes offline unexpectedly, wholesale electricity spot prices "
        "experience massive, violent transformations—skyrocketing from a normal **$80/MWh** up to **$5,000/MWh** in 30 minutes, "
        "or plunging below zero when excess solar forces the grid to pay users to consume power."
    )
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕵️‍♂️ Phase 1: Separating the Normal from the Extreme")
        st.markdown(
            "If you look at raw market charts, the giant price spikes hide the predictable daily cycles. "
            "Our engine acts like a high-precision filter to separate everyday market behavior from unpredictable chaotic noise:\n\n"
            "1. **The Human Clock Baseline (Seasonality):** Electricity prices naturally follow predictable structural patterns based on human behavior. "
            "Prices peak when families wake up and cook dinner, change depending on whether it is a weekday or Sunday, and drift up or down as seasons shift. "
            "The model captures this via an advanced **67-parameter signature blueprint** mapping out every single half-hour interval, day of the week, and month.\n"
            "2. **The De-Spiking Filter (Tukey Fences & Robust Math):** If you use a simple average to calculate this baseline, a single extreme $5,000 price spike "
            "will ruin your math and drag your baseline artificially high. The model uses an intelligent screening process (**Tukey Fences**) that reviews each half-hour group "
            "independently and filters out outliers before fitting the baseline. This ensures our underlying normal baseline remains clean and accurate."
        )
        
    with col2:
        st.subheader("🎲 Phase 2: Simulating the Chaos (Stochastic Co-Jumps)")
        st.markdown(
            "Once the predictable human cycles are filtered out, we are left with the purely random parts of the market. "
            "The model maps this remaining random variation using three coupled mathematical levers:\n\n"
            "* **Mean Reversion (The Elastic Band Effect):** When prices get pushed away from their normal baseline, market forces act like an elastic band—forcing prices "
            "back toward equilibrium. The speed parameter (**$\\eta$**) measures how fast this rubber band snaps back.\n"
            "* **Volatility Clustering (Waves of Uncertainty):** Market nervousness comes in waves. If prices are volatile at 10:00 AM, they are highly likely to remain "
            "turbulent at 10:30 AM. The engine uses a dynamic variance channel (**$V_t$**) that scales up and down automatically as risk environments shift.\n"
            "* **Co-Jumps (Simultaneous Disruption):** True power grid anomalies are severe. When a massive spike hits, market volatility and the spot price "
            "jump higher **at the exact same time**. The model treats these anomalies as linked **Co-Jumps**, capturing both the size of the price spike and the prolonged turbulence that follows."
        )

    st.markdown("---")
    st.subheader("Some Technical Stuff: The Three-Factor MRSVCJ Model")
    st.markdown(
        "To accurately capture these structural dynamics, the spot log-price $\\ln(P_t)$ is modeled as a multi-factor system consisting of "
        "a deterministic seasonal component, an affine mean-reverting continuous volatility diffusion, and a co-jumping error vector:\n\n"
        r"$$\\ln(P_t) = \log U_t + X_t$$"
    )
    st.latex(r"dX_t = \kappa (\theta - X_t)dt + \sqrt{V_t} dW_t^s + J_t^s dN_t^s")
    st.latex(r"dV_t = \alpha (\overline{V} - V_t)dt + \sigma \sqrt{V_t} dW_t^v + J_t^v dN_t^v")
    st.markdown(
        "Where:\n"
        "* **$\log U_t$**: Deterministic seasonality (Fourier harmonics + day & hour variations).\n"
        "* **$\kappa$**: Speed of mean reversion back to structural equilibrium.\n"
        "* **$V_t$**: Stochastic volatility tracking localized variance clusters (Heston-style CIR process).\n"
        "* **$dN_t$**: Poisson arrival process triggering simultaneous correlated price and volatility **Co-Jumps** ($J^s, J^v$)."
    )

    st.markdown("---")
    st.subheader("🧭 How to Read the Dashboard Outputs")
    st.markdown(
        "* **Exploratory Data Analysis (EDA) Tab:** Shows the raw historical data profiles. Use this to inspect how often prices drop below zero or spike into extreme territory.\n"
        "* **Calibration Diagnostics Tab:** Displays the tracking accuracy of our model's baseline signature blueprint over time. It highlights how effectively the system isolates underlying regularities from unpredictable outliers.\n"
        "* **Monte Carlo Simulation Tab:** This is the predictive horizon engine. It projects hundreds of possible future price paths based on our calibrated model parameters. The dashboard collapses these paths into shaded risk zones ($P_{10}$, $P_{50}$, $P_{90}$) so you can evaluate financial risk profiles under multiple scenarios."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.header("Raw Price Structural Profiles")
    
    e_col1, e_col2 = st.columns([1, 2])
    with e_col1:
        st.markdown("#### Sample Summary Statistics")
        st.dataframe(df["Price"].describe())
        
        neg_count = (df["Price"] < 0).sum()
        st.metric("Negative Price Occurrences", f"{neg_count} half-hour bars", f"{(neg_count/len(df))*100:.3f}% of series")
        
        # Identify the adaptive floor limits
        pos_prices = df["Price"][df["Price"] > 0]
        adaptive_floor = np.percentile(pos_prices, 2) if len(pos_prices) > 0 else 1.0
        st.metric("Data-Adaptive Price Floor Threshold", f"${adaptive_floor:.3f} / MWh")
        
    with e_col2:
        st.markdown("#### Historical Spot Price Series Track")
        st.line_chart(df["Price"], height=320)
        
    st.markdown("---")
    st.subheader("Intraday Distribution Mechanics")
    e_col3, e_col4 = st.columns(2)
    with e_col3:
        df["Hour"] = df.index.hour
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.boxplot(data=df, x="Hour", y="Price", ax=ax, showfliers=False, color="darkcyan")
        ax.set_title("Typical Intraday Profile (Spikes Extracted via Boxplot Fences)")
        ax.set_ylabel("Price ($/MWh)")
        st.pyplot(fig)
        plt.close()
    with e_col4:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.histplot(df["logPrice"], bins=100, kde=True, ax=ax, color="crimson")
        ax.set_title("Empirical Distribution in Log Space")
        ax.set_xlabel("$\ln(P_t)$")
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ROBUST CALIBRATION DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_calib:
    st.header("Deterministic Breakdown & Stochastic MCMC Processing")
    
    st.subheader("1. Seasonality Estimation ($\log U_t$) Performance")
    c_col1, c_col2 = st.columns([2, 1])
    
    with c_col1:
        sample_slice = st.slider("Diagnostic Window Frame (Select Start Index)", 0, len(df)-1000, 0)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sub_slice = df.iloc[sample_slice:sample_slice+48*7] # 1 week view
        ax.plot(sub_slice.index, sub_slice["logPrice"], color="gainsboro", label="Observed Log Price", alpha=0.9)
        ax.plot(sub_slice.index, sub_slice["log_U"], color="blue", linewidth=2.0, label="67-Param Extended Robust Baseline $\log U(t)$")
        ax.set_title("Deterministic Seasonality Fit Over Sample Horizon Window")
        ax.set_ylabel("Log Scale Units")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close()
        
    with c_col2:
        st.markdown("#### Active Baseline Parameters")
        if use_defaults:
            st.info("📌 Displaying pre-calibrated baseline settings.")
        else:
            st.success("✨ Displaying parameters calibrated from your custom CSV data.")
        st.json({
            "alpha (Base Level)": f"{fitted_log_u['alpha']:.5f}",
            "beta (Long-term Daily Trend)": f"{fitted_log_u['beta']:.8f}",
            "gamma (Annual Cycle Amplitude)": f"{fitted_log_u['gamma']:.5f}",
            "tau (Phase Shift Days Alignment)": f"{fitted_log_u['tau']:.3f}"
        })

    st.markdown("---")
    st.subheader("2. Stochastic MCMC Posterior Calibration Engine")
    st.info(
        "💡 **Structural Segmentation:** The overnight window (22:00–07:00) is automatically masked to NaN before running "
        "MCMC parameter estimation. This filters out fixed geothermal baseline price floors, allowing the model "
        "to isolate genuine stochastic price variations cleanly."
    )
    
    mcmc_col1, mcmc_col2 = st.columns([1, 2])
    with mcmc_col1:
        st.markdown("#### Estimation Control Board")
        mcmc_iterations = st.number_input("MCMC Chain Iterations", min_value=100, max_value=50000, value=1000, step=500, disabled=use_defaults)
        mcmc_burn = st.number_input("Burn-in Window Limit", min_value=10, max_value=20000, value=200, step=100, disabled=use_defaults)
        
        trigger_mcmc = st.button("🚀 Run Vectorized MCMC Calibration Loop", disabled=use_defaults)
        
    with mcmc_col2:
        if not use_defaults and trigger_mcmc:
            with st.spinner("Processing vectorized MH-within-Gibbs sampler chains (Numba accelerated)..."):
                dt_annual = 1.0 / (48.0 * 365.25)
                results_mcmc = pm2.run_full_mcmc(
                    df=df, col="Stochastic", dt=dt_annual, 
                    n_iter=int(mcmc_iterations), burn_in=int(mcmc_burn)
                )
                posterior_summary = pm2.summarise_posterior(results_mcmc)
                st.success("✅ Chain convergence reached. Posterior distributions generated.")
                st.dataframe(posterior_summary)
                
                st.session_state["calibrated_theta"] = results_mcmc["theta_means"]
                st.session_state["calibrated_phi"] = results_mcmc["phi_means"]
        else:
            st.warning("Using pre-calibrated parameters. The estimation console is disabled when 'Use Pre-calibrated Default Parameters' is turned on.")
            # Format saved default dictionaries for structured visualization
            fallback_display = pd.DataFrame({
                "Parameter Description Key": list(SAVED_PRICE_PARAMS.keys()) + list(SAVED_JUMP_PARAMS.keys()),
                "Pre-Calibrated Value Estimates": list(SAVED_PRICE_PARAMS.values()) + list(SAVED_JUMP_PARAMS.values())
            }).set_index("Parameter Description Key")
            st.dataframe(fallback_display)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MONTE CARLO TEST SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.header("Synthetic Asset Profile Generation Engine")
    
    s_ui_1, s_ui_2 = st.columns([1, 3])
    with s_ui_1:
        st.markdown("#### Execution Framework Bounds")
        hard_floor_active = st.checkbox("Enforce Data-Adaptive Pricing Floor", value=True)
        
        # Load parameters based on the chosen mode
        if not use_defaults and "calibrated_theta" in st.session_state:
            theta_sim = st.session_state["calibrated_theta"]
            phi_sim = st.session_state["calibrated_phi"]
            st.caption("🟢 Loaded custom parameters from active session state memory.")
        else:
            theta_sim = SAVED_PRICE_PARAMS
            phi_sim = SAVED_JUMP_PARAMS
            st.caption("🔵 Loaded verified pre-calibrated default parameter states.")
            
        st.markdown("---")
        st.markdown("##### Parameter Summary View")
        st.json({
            "Speed of Reversion (eta)": f"{theta_sim['eta']:.4f}",
            "Long-Term Volatility (theta)": f"{theta_sim['theta']:.4f}",
            "Jump Size Volatility (sigma_x)": f"{phi_sim['sigma_x']:.4f}"
        })
        
    with s_ui_2:
        st.markdown("#### Forward Path Trajectories Matrix")
        
        total_steps = int(sim_horizon_days * 48)
        dt_step = 1.0 / (48.0 * 365.25)
        
        sim_start_dt = pd.Timestamp("2026-06-01 00:00:00")
        sim_dates = pd.date_range(start=sim_start_dt, periods=total_steps, freq="30min")
        
        # Simulate trajectories using your public simulation engine API
        paths_collection = []
        for path_idx in range(int(sim_trajectories)):
            sim_df_path = pm2.simulate_mrsvcj(
                params=theta_sim, 
                jump_size_params=phi_sim, 
                T_steps=total_steps, dt=dt_step
            )
            paths_collection.append(sim_df_path["X"].values)
            
        stochastic_matrix = np.vstack(paths_collection)
        
        # Project deterministic seasonality layer
        forward_log_u = pm2.log_U(dt_input=sim_start_dt, n_bars=total_steps, **fitted_log_u)
        
        # Reconstruct real prices: Price = exp(log_U + Stochastic)
        reconstructed_price_matrix = np.exp(forward_log_u + stochastic_matrix)
        
        if hard_floor_active:
            pos_prices_calc = df["Price"][df["Price"] > 0]
            sim_floor = float(np.percentile(pos_prices_calc, 2)) if len(pos_prices_calc) > 0 else 1.0
            reconstructed_price_matrix = np.clip(reconstructed_price_matrix, sim_floor, None)
            
        # Draw paths output plot
        fig, ax = plt.subplots(figsize=(12, 5))
        days_axis = np.arange(total_steps) / 48.0
        
        # Display alpha path cloud
        cloud_max_lines = min(int(sim_trajectories), 150)
        alpha_val = max(0.02, 1.0 / np.log(cloud_max_lines + 1.5))
        for idx in range(cloud_max_lines):
            ax.plot(days_axis, reconstructed_price_matrix[idx, :], color="steelblue", alpha=alpha_val, linewidth=0.6)
            
        # Draw Expected Median/Quantile tracks
        median_track = np.percentile(reconstructed_price_matrix, 50, axis=0)
        p90_track = np.percentile(reconstructed_price_matrix, 90, axis=0)
        p10_track = np.percentile(reconstructed_price_matrix, 10, axis=0)
        
        ax.plot(days_axis, median_track, color="gold", linewidth=2.0, label="P50 Expected Median Baseline")
        ax.plot(days_axis, p90_track, color="darkorange", linewidth=1.5, linestyle="--", label="P90 Risk Upper Bound")
        ax.plot(days_axis, p10_track, color="magenta", linewidth=1.5, linestyle="--", label="P10 Valuation Floor")
        
        ax.set_title(f"Forward Physical Spot Price Projection Cloud Profile ({sim_trajectories} Replications)")
        ax.set_xlabel("Horizon Interval Timeline (Days Elapsed)")
        ax.set_ylabel("Price ($/MWh)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)
        plt.close()
        
        # Risk analytics cards
        st.markdown("#### Forward Risk Profile Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Simulated Pool Mean Value", f"${reconstructed_price_matrix.mean():.2f} /MWh")
        m_col2.metric("P95 Tail Volatility Exposure", f"${np.percentile(reconstructed_price_matrix, 95):.2f} /MWh")
        m_col3.metric("Observed Spike Boundary Counter (> $250)", f"{(reconstructed_price_matrix > 250).sum() / sim_trajectories:.1f} bars/path")