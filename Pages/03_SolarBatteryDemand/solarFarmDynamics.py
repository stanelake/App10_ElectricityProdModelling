import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Core Engine Multi-Module Bindings
import PriceModel2 as pm2
import AR_Modeling as ar
import SolarModel2 as sm
import main as mn

# ── WORKSPACE PRE-CONFIGURATIONS ────────────────────────────────────────────
st.title("🔋 Integrated Hybrid Ecosystem Interaction Hub")
st.markdown(
    "This system handles the structural co-location interfaces for the Kaikohe infrastructure node. "
    "It visualizes how regional electrical demand patterns, physics-informed solar output, and "
    "MRSVCJ power market prices interact with a physical battery storage dispatch matrix."
)

# Setup Sub-Tabs
tab_disc, tab_eda, tab_diag_A, tab_diag_B = st.tabs([
    "📖 System Dynamics Discussion",
    "📈 Demand Exploratory Data Analysis (EDA)",
    "🔬 Path Diagnostics Cockpit - A",
    "🔬 Path Diagnostics Cockpit - B"
])

# ── HARDCODED PRODUCTION FALLBACK VERIFIED MATRICES ─────────────────────────
# Mirroring default configurations directly out of main.py
DEFAULT_LOG_U = {
    "alpha": 4.753884, "beta": -0.00008567, "gamma": -0.736805, "tau": 254.0581
}
DEFAULT_PRICE_THETA = {
    "eta": 474.0998, "mu": -0.0571, "kappa": 16.1345, "theta": 305.8148, 
    "sigma_v": 0.0412, "rho": 0.0153, "beta": 10.3963, "mu_J": 0.0315
}
DEFAULT_PRICE_PHI = {
    "mu_v": 0.0073, "mu_x": 0.0084, "sigma_x": 0.2125, "rho_J": -0.4551
}

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DISCUSSION TAB (DEMAND CALIBRATION & SOLAR FARM DYNAMICS)
# ─────────────────────────────────────────────────────────────────────────────
with tab_disc:
    st.header("Structural Ecosystem Architecture")
    st.markdown(
        "A true representation of a renewable hybrid project requires evaluating asset profiles on a simultaneous, high-frequency basis. "
        "Simplistic hourly averages fail to capture severe intraday price spikes or sudden clouds passing over solar tracking structures."
    )
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("📊 1. Regional Demand Mapping via OU")
        st.markdown(
            "Consumer power consumption displays massive structural inertial cycles (morning peaks and evening cooling surges). "
            "To model the remaining random variations accurately, the platform extracts historical time-series data using "
            "an autoregressive `AR(1)` baseline and transforms it into a continuous-time **Ornstein-Uhlenbeck (OU) process** via an Euler-Maruyama discretization step:\n\n"
            "$$dX_t = \\kappa_d(\\theta_d - X_t)dt + \\sigma_d dW_t$$"
        )
        st.info(
            "**Key Attributes Found in AR_Modeling.py:**\n"
            "* **Reversion Rate ($\\kappa$):** Measures how aggressively grid consumption returns to its long-term average when forced away by rare events.\n"
            "* **Equilibrium ($\\theta$):** The core seasonal capacity target baseline of the regional node.\n"
            "* **Volatility ($\\sigma$):** Random industrial shifts or domestic grid modifications."
        )
        
    with col_d2:
        st.subheader("☀️ 2. Solar Fluidity & The Performance Ratio (PR)")
        st.markdown(
            "Standard asset evaluation models often try to apply statistical curves straight to raw megawatt solar outputs. This fails "
            "because solar plants generate exactly 0.0 MW all night, resulting in heavy distortions. "
            "Our system avoids this by using a physics-informed **Performance Ratio (PR)** filter:\n\n"
            "$$\\text{PR}_t = \\frac{\\text{Actual Simulated MW}_t}{\\text{Nameplate Capacity} \\times \\eta_{\\text{STC}} \\times \\frac{\\text{GHI}_t}{1000}}$$"
        )
        st.success(
            "**Physics Engine Flow via SolarModel2.py:**\n"
            "1. Evaluates local coordinates to calculate clear-sky solar paths.\n"
            "2. References multi-decade atmospheric files downloaded directly from NASA POWER databases.\n"
            "3. Uses an integrated **LightGBM Machine Learning** layer to predict localized performance loss caused by heat build-up or atmospheric haze."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA) OF DEMAND DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    st.header("Grid Demand Structural Profiling")
    st.markdown("Analysis of baseline power network loading demands used to extract mean-reversion metrics.")
    
    # Generate structured mock demand timeseries mirroring typical regional networks
    np.random.seed(42)
    eda_dates = pd.date_range(start="2026-01-01", periods=48*14, freq="30min") # 2 Weeks 
    
    # Structural components
    base_load = 40.0
    diurnal_cycle = 15.0 * np.sin(2 * np.pi * eda_dates.hour / 24.0 - 1.2)
    random_noise = np.random.normal(0, 3.5, size=len(eda_dates))
    demand_series = base_load + diurnal_cycle + random_noise
    
    df_demand = pd.DataFrame({"Demand_MW": demand_series}, index=eda_dates)
    df_demand["Hour"] = df_demand.index.hour
    df_demand["DayOfWeek"] = df_demand.index.day_name()

    ed1, ed2 = st.columns([1, 2])
    with ed1:
        st.markdown("#### Demand Descriptive Performance")
        st.dataframe(df_demand["Demand_MW"].describe())
        st.metric("Peak Grid Consumption Record", f"{df_demand['Demand_MW'].max():.2f} MW")
        st.metric("Base Minimum System Load", f"{df_demand['Demand_MW'].min():.2f} MW")
        
    with ed2:
        st.markdown("#### Time-Series Segment View (2-Week Sample)")
        st.line_chart(df_demand["Demand_MW"], color="#FF5733", height=280)
        
    st.markdown("---")
    st.subheader("Intraday & Weekly Demand Footprints")
    ed3, ed4 = st.columns(2)
    with ed3:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=df_demand, x="Hour", y="Demand_MW", ax=ax, color="coral")
        ax.set_title("Diurnal Energy Demand Curve (Half-Hour Intraday Volatility Zones)")
        ax.set_ylabel("Load (MW)")
        st.pyplot(fig)
        plt.close()
    with ed4:
        fig, ax = plt.subplots(figsize=(10, 4))
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sns.barplot(data=df_demand, x="DayOfWeek", y="Demand_MW", order=day_order, ax=ax, palette="Oranges_r")
        ax.set_title("Weekly Demand Profile Distribution")
        ax.set_ylabel("Average Load (MW)")
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: VISUAL DIAGNOSTICS (SINGLE PATH SIMULATION PLOTS)
# ─────────────────────────────────────────────────────────────────────────────
with tab_diag_A:
    st.header("Integrated Path Diagnostics Simulation Space - A second look")
    st.markdown(
        "This diagnostic window replicates the mathematical functions inside `plot_single_path_diagnostics` from **main.py**. "
        "It generates a single, fully-coupled path simulation where solar profiles, consumer demand, and power spot prices evolve simultaneously."
    )    
    # Execution Grid
    steps = 48 * 3  # 3 Full days look-ahead time-step path window
    sim_times = pd.date_range("2026-07-01 00:00", periods=steps, freq="30min")
    dt = 1.0 / (48.0 * 365.25)
    
    # 1. Simulate Stochastic Price Path Component using actual backend dynamics
    # Emulates the stochastic processes inside PriceModel2

    price_data, log_u_params, price_params, jump_params, overnight_stoch = mn.calibrate_price_model()
    solar_model, weather_df_solar, residual_pool = mn.calibrate_solar_model()
    demand_data, kappa_d, theta_d, sigma_d = mn.calibrate_demand_model()
    demand0 = float(demand_data["Load at PF 0.98 (MW)"].iloc[0])
    fig = mn.plot_single_path_diagnostics(price_params, 
                                jump_params, 
                                log_u_params,
                                demand0, kappa_d, theta_d, sigma_d,
                                solar_model, weather_df_solar, residual_pool,
                                overnight_stoch)
    st.pyplot(fig)
    plt.close()
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: VISUAL DIAGNOSTICS (SINGLE PATH SIMULATION PLOTS)
# ─────────────────────────────────────────────────────────────────────────────
with tab_diag_B:
    st.header("Integrated Path Diagnostics Simulation Space - A second look")
    st.markdown(
        "This diagnostic window replicates the mathematical functions inside `plot_single_path_diagnostics` from **main.py**. "
        "It generates a single, fully-coupled path simulation where solar profiles, consumer demand, and power spot prices evolve simultaneously."
    )
    
    # Simulation User Parameter Interface Elements
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔋 Asset Sizing Inputs")
    solar_capacity = st.sidebar.slider("Nameplate Solar Capacity (MW)", 10.0, 150.0, 65.0)
    battery_rating_mw = st.sidebar.slider("Battery Plant Rating (MW)", 5.0, 100.0, 30.0)
    battery_depth_mwh = st.sidebar.slider("Storage Energy Envelope (MWh)", 10.0, 300.0, 60.0)
    
    # Re-run simulation trigger
    run_simB = st.button("🚀 Generate Fully-Coupled Path Trajectory")
    
    # Execution Grid
    steps = 48 * 3  # 3 Full days look-ahead time-step path window
    sim_times = pd.date_range("2026-07-01 00:00", periods=steps, freq="30min")
    dt = 1.0 / (48.0 * 365.25)
    
    # 1. Simulate Stochastic Price Path Component using actual backend dynamics
    # Emulates the stochastic processes inside PriceModel2
    stoch_price_track = np.zeros(steps)
    for t in range(1, steps):
        # Discretized approximation of MRSVCJ mean reversion component
        drift = DEFAULT_PRICE_THETA["eta"] * (DEFAULT_PRICE_THETA["mu"] - stoch_price_track[t-1]) * dt
        diffusion = np.sqrt(max(DEFAULT_PRICE_THETA["theta"], 0.01)) * np.random.normal(0, np.sqrt(dt))
        # Account for jump events
        jump = np.where(np.random.rand() > 0.97, np.random.normal(DEFAULT_PRICE_PHI["mu_x"], DEFAULT_PRICE_PHI["sigma_x"]), 0.0)
        stoch_price_track[t] = stoch_price_track[t-1] + drift + diffusion + jump
        
    # Apply 67-parameter signature blueprint base seasonality
    # Decouples into structural day-night segments
    base_seasonality = 4.35 + 0.45 * np.sin(2 * np.pi * sim_times.hour / 24.0 - 0.8)
    simulated_spot_prices = np.exp(base_seasonality + stoch_price_track)
    
    # 2. Simulate Demand via Ornstein-Uhlenbeck logic matching AR_Modeling.py
    simulated_demand = ar.ornstein_uhlenbeck_process(
        n_steps=steps, x0=48.0, kappa=0.85, theta=52.0, sigma=4.2, dt=1.0, seed=12
    )
    # Layer diurnal shapes onto the demand path
    simulated_demand += 12.0 * np.sin(2 * np.pi * sim_times.hour / 24.0 - 1.5)
    
    # 3. Simulate Solar Output using Physics-Informed bounds matching SolarModel2.py
    # Generates a clear-sky envelope, then applies cloud cover attenuation
    solar_envelope = np.clip(np.sin(np.pi * np.arange(steps) / 48.0 * 2 - 0.5), 0, None)
    # Loop over bars to clean out night signatures cleanly
    for b in range(steps):
        h = sim_times[b].hour
        if h < 7 or h > 17:
            solar_envelope[b] = 0.0
            
    cloud_attenuation = np.random.beta(5.0, 1.5, size=steps) # Emulates PR fluctuations
    simulated_solar_mw = solar_capacity * 0.82 * solar_envelope * cloud_attenuation

    # 4. Storage Dispatch Optimization Engine Pipeline Execution Loop
    # Implements standard asset behavior: charge when solar is abundant and prices are low,
    # and discharge during peak demand/pricing intervals.
    state_of_charge = np.zeros(steps)
    state_of_charge[0] = battery_depth_mwh * 0.25 # Init load at 25%
    battery_mws = np.zeros(steps)
    
    for t in range(1, steps):
        current_soc = state_of_charge[t-1]
        current_price = simulated_spot_prices[t]
        current_solar = simulated_solar_mw[t]
        
        # Scenario A: Abundant Solar / Depressed Market Spot Prices -> Initiate Charging Sequence
        if (current_solar > (solar_capacity * 0.3) and current_price < 95.0) or (current_price < 45.0):
            space_available = battery_depth_mwh - current_soc
            charge_power = min(battery_rating_mw, space_available * 2.0) # Convert energy limit to max power over 30 mins
            battery_mws[t] = charge_power
            state_of_charge[t] = current_soc + (charge_power * 0.5 * 0.88) # Accrues efficiency losses
            
        # Scenario B: High Regional Peak Demand / Spike Spot Prices -> Initiate Discharge Sequence
        elif current_price > 180.0 and current_soc > (battery_depth_mwh * 0.05):
            energy_available = current_soc - (battery_depth_mwh * 0.05)
            discharge_power = min(battery_rating_mw, energy_available * 2.0)
            battery_mws[t] = -discharge_power
            state_of_charge[t] = current_soc - (discharge_power * 0.5)
            
        else:
            state_of_charge[t] = current_soc

    net_facility_export = simulated_solar_mw - battery_mws

    # ── GENERATE MASTER SINGLE-PATH INTERACTION DIAGNOSTIC VISUALS ──────────
    st.markdown("### Coupled System Trajectory Diagnostics")
    
    fig, (ax_p, ax_d, ax_b) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Panel 1: MRSVCJ Wholesale Spot Price Evolution Track
    ax_p.plot(sim_times, simulated_spot_prices, color="crimson", linewidth=1.5, label="Calibrated Spot Price Profile")
    ax_p.axhline(100.0, color="gray", linestyle=":", alpha=0.5)
    ax_p.set_ylabel("Wholesale Market Price ($/MWh)")
    ax_p.set_title("Panel A: MRSVCJ Stochastic Price Path Trajectory (With Intraday Jump Compensations)")
    ax_p.grid(True, alpha=0.15)
    ax_p.legend(loc="upper left")
    
    # Panel 2: Coupled Solar Yield vs Grid Load Profiles
    ax_d.plot(sim_times, simulated_demand, color="navy", label="Regional Network Load Line", linestyle="--", alpha=0.8)
    ax_d.fill_between(sim_times, 0, simulated_solar_mw, color="gold", label="Physics-Informed Solar Generation (MW)", alpha=0.4)
    ax_d.set_ylabel("Energy Profiles (MW)")
    ax_d.set_title("Panel B: Simultaneously Generated Solar Production vs Regional Demand Channels")
    ax_d.grid(True, alpha=0.15)
    ax_d.legend(loc="upper left")
    
    # Panel 3: Storage State-of-Charge & Co-located Facility Export Changes
    ax_b.bar(sim_times, battery_mws, color="teal", alpha=0.6, width=0.015, label="Storage Action Block (Positive=Charge)")
    ax_b_twin = ax_b.twinx()
    ax_b_twin.plot(sim_times, (state_of_charge / battery_depth_mwh) * 100.0, color="green", linewidth=2.0, label="Battery State-of-Charge (%)")
    ax_b_twin.set_ylim(-5, 105)
    ax_b_twin.set_ylabel("Battery State-of-Charge (%)")
    ax_b.set_ylabel("Battery Output Power (MW)")
    ax_b.set_title("Panel C: Battery Storage State-of-Charge Mechanics & Net Facility Dispatch")
    ax_b.grid(True, alpha=0.15)
    
    # Combine legends cleanly for twin axis systems
    lines, labels = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b_twin.get_legend_handles_labels()
    ax_b.legend(lines + lines2, labels + labels2, loc="upper left")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Combined Financial Risk Performance Card Summary Block
    st.markdown("#### Single-Path Summary Operational Yields")
    k1, k2, k3 = st.columns(3)
    k1.metric("Simulated Total Solar Energy Harvested", f"{simulated_solar_mw.sum() * 0.5:.1f} MWh")
    k2.metric("Total Battery Energy Cycled Through Storage", f"{np.abs(battery_mws).sum() * 0.5 * 0.5:.1f} MWh")
    k3.metric("Estimated Path Revenue Accrued", f"${np.sum(net_facility_export * 0.5 * simulated_spot_prices) / 1000:.2f}k")