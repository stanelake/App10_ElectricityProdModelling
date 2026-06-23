import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Module hooks to the core quantitative models
import PriceModel2 as pm2
import AR_Modeling as ar
import main as mn

# ── STREAMLIT INITIAL WORKSPACE CONFIGURATION ───────────────────────────────
st.title("🎲 High-Dimensional Monte Carlo Project Valuation Engine")
st.markdown(
    "Welcome to the investment underwriting room for the Kaikohe Hybrid Energy Asset. "
    "As the final capstone module, this engine coordinates simultaneous forward trajectories "
    "of physics-informed solar output, regional demand load, and wholesale electricity spot prices "
    "to simulate long-term asset distributions, capital risk horizons, and project Net Present Value (NPV)."
)

tab_underwrite, tab_risk_eda, tab_monte_carloA = st.tabs([
    "📖 Underwriting Methodology",
    "📊 Pre-Simulation Asset Vectors",
    "🔮 Monte Carlo Engine, Valuation & Financial Tail Risk",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: UNDERWRITING METHODOLOGY (EXHAUSTIVE & ACCESSIBLE)
# ─────────────────────────────────────────────────────────────────────────────
with tab_underwrite:
    st.header("Financial Valuation Framework")
    st.markdown(
        "Standard renewable energy investments are typically underwriting using static spreadsheets that assume "
        "average solar production and fixed power price forecasts. In real-world merchant markets, that approach "
        "hides critical vulnerabilities. True financial risk is found in the **co-variance** of the ecosystem: "
        "when solar generation peaks across a region, wholesale market prices often crash because of oversupply "
        "(the 'Solar Cannibalization' effect). Conversely, extreme pricing spikes usually occur when system demand "
        "is soaring and renewable output is low."
    )
    
    st.markdown("---")
    u_col1, u_col2 = st.columns(2)
    with u_col1:
        st.subheader("🏁 The Capstone Architecture")
        st.markdown(
            "This engine replaces static forecasts with a high-dimensional financial simulator. It evaluates "
            "thousands of fully-realized futures, where every single day of the **20-year project lifetime** is simulated "
            "at 30-minute intervals (48 settlement periods per day) to accurately assess financial risk metrics:\n\n"
            "1. **Stochastic Volatility with Co-Jumps (MRSVCJ):** Captures volatile, mean-reverting electricity pricing "
            "dynamics, tracking how random power grid shocks drive spot prices and market volatility up at the exact same moment.\n"
            "2. **Ornstein-Uhlenbeck (OU) Macro Grid Demand:** Simulates regional power network loading profiles, providing "
            "the baseline economic environment that shapes local storage optimization and grid interaction pathways.\n"
            "3. **Physics-Driven Solar Output:** Generates clear-sky geometry profiles mapped against decades of historical weather logs, "
            "accounting for structural cloud attenuation trends."
        )
        
    with u_col2:
        st.subheader("💵 Financial Translation Keys")
        st.markdown(
            "For every simulated path, the engine feeds operational power metrics directly into the project's financial ledgers:\n\n"
            "* **The Dispatch Ledger:** Revenue is calculated at each half-hour block: $$\\text{Revenue}_t = (\\text{Solar}_t - \\text{Battery Discharge}_t + \\text{Battery Charge}_t) \\times \\text{Spot Price}_t$$\n"
            "* **Physical Asset Decay:** Solar cells experience standard efficiency degradation over time, and battery cells lose "
            "storage capacity with every energy cycle. The model factors in a **0.6% annual solar degradation curve** and track cycle-based storage wear.\n"
            "* **Net Present Value (NPV):** Future net cash flows are discounted back to today's terms using the project's **Weighted Average Cost of Capital (WACC)**. "
            "An investment delivers positive financial returns if its total discounted cash flows exceed the initial capital expenditure (CAPEX)."
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: PRE-SIMULATION ASSET VECTORS (EXPLORATORY RISK MATRIX)
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk_eda:
    st.header("Pre-Simulation Boundary Analysis")
    st.markdown(
        "Before running the full-scale simulation, this section analyzes our baseline data vectors. "
        "Use these metrics to audit the underlying market trends and core cost curves before executing the valuation engine."
    )
    
    # Financial Inputs Panel
    st.subheader("💰 Capital Asset Cost Curves")
    eda_c1, eda_c2, eda_c3 = st.columns(3)
    solar_cost_mw = eda_c1.number_input("Solar Build Cost ($k / MW DC)", value=1200)
    battery_cost_mwh = eda_c2.number_input("Battery Storage Build Cost ($k / MWh)", value=350)
    fixed_opex_year = eda_c3.number_input("Fixed Asset O&M Operations Cost ($k / Year)", value=1800)
    
    # Calculate initial investment thresholds based on plant size inputs
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Capstone Facility Layout")
    p_solar_capacity = st.sidebar.slider("Asset Solar Sizing (MW)", 10.0, 200.0, 65.0)
    p_battery_mw = st.sidebar.slider("Storage Inverter Power (MW)", 5.0, 150.0, 30.0)
    p_battery_mwh = st.sidebar.slider("Storage Energy Capacity (MWh)", 10.0, 400.0, 60.0)
    
    calculated_capex = ((p_solar_capacity * solar_cost_mw) + (p_battery_mwh * battery_cost_mwh)) / 1000.0
    st.metric("Total Calculated Project Initial CAPEX Budget", f"${calculated_capex:.2f} Million", help="Derived dynamically from your system cost parameters.")
    
    st.markdown("---")
    st.subheader("📈 Long-Term Technical Decay Profiles")
    
    # Project 20-Year degradation trends
    years_horizon = np.arange(1, 21)
    solar_efficiency = (1.0 - 0.005) ** years_horizon
    battery_capacity_retention = 1.0 - (0.018 * years_horizon) # Linear proxy for cycle wear
    
    df_decay = pd.DataFrame({
        "Year": years_horizon,
        "Solar Panel Yield Performance (%)": solar_efficiency * 100.0,
        "Battery Storage Volume Retention (%)": battery_capacity_retention * 100.0
    }).set_index("Year")
    
    fig_dec, ax_dec = plt.subplots(figsize=(11, 3.5))
    ax_dec.plot(df_decay.index, df_decay["Solar Panel Yield Performance (%)"], color="gold", linewidth=2, label="Solar Performance Ratio Degradation (-0.6%/yr)")
    ax_dec.plot(df_decay.index, df_decay["Battery Storage Volume Retention (%)"], color="teal", linewidth=2, label="Battery Cell Capacity Decay Baseline (-1.8%/yr)")
    ax_dec.set_ylabel("Asset Operational Health (%)")
    ax_dec.set_xlabel("Project Operational Lifecycle Timeline (Years)")
    ax_dec.set_xticks(years_horizon)
    ax_dec.set_ylim(60, 105)
    ax_dec.grid(True, alpha=0.15)
    ax_dec.legend(loc="lower left")
    st.pyplot(fig_dec)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MONTE CARLO ENGINE & FINANCIAL TAIL RISK (CAPSTONE VALUATION)
# ─────────────────────────────────────────────────────────────────────────────

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
    
# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MONTE CARLO ENGINE & PROJECT VALUATION (CAPSTONE VALUATION)
# ─────────────────────────────────────────────────────────────────────────────

with tab_monte_carloA:  
    st.header("Full-Scale Asset Lifecycle Portfolio Engine")
    st.markdown(
        "Click the button below to trigger the high-dimensional portfolio underwriting simulation loop. "
        "The model projects complete 20-year operational horizons across hundreds of independent paths, "
        "interlocking your asset parameters with random market variables."
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📉 Corporate Discount Hurdles")
    wacc_rate = st.sidebar.slider("Weighted Average Cost of Capital (WACC %)", 4.0, 16.0, 8.5, step=0.5) / 100.0
    mc_iterations_count = st.sidebar.number_input("Total Valuation Paths to Simulate", value=250, min_value=25, max_value=2000, step=25)
    
    execute_valuation = st.button("🚀 Execute Joint Portfolio Monte Carlo Underwriting Engine")
    
    st.header("Project NPV Simulation")  
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
    summary1, fig1, metrics2, fig2 = mn.run_monte_carlo(
        price_params, jump_params, log_u_params,
        demand0, kappa_d, theta_d, sigma_d,
        solar_model, weather_df_solar, residual_pool,
        overnight_stoch=overnight_stoch,
    )

    # ── Stage 8: results ─────────────────────────────────────────────────
    st.pyplot(fig1)
    plt.close()
    
    st.subheader("Final NPV Statistics")
    st.dataframe(summary1)

    st.pyplot(fig2)
    plt.close()
    
    st.subheader("Final NPV Statistics")
    st.dataframe(metrics2)


