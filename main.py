"""
main.py  –  Solar + Battery Project Valuation Pipeline
=======================================================

Pipeline stages
---------------
1.  Load & clean wholesale price data            (PriceModel2)
2.  Fit deterministic seasonal component log_U   (Huber MLR)
3.  Calibrate SV-Jump model on stochastic residual (MCMC)
4.  Load & calibrate demand OU process           (AR_Modeling)
5.  Load & calibrate solar OU process            (AR_Modeling)
6.  Monte Carlo: simulate N revenue paths        (MonteCarloSimulation)
7.  Discount cash-flows → NPV distribution
8.  Plot results + confidence intervals

Usage
-----
    Set is_calibrated = False for a full run.
    Set is_calibrated = True  to skip calibration and use hardcoded params.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from concurrent.futures import ProcessPoolExecutor, as_completed

import AR_Modeling          as ar
import SolarModel2           as sm

# ── Sanity check: confirm fixed AR_Modeling is loaded ────────────────────────
def _check_ar_modeling():
    """Warn if the unfixed AR_Modeling (theta=phi/(1-phi)) is in use."""
    import inspect
    src = inspect.getsource(ar.OU_parameters)
    if "phi/(1-phi)" in src or "phi / (1 - phi)" in src:
        import warnings
        warnings.warn(
            "\n*** WRONG AR_Modeling.py LOADED ***\n"
            "    The old theta=phi/(1-phi) bug is present.\n"
            "    Replace AR_Modeling.py with the fixed version from outputs/.\n"
            "    Demand theta will be wrong (e.g. 156 instead of 11).",
            UserWarning, stacklevel=2,
        )
    else:
        print("  AR_Modeling: fixed version confirmed (theta = c/(1-phi))")

_check_ar_modeling()
import PriceModel2          as pm2

# ─────────────────────────────────────────────────────────────────────────────
# USER-FACING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Set False to run full calibration; True to load hardcoded params
IS_CALIBRATED = True

# ---- Data paths -------------------------------------------------------------
PRICE_CSV      = "DataSource/Wholesale_Prices/Wholesale_price_trends_20260505202423.csv"
DEMAND_XLSX    = r"DataSource\Kaikohe_Demand_Data\Kaikohe_GXP_8760_Profile-Load.xlsx"
SOLAR_XLSX     = r"DataSource\Solar_Generation_Data\Results from Simulations.xlsx"

# ---- Time step --------------------------------------------------------------
# All three processes must share a consistent dt.
# 1/(48*365.25) years  =  30-minute bar on a continuous 24/7 market
DT_YEARS       = 1 / (48 * 365.25)   # used for SV-Jump MCMC and price simulation
DT_HOURS       = 0.5                  # 30 min expressed in hours (for simulate_power_dummy)

# ---- MCMC settings ----------------------------------------------------------
MCMC_ITER      = 10_000
MCMC_BURNIN    =  2_000

# ---- Monte Carlo settings ---------------------------------------------------
N_SIMULATIONS  = 200          # number of independent revenue paths
SIM_DAYS       = 365          # projection horizon per path
N_WORKERS      = 4            # parallel workers (set 1 to disable parallelism)

# ---- Plant specifications ---------------------------------------------------
SOLAR_MAX_MW          = 65.0
BATTERY_CAPACITY_MWH  = 35.0
BATTERY_POWER_MAX_MW  = 20.0
ETA_CHARGE            = 0.95
ETA_DISCHARGE         = 0.95
DISCOUNT_RATE         = 0.08  # annual

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED PARAMS  (used when IS_CALIBRATED = True)
# ─────────────────────────────────────────────────────────────────────────────

SAVED_LOG_U_PARAMS = {
    "alpha":  4.794884 - 0.0410,   # recentred: absorbed median -0.0410
    "beta":  -0.00008567,
    "gamma": -0.736805,
    "tau":    254.0581,
    "phi":   np.array([0.100192, 0.159623, 0.201218,
                       0.190828, 0.163910, 0.041892]),   # Mon–Sat
    "zeta":  np.array([ 0.381820,  0.263283,  0.073771,
                       -0.249871, -0.358883, -0.463163,
                       -0.398027, -0.100464, -0.026599,
                        0.111779,  0.204746]),            # Jan–Nov
    "psi":   np.array([
        -0.002218, -0.047205, -0.067158, -0.113031, -0.172792,
        -0.328729, -0.379421, -0.421409, -0.398954, -0.326499,
        -0.100613, -0.039312,  0.031389,  0.094874,  0.190808,
         0.301810,  0.361674,  0.309555,  0.291363,  0.263849,
         0.250891,  0.233304,  0.240551,  0.233940,  0.242874,
         0.235213,  0.217599,  0.172879,  0.146981,  0.140640,
         0.178018,  0.250421,  0.262457,  0.300956,  0.300810,
         0.412941,  0.421734,  0.409162,  0.334900,  0.303174,
         0.303599,  0.289456,  0.325971,  0.260650,  0.169627,
         0.066564,  0.088458,
    ]),  # psi_01:00 … psi_23:30  (47 half-hour dummies, baseline = 00:30)
}

SAVED_PRICE_PARAMS = {
    # Posterior means from MCMC run with overnight exclusion + eta=100 prior
    # eta hits hard cap 500 — treat as: model prefers fast reversion
    "eta":     474.099821,
    "mu":       -0.057159,
    "kappa":    16.134595,
    "theta":   305.814897,
    "sigma_v":   0.041232,
    "rho":       0.015383,
    "beta":     10.396368,
    "mu_J":      0.031577,   # derived: exp(mu_x + 0.5*sigma_x²) - 1 = 0.031574
}

SAVED_JUMP_PARAMS = {
    "mu_v":    0.007311,
    "mu_x":    0.008495,
    "sigma_x": 0.212559,
    "rho_J":  -0.455157,
}

SAVED_DEMAND_PARAMS = dict(kappa=0.1687, theta=11.3658, sigma=4.3668)
# NOTE: theta=11.37 MW is correct for Kaikohe (~11 MW peak load).
# If you see theta=156, you are importing the OLD AR_Modeling.py (unfixed
# phi/(1-phi) bug).  Replace with the AR_Modeling.py from outputs folder.
SAVED_SOLAR_PARAMS  = dict(kappa=0.1098, theta=17.7227, sigma=7.3824)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1–3 :  PRICE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_price_model():
    """
    Returns (price_data, log_u_params, price_params, jump_params).
    Runs full calibration pipeline or returns saved params.
    """
    print("=" * 64)
    print("STAGE 1-3 : Price model calibration")
    print("=" * 64)

    if IS_CALIBRATED:
        print("  → Using saved parameters (IS_CALIBRATED = True).")
        price_data, _ = pm2.loadPriceData(
            csv_path=PRICE_CSV,
            log_U_params=SAVED_LOG_U_PARAMS,
        )
        overnight_stoch = price_data["OvernightStochastic"].dropna().values \
            if "OvernightStochastic" in price_data.columns else None
        return price_data, SAVED_LOG_U_PARAMS, SAVED_PRICE_PARAMS, SAVED_JUMP_PARAMS, overnight_stoch

    # ── Stage 1: load & clean ─────────────────────────────────────────────
    print("\n  [1/3] Loading price data …")
    price_data, log_u_params = pm2.loadPriceData(
        csv_path=PRICE_CSV,
        log_U_params="fit",
    )

    # Extract empirical overnight stochastic residuals for bootstrap simulation
    overnight_stoch = price_data["OvernightStochastic"].dropna().values \
        if "OvernightStochastic" in price_data.columns else None

    print(f"  Data shape: {price_data.shape}  |  "
          f"  Date range: {price_data.index[0]} → {price_data.index[-1]}")
    print(price_data[["Price", "logPrice", "log_U", "Stochastic"]].head(15))

    # ── Stage 2-3: MCMC calibration of SV-Jump model ─────────────────────
    print(f"\n  [2/3] Running MCMC ({MCMC_ITER} iterations) …")
    results = pm2.run_full_mcmc(
        price_data, col="Stochastic",
        dt=DT_YEARS,
        n_iter=MCMC_ITER, burn_in=MCMC_BURNIN,
        seed=0, verbose=True,
    )

    print("\n  [3/3] Posterior summary:")
    summary = pm2.summarise_posterior(results)
    print(summary.to_string())

    # Split posterior means into structural and jump-size dicts
    means     = summary["mean"].to_dict()
    jmp_keys  = {"mu_v", "mu_x", "sigma_x", "rho_J"}
    jump_params  = {k: means[k] for k in jmp_keys}
    price_params = {k: means[k] for k in means if k not in jmp_keys}

    return price_data, log_u_params, price_params, jump_params, overnight_stoch


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 :  DEMAND MODEL
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_demand_model():
    """Returns (demand_data, kappa, theta, sigma)."""
    print("\n" + "=" * 64)
    print("STAGE 4 : Demand model calibration")
    print("=" * 64)

    demand_data = pd.read_excel(DEMAND_XLSX)
    demand_data["Timestamp"] = pd.to_datetime(
        demand_data["Timestamp"], format="%Y-%m-%d %H:%M"
    )
    demand_col = "Load at PF 0.98 (MW)"

    if IS_CALIBRATED:
        print("  → Using saved demand OU parameters.")
        p = SAVED_DEMAND_PARAMS
        return demand_data, p["kappa"], p["theta"], p["sigma"]

    kappa, theta, sigma = ar.calibrate_ou_process(
        demand_data[demand_col], dt=DT_HOURS
    )
    return demand_data, kappa, theta, sigma


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 :  SOLAR MODEL  (physics-informed PR model)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_solar_model():
    """
    Returns (solar_model, weather_df, residual_pool).

    Replaces the OU solar process with the SolarPRModel pipeline:
      1. Download 15-yr hourly weather from NASA POWER (cached to parquet).
      2. Compute pvlib solar geometry + cloud attenuation factor.
      3. Fit LightGBM PR model on actual plant data if available.
      4. Build bootstrap residual pool for stochastic simulation.
    """
    print("\n" + "=" * 64)
    print("STAGE 5 : Solar model calibration (PR model)")
    print("=" * 64)

    # Load actual plant data if available
    plant_df = None
    try:
        raw = pd.read_excel(SOLAR_XLSX)
        raw["MW"] = pd.to_numeric(raw["MW"], errors="coerce")
        start = pd.Timestamp("2023-01-01 00:00:00")
        raw["Date"] = start + pd.to_timedelta(
            raw["Hours since 00:00 Jan 1"], unit="h"
        )
        plant_df = (
            raw.set_index("Date")
            .asfreq("30min")[["MW"]]
        )
        plant_df = plant_df.interpolate(method='linear')
        plant_df = plant_df[plant_df["MW"] > 0]
        print(f"  Actual plant data: {len(plant_df):,} bars")
    except Exception as e:
        print(f"  WARNING: Could not load plant data ({e}). "
              "Using physics fallback.")        
        plant_df = sm.test_geometry_only_model(diagnostic_plot=False)
        
    print(plant_df.head())
    solar_model, weather_df, _ = sm.get_solar_profile(
        plant_df=plant_df,
        plant_mw_col="MW",
    )
    residual_pool = solar_model.residual_pool   # stored on model instance
    return solar_model, weather_df, residual_pool


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 :  SINGLE SIMULATION PATH  (used by Monte Carlo workers)
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_path(args):
    """
    Run a single simulation path and return total discounted revenue.
    Returns NaN on failure (caller filters these out).
    """
    (seed, price_params, jump_params, log_u_params,
     demand0, kappa_d, theta_d, sigma_d,
     solar_model, weather_df, residual_pool,
     overnight_stoch, n_days) = args

    try:
        n_steps = int(n_days * 24 / DT_HOURS)

        # ── Price: SV-Jump stochastic component ──────────────────────────
        sim_sv  = pm2.simulate_mrsvcj(
            price_params, jump_params,
            T_steps=n_steps, dt=DT_YEARS, seed=seed,
            overnight_stoch=overnight_stoch,
        )
        X_t      = sim_sv["X"].values
        log_u_arr = pm2.log_U(
            dt_input=pm2._ORIGIN, n_bars=n_steps, **log_u_params,
        )
        price = np.exp(np.clip(X_t + log_u_arr, None, np.log(15_000.0)))
        price = np.where(np.isfinite(price), price, 0.0)

        # ── Demand: OU process ────────────────────────────────────────────
        demand_mw = ar.ornstein_uhlenbeck_process(
            n_steps=n_steps, x0=demand0,
            kappa=kappa_d, theta=theta_d, sigma=sigma_d,
            dt=DT_HOURS, seed=seed + 1,
        )
        demand_mw = np.clip(demand_mw, 0.0, None)

        # ── Solar: physics-informed PR model ─────────────────────────────
        solar_mw = solar_model.simulate(
            n_steps=n_steps,
            dt_hours=DT_HOURS,
            start_dt=pd.Timestamp("2020-05-05 00:30:00"),
            weather_df=weather_df,
            residual_pool=residual_pool,
            seed=seed + 2,
        )

        # ── Battery dispatch ──────────────────────────────────────────────
        battery_mwh  = np.zeros(n_steps)
        charge_mw    = np.zeros(n_steps)
        discharge_mw = np.zeros(n_steps)
        export_mw    = np.zeros(n_steps)
        battery_mwh[0] = 0.5 * BATTERY_CAPACITY_MWH

        for i in range(1, n_steps):
            batt   = battery_mwh[i - 1]
            excess = max(0.0, solar_mw[i] - demand_mw[i])
            short  = max(0.0, demand_mw[i] - solar_mw[i])

            if excess > 0:
                space = (BATTERY_CAPACITY_MWH - batt) / (ETA_CHARGE * DT_HOURS)
                chg   = min(excess, BATTERY_POWER_MAX_MW, space)
                charge_mw[i]  = chg
                export_mw[i]  = excess - chg
            elif short > 0:
                avail = (batt * ETA_DISCHARGE) / DT_HOURS
                discharge_mw[i] = min(short, BATTERY_POWER_MAX_MW, avail)

            net = (charge_mw[i] * ETA_CHARGE
                   - discharge_mw[i] / ETA_DISCHARGE) * DT_HOURS
            battery_mwh[i] = np.clip(batt + net, 0.0, BATTERY_CAPACITY_MWH)

        # ── Revenue & NPV ─────────────────────────────────────────────────
        time_years       = np.arange(n_steps) * DT_HOURS / 8760.0
        discount         = 1.0 / (1.0 + DISCOUNT_RATE) ** time_years
        revenue_per_step = (export_mw + discharge_mw) * price * DT_HOURS
        npv              = float(np.sum(revenue_per_step * discount))

        return npv if np.isfinite(npv) else float("nan")

    except Exception as e:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6–7 :  MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(price_params, jump_params, log_u_params,
                    demand0, kappa_d, theta_d, sigma_d,
                    solar_model, weather_df, residual_pool,
                    overnight_stoch=None,
                    n_sims = N_SIMULATIONS,
                    n_days = SIM_DAYS,
                    solar_cost_mw = 500,
                    battery_cost_mwh = 200,
                    fixed_opex_year = 0):
    """
    Runs a high-frequency parallelized Monte Carlo simulation integrated with a 
    20-year asset lifecycle cash flow model. Removes redundant macro distributions,
    relying entirely on bottom-up high-frequency paths.

    Returns:
    --------
    df_metrics : pd.DataFrame
        Summary statistics and percentiles for institutional project underwriting.
    figures : dict
        A dictionary containing matplotlib Figure objects:
        - 'lifecycle_distribution': The financial NPV density risk profile chart.
    """
    print("\n" + "=" * 64)
    print(f"STAGE 6-7 : Monte Carlo Capstone Engine ({n_sims} paths × {n_days} days)")
    print("=" * 64)

    # ── 1. HIGH-FREQUENCY TIME-SERIES TRAJECTORY EXECUTION ──────────────────
    base_args = (
        price_params, jump_params, log_u_params,
        demand0, kappa_d, theta_d, sigma_d,
        solar_model, weather_df, residual_pool,
        overnight_stoch, n_days
    )
    arg_list = [(seed,) + base_args for seed in range(n_sims)]
    high_freq_base_revenues = []

    if N_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_run_one_path, a): i for i, a in enumerate(arg_list)}
            for i, fut in enumerate(as_completed(futures)):
                high_freq_base_revenues.append(fut.result())
                if (i + 1) % max(1, N_SIMULATIONS // 10) == 0:
                    print(f"  [Time-Series Engine] {i+1}/{N_SIMULATIONS} paths complete …")
    else:
        for i, a in enumerate(arg_list):
            high_freq_base_revenues.append(_run_one_path(a))
            if (i + 1) % max(1, N_SIMULATIONS // 10) == 0:
                print(f"  [Time-Series Engine] {i+1}/{N_SIMULATIONS} paths complete …")

    arr_base = np.array(high_freq_base_revenues)
    arr_base = arr_base[np.isfinite(arr_base)]
    data1, fig_1 = plot_npv_distribution(arr_base,n_days)
    
    if len(arr_base) == 0:
        raise RuntimeError("All high-frequency Monte Carlo paths returned NaN. Verify core parameters.")

    # ── 2. CLEAN 20-YEAR LIFECYCLE CASH FLOW MATRIX ─────────────────────────
    # Establish capital cost architecture bounds
    calculated_capex = ((SOLAR_MAX_MW * solar_cost_mw) + (BATTERY_CAPACITY_MWH * battery_cost_mwh)) / 1000.0

    portfolio_npv_distribution = np.zeros(len(arr_base))
    
    # Map the organic high-frequency variations across the project lifecycle horizon
    for run, base_annual_revenue in enumerate(arr_base):
        path_cash_flows_mil = []
        
        # Convert base high-frequency step dollar revenue into millions
        revenue_base_mil = base_annual_revenue / 1e6
        
        for yr in range(1, 21):
            # Apply technical degradation components directly to the cash-flow logic
            current_yr_solar_decay = (1.0 - 0.006) ** yr
            current_yr_battery_decay = 1.0 - (0.018 * yr)
            
            # Blend degradation components proportionally based on asset size contributions
            total_weight = SOLAR_MAX_MW + BATTERY_CAPACITY_MWH
            blended_decay = (
                (SOLAR_MAX_MW * current_yr_solar_decay) + 
                (BATTERY_CAPACITY_MWH * current_yr_battery_decay)
            ) / total_weight
            
            # Dynamic yearly revenues derived from bottom-up paths
            total_gross_revenue_mil = revenue_base_mil * blended_decay
            yearly_opex_mil = fixed_opex_year / 1000.0
            
            net_cash_flow_yr = total_gross_revenue_mil - yearly_opex_mil
            
            # Discount cash flows back to present value using native DISCOUNT_RATE
            discounted_cash_flow = net_cash_flow_yr / ((1.0 + DISCOUNT_RATE) ** yr)
            path_cash_flows_mil.append(discounted_cash_flow)
            
        portfolio_npv_distribution[run] = sum(path_cash_flows_mil) - calculated_capex

    # ── 3. METRICS GENERATION & DATAFRAME COMPILATION ───────────────────────
    mean_portfolio_npv = portfolio_npv_distribution.mean()
    p10_var_floor = np.percentile(portfolio_npv_distribution, 10)
    p90_upside_bound = np.percentile(portfolio_npv_distribution, 90)
    probability_of_loss = (portfolio_npv_distribution < 0.0).sum() / len(portfolio_npv_distribution) * 100.0

    percentiles_labels = [
        "P5 (Severe Downside Loss)", 
        "P10 (Value-at-Risk Floor)", 
        "P25 (Conservative Scenario)", 
        "P50 (Expected Median)", 
        "P75 (Optimistic Scenario)", 
        "P90 (Target Growth Upside)", 
        "P95 (Maximum Structural Potential)"
    ]
    percentile_values = np.percentile(portfolio_npv_distribution, 
                                      [5, 10, 25, 50, 75, 90, 95])
    
    # Build structural financial reporting ledger
    metrics_data = {
        "Metric Attribute": [
            "Project Initial CAPEX Expenditure",
            "Expected Mean NPV Outcome",
            "P10 Downside Risk Floor",
            "P90 Growth Upside Bound",
            "Probability of Capital Loss (%)"
        ] + percentiles_labels,
        "Value Mapping": [
            f"${calculated_capex:.3f} Million",
            f"${mean_portfolio_npv:.3f} Million",
            f"${p10_var_floor:.3f} Million",
            f"${p90_upside_bound:.3f} Million",
            f"{probability_of_loss:.2f}%"
        ] + [f"${v:.3f} Million" for v in percentile_values]
    }
    df_metrics = pd.DataFrame(metrics_data).set_index("Metric Attribute")

    # ── 4. GRAPHICAL DIAGNOSTICS GENERATION (PURE FIG OBJECTS) ─────────────
    figures = {}
    
    fig_m, ax_m = plt.subplots(figsize=(11, 5))
    counts, bins, patches = ax_m.hist(portfolio_npv_distribution, bins=45, edgecolor="white", alpha=0.75, color="teal")
    
    # Dynamic risk coloring mapping (negative yields highlighted in deep crimson)
    for patch, bin_left in zip(patches, bins[:-1]):
        if bin_left < 0.0:
            patch.set_facecolor("#A03B37")
            
    ax_m.axvline(0.0, color="black", linestyle="-", linewidth=1.5, label="Capital Breakeven Threshold")
    ax_m.axvline(mean_portfolio_npv, color="blue", linestyle="--", linewidth=2.5, label=f"Expected Mean NPV (${mean_portfolio_npv:.1f}M)")
    ax_m.axvline(p10_var_floor, color="purple", linestyle=":", linewidth=2, label=f"P10 Value-at-Risk (${p10_var_floor:.1f}M)")
    
    ax_m.set_title(f"Implied Investment NPV Distribution Profile ({N_SIMULATIONS} Bottom-Up Lifecycle Trajectories)")
    ax_m.set_xlabel("Net Present Value Yield Outcome ($ Millions)")
    ax_m.set_ylabel("Frequency Path Count Hits")
    ax_m.legend(loc="upper right")
    ax_m.grid(True, alpha=0.1)
    
    plt.tight_layout()
    figures['lifecycle_distribution'] = fig_m
    
    return pd.DataFrame(data1), fig_1, df_metrics, fig_m


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 :  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_diagnostics(price_data, log_u_params, price_params, jump_params):
    """4-panel plot comparing historical vs two simulated stochastic paths."""
    n = len(price_data)
    sim1 = pm2.simulate_mrsvcj(price_params, jump_params, T_steps=n, dt=DT_YEARS, seed=1)
    sim2 = pm2.simulate_mrsvcj(price_params, jump_params, T_steps=n, dt=DT_YEARS, seed=2)

    log_u_arr = pm2.log_U(
        dt_input=price_data.index[0], n_bars=n, **log_u_params
    )
    sim_log_price = log_u_arr + sim1["X"].values
    sim_price     = np.exp(np.clip(sim_log_price, None, np.log(5_000)))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("Price Model Diagnostics", fontsize=13, fontweight="bold")

    idx = price_data.index

    axes[0].plot(idx, price_data["Stochastic"], alpha=0.6,
                 label="Historical stochastic", linewidth=0.7)
    axes[0].plot(idx, sim1["X"].values, alpha=0.7,
                 label="Simulated path 1", linewidth=0.7)
    axes[0].plot(idx, sim2["X"].values, alpha=0.7,
                 label="Simulated path 2", linewidth=0.7)
    axes[0].set_title("Stochastic Component  X(t)")
    axes[0].set_ylabel("Log-price residual")
    axes[0].legend(fontsize=8)

    resid = price_data["Stochastic"].values - sim1["X"].values
    axes[1].plot(idx, resid, alpha=0.7, linewidth=0.6, color="tab:orange")
    axes[1].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[1].set_title("Residuals  (Historical − Simulated 1)")
    axes[1].set_ylabel("Δ log-price")

    axes[2].plot(idx, price_data["logPrice"], alpha=0.6,
                 label="Historical", linewidth=0.7)
    axes[2].plot(idx, sim_log_price, alpha=0.7,
                 label="Simulated 1", linewidth=0.7)
    axes[2].set_title("Log-Price")
    axes[2].set_ylabel("log($/MWh)")
    axes[2].legend(fontsize=8)

    axes[3].plot(idx, price_data["Price"], alpha=0.6,
                 label="Historical", linewidth=0.7)
    axes[3].plot(idx, sim_price, alpha=0.7, label="Simulated 1", linewidth=0.7)
    axes[3].set_title("Electricity Price")
    axes[3].set_ylabel("$/MWh")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("price_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_npv_distribution(npv_values: np.ndarray,
                          n_days = SIM_DAYS):
    """
    Histogram + KDE of NPV distribution with confidence interval markers.
    """
    pcts   = [5, 25, 50, 75, 95]
    qvals  = np.percentile(npv_values, pcts)
    mean   = np.mean(npv_values)
    std    = np.std(npv_values)

    print("\n" + "=" * 64)
    print("Monte Carlo NPV Summary")
    print("=" * 64)
    print(f"  Simulations : {len(npv_values)}")
    print(f"  Mean NPV    : ${mean:,.0f}")
    print(f"  Std dev     : ${std:,.0f}")
    for p, q in zip(pcts, qvals):
        print(f"  P{p:2d}         : ${q:,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Solar + Battery NPV Distribution  "
        f"({n_days}-day horizon, {DISCOUNT_RATE*100:.0f}% discount rate)",
        fontsize=12, fontweight="bold",
    )

    # ── Left: histogram ──────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(npv_values, bins=max(20, N_SIMULATIONS // 10),
            color="steelblue", edgecolor="white", alpha=0.85)
    colours = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]
    labels  = [f"P{p}" for p in pcts]
    for q, c, lbl in zip(qvals, colours, labels):
        ax.axvline(q, color=c, linewidth=1.6, linestyle="--",
                   label=f"{lbl}: ${q:,.0f}")
    ax.axvline(mean, color="black", linewidth=2.0, label=f"Mean: ${mean:,.0f}")
    ax.set_xlabel("NPV ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("NPV Histogram")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e3:.0f}k"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: sorted paths + CI band ────────────────────────────────────
    ax2 = axes[1]
    sorted_npv = np.sort(npv_values)
    cdf        = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
    ax2.plot(sorted_npv, cdf * 100, color="steelblue", linewidth=1.8)
    ci_colours = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]
    for q, c, lbl in zip(qvals, ci_colours, labels):
        pct_val = pcts[labels.index(lbl)]
        ax2.axvline(q, color=c, linewidth=1.4, linestyle="--")
        ax2.axhline(pct_val, color=c, linewidth=0.8, linestyle=":")
        ax2.annotate(f" {lbl}",
                     xy=(q, pct_val), fontsize=8, color=c,
                     va="bottom", ha="left")
    ax2.set_xlabel("NPV ($)")
    ax2.set_ylabel("Cumulative probability (%)")
    ax2.set_title("Empirical CDF")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e3:.0f}k"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("npv_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    result = {"Measure": [f"P{p}" for p in pcts] + ["Mean", "Std Dev"],
                "Value": list(qvals) + [mean, std]}

    return pd.DataFrame(result).set_index("Measure"), fig


def plot_single_path_diagnostics(price_params, 
                                 jump_params, 
                                 log_u_params,
                                 demand0, kappa_d, theta_d, sigma_d,
                                 solar_model, weather_df, residual_pool,
                                 overnight_stoch):
    """
    Run a single forward simulation and plot price, demand, solar,
    battery SOC, and revenue over time.
    """
    n_steps = int(SIM_DAYS * 24 / DT_HOURS)

    sim_sv    = pm2.simulate_mrsvcj(price_params, jump_params,
                                    T_steps=n_steps, dt=DT_YEARS, seed=99,
                                    overnight_stoch=overnight_stoch)
    log_u_arr = pm2.log_U(dt_input=pm2._ORIGIN, n_bars=n_steps, **log_u_params)
    price     = np.clip(np.exp(sim_sv["X"].values + log_u_arr), 0.0, 15_000.0)

    demand_mw = np.clip(ar.ornstein_uhlenbeck_process(
        n_steps=n_steps, x0=demand0,
        kappa=kappa_d, theta=theta_d, sigma=sigma_d,
        dt=DT_HOURS, seed=100), 0.0, None)

    solar_mw = solar_model.simulate(
        n_steps=n_steps,
        dt_hours=DT_HOURS,
        start_dt=pd.Timestamp("2020-05-05 00:30:00"),
        weather_df=weather_df,
        residual_pool=residual_pool,
        seed=101,
    )

    # Dispatch
    battery_mwh  = np.zeros(n_steps);  battery_mwh[0] = 0.5 * BATTERY_CAPACITY_MWH
    charge_mw    = np.zeros(n_steps)
    discharge_mw = np.zeros(n_steps)
    export_mw    = np.zeros(n_steps)
    for i in range(1, n_steps):
        batt   = battery_mwh[i - 1]
        excess = max(0.0, solar_mw[i] - demand_mw[i])
        short  = max(0.0, demand_mw[i] - solar_mw[i])
        if excess > 0:
            space = (BATTERY_CAPACITY_MWH - batt) / (ETA_CHARGE * DT_HOURS)
            chg   = min(excess, BATTERY_POWER_MAX_MW, space)
            charge_mw[i] = chg;  export_mw[i] = excess - chg
        elif short > 0:
            avail = (batt * ETA_DISCHARGE) / DT_HOURS
            discharge_mw[i] = min(short, BATTERY_POWER_MAX_MW, avail)
        net = (charge_mw[i] * ETA_CHARGE
               - discharge_mw[i] / ETA_DISCHARGE) * DT_HOURS
        battery_mwh[i] = np.clip(batt + net, 0.0, BATTERY_CAPACITY_MWH)

    time_years = np.arange(n_steps) * DT_HOURS / 8760.0
    discount   = 1.0 / (1.0 + DISCOUNT_RATE) ** time_years
    revenue_ts = (export_mw + discharge_mw) * price * DT_HOURS * discount
    cum_rev    = np.cumsum(revenue_ts)

    # Limit plot to first 30 days for clarity
    nplot = min(n_steps, 30 * 48)
    dyz = nplot/48
    t     = np.arange(nplot) * DT_HOURS   # hours

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Single Forward Path Diagnostics  (first {dyz} days)",
                 fontsize=12, fontweight="bold")

    axes[0].plot(t, price[:nplot], linewidth=0.8, color="tab:red")
    axes[0].set_ylabel("Price\n($/MWh)")
    axes[0].set_title("Electricity Price")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, demand_mw[:nplot], label="Demand", linewidth=0.8)
    axes[1].plot(t, solar_mw[:nplot],  label="Solar (PR model)",  linewidth=0.8,
                 color="gold")
    axes[1].set_ylabel("Power (MW)")
    axes[1].set_title("Demand vs Solar Generation")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].stackplot(t,
                      export_mw[:nplot],
                      discharge_mw[:nplot],
                      charge_mw[:nplot],
                      labels=["Export", "Discharge", "Charge"],
                      alpha=0.7)
    axes[2].set_ylabel("Power (MW)")
    axes[2].set_title("Battery & Export Dispatch")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, battery_mwh[:nplot] / BATTERY_CAPACITY_MWH * 100,
                 linewidth=0.9, color="tab:green")
    axes[3].set_ylabel("SOC (%)")
    axes[3].set_title("Battery State of Charge")
    axes[3].set_ylim(0, 105)
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(np.arange(n_steps)[:nplot] * DT_HOURS, cum_rev[:nplot],
                 linewidth=1.0, color="tab:purple")
    axes[4].set_ylabel("Cum. NPV ($)")
    axes[4].set_xlabel("Hours from simulation start")
    axes[4].set_title("Cumulative Discounted Revenue (full path)")
    axes[4].yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e3:.0f}k"))
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("single_path_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IS_CALIBRATED = True
    # ── Stages 1-3: price ────────────────────────────────────────────────
    price_data, log_u_params, price_params, jump_params, overnight_stoch = calibrate_price_model()

    # ── Stage 4: demand ──────────────────────────────────────────────────
    demand_data, kappa_d, theta_d, sigma_d = calibrate_demand_model()
    demand0 = float(demand_data["Load at PF 0.98 (MW)"].iloc[0])

    # ── Stage 5: solar ───────────────────────────────────────────────────
    solar_model, weather_df_solar, residual_pool = calibrate_solar_model()

    # ── Solar diagnostics ────────────────────────────────────────────────
    try:
        plant_df_diag = None
        try:
            raw = pd.read_excel(SOLAR_XLSX)
            raw["MW"] = pd.to_numeric(raw["MW"], errors="coerce")
            start_diag = pd.Timestamp("2023-01-01 00:00:00")
            raw["Date"] = start_diag + pd.to_timedelta(
                raw["Hours since 00:00 Jan 1"], unit="h"
            )
            plant_df_diag = raw.set_index("Date").asfreq("30min")[["MW"]]
            plant_df_diag = plant_df_diag[plant_df_diag["MW"] > 0]
        except Exception:
            pass
        sm.plot_solar_diagnostics2(
            model=solar_model,
            weather_df=weather_df_solar,
            residual_pool=residual_pool,
            plant_df=plant_df_diag,
            plant_mw_col="MW",
            save_path="solar_diagnostics.png",
        )
    except Exception as e:
        print(f"  WARNING: Solar diagnostics failed: {e}")

    # ── Diagnostic: single forward path ──────────────────────────────────
    plot_price_diagnostics(price_data, log_u_params, price_params, jump_params)
    plot_single_path_diagnostics(
        price_params, jump_params, log_u_params,
        demand0, kappa_d, theta_d, sigma_d,
        solar_model, weather_df_solar, residual_pool,
        overnight_stoch, 
        )

    # ── Stages 6-7: Monte Carlo ───────────────────────────────────────────
    npv_values = run_monte_carlo(
        price_params, jump_params, log_u_params,
        demand0, kappa_d, theta_d, sigma_d,
        solar_model, weather_df_solar, residual_pool,
        overnight_stoch=overnight_stoch,
    )

    # ── Stage 8: results ─────────────────────────────────────────────────
    stats = plot_npv_distribution(npv_values)

    print("\nFinal NPV Statistics")
    print("-" * 40)
    for k, v in stats.items():
        print(f"  {k:6s}: ${v:,.0f}")