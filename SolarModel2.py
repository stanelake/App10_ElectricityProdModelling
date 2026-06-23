"""
SolarModel.py
=============
Physics-informed solar generation model for Kaikohe, NZ.
Strictly handles timezone unification to naive local time at ingestion.

Architecture
------------
Rather than fitting a simple OU process to raw MW output — which fails
because solar is zero for ~12 hrs/day and highly seasonal — this module
models the **Performance Ratio (PR)**: the fraction of theoretically
available irradiance that the plant converts to electricity.

    PR_t = actual_MW_t / (capacity_MW × η_STC × GHI_t / 1000)

PR is stable (0.7–0.95 on good days, lower with cloud/heat), learnable
from limited historical data, and physically bounded.  Generation is then:

    solar_MW_t = capacity_MW × η_STC × (GHI_t / 1000) × PR_t

For Monte Carlo simulation we bootstrap PR residuals from historical data
(conditioned on time-of-day and season) so each simulated year is a
physically consistent realisation of what the plant could have produced.

Pipeline
--------
1. fetch_nasa_weather()      — download 15-yr hourly GHI + met from NASA POWER
2. build_solar_features()    — compute pvlib solar geometry + cloud factor
3. SolarPRModel.fit()        — train LightGBM PR model on actual plant data
4. SolarPRModel.simulate()   — generate 30-min synthetic MW paths
5. get_solar_profile()       — convenience wrapper used by main.py

Dependencies
------------
    pip install pvlib lightgbm scikit-learn requests
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import requests
from pvlib.location import Location
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# PLANT CONSTANTS  (Kaikohe, Northland NZ)
# ─────────────────────────────────────────────────────────────────────────────

PLANT_LAT          = -35.4
PLANT_LON          = 173.8
PLANT_TZ           = "Pacific/Auckland"
PLANT_CAPACITY_MW  = 65.0
PLANT_ETA_STC      = 0.85     # panel efficiency at Standard Test Conditions
ANNUAL_DEGRADATION = 0.005    # 0.5% per year panel degradation
NASA_CACHE_FILE    = "DataSource/Solar_Generation_Data/nasa_weather_cache.csv"
PRODUCTION_CACHE_FILE     = "DataSource/Solar_Generation_Data/Results from Simulations.csv"




def fetch_nasa_weather(force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads historical hourly weather data from NASA POWER for Kaikohe, NZ.
    Converts timestamps from UTC to Pacific/Auckland, then strips timezone awareness
    to output a clean DataFrame matching the local wall-clock time.
    """
    if not force_refresh and os.path.exists(NASA_CACHE_FILE):
        print(f"  [SolarModel] Loading NASA weather from local cache: {NASA_CACHE_FILE}")
        df = pd.read_csv(NASA_CACHE_FILE, index_col=0, parse_dates=True)
        return df

    print("  [**SolarModel] Fetching 15-year weather history from NASA POWER API...")
    url = (
        f"https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M,RH2M,WS2M"
        f"&community=RE&longitude={PLANT_LON}&latitude={PLANT_LAT}"
        f"&start=20100101&end=20251231&format=JSON"
    )
    
    response = requests.get(url).json()
    parameters = response["properties"]["parameter"]
    
    # Reconstruct dataframe from nested JSON keys (Format: YYYYMMDDHH)
    raw_df = pd.DataFrame(parameters)
    raw_df.index = pd.to_datetime(raw_df.index, format="%Y%m%d%H")
    
    # Crucial Conversion: Localize to UTC (NASA default) -> Convert to NZ -> Strip awareness
    raw_df = raw_df.tz_localize("UTC").tz_convert(PLANT_TZ).tz_localize(None)
    
    # Map raw parameter tags to intuitive engineering names
    rename_dict = {
        "ALLSKY_SFC_SW_DWN": "GHI",
        "CLRSKY_SFC_SW_DWN": "CLRSKY_GHI",
        "T2M": "T2M",
        "RH2M": "RH2M",
        "WS2M": "WS2M"
    }
    df = raw_df.rename(columns=rename_dict)[list(rename_dict.values())]
    
    df.to_csv(NASA_CACHE_FILE)
    print(f"  [SolarModel] Cached clean weather file ({len(df)} rows).")
    return df

def load_default_prod_data():
    plant_df = pd.read_csv(PRODUCTION_CACHE_FILE, index_col=0)
    start = pd.Timestamp("2023-01-01 00:00:00")
    plant_df["Date"] = start + pd.to_timedelta(
        np.arange(len(plant_df))+0.5, unit="h"
    )
    plant_df = (
        plant_df.set_index("Date")
        .asfreq("30min")[["MW"]]
    )
    plant_df = plant_df.interpolate(method='linear')
    plant_df = plant_df[plant_df["MW"] > 0]
    return plant_df

def build_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates clear-sky geometry metrics and cyclical variables based on 
    the naive local datetime index.
    """
    df = df.copy()
    
    # CRITICAL ADDITION: If the index is already tz-aware (e.g. from a cached file), 
    # strip it back to a naive local clock before continuing.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    site = Location(latitude=PLANT_LAT, longitude=PLANT_LON, tz=PLANT_TZ)
    
    # pvlib requires a localized index to calculate accurate zenith angles,
    # so we provisionally localize it, parse metrics, and drop back to naive
    localized_idx = df.index.tz_localize(PLANT_TZ, ambiguous="NaT", nonexistent="shift_forward")
    solar_pos = site.get_solarposition(localized_idx)
    
    df["elevation"] = solar_pos["elevation"].values
    df["azimuth"] = solar_pos["azimuth"].values
    
    # Calculate Cloud Attenuation Factor (Handle zero-division at night)
    df["cloud_factor"] = np.where(
        df["CLRSKY_GHI"] > 0,
        (df["GHI"] / df["CLRSKY_GHI"]).clip(0.0, 1.0),
        0.0
    )
    
    # Construct time vectors
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    
    return df

def _geometry_only_mw(
    times: pd.DatetimeIndex,
    capacity_mw: float,
    eta_stc: float,
    pr_fallback: float,
    cloud_attenuation: float,
    degradation: float,
) -> np.ndarray:
    """
    Physics-only fallback generator when satellite weather data is unavailable.
    Calculates clear-sky GHI entirely using geometry positions.
    """
    site = Location(latitude=PLANT_LAT, longitude=PLANT_LON, tz=PLANT_TZ)
    loc_times = times.tz_localize(PLANT_TZ, ambiguous="NaT", nonexistent="shift_forward") if times.tz is None else times.tz_convert(PLANT_TZ)
    
    solar_pos = site.get_solarposition(loc_times)
    clearsky = site.get_clearsky(loc_times)
    
    ghi = clearsky["ghi"].values
    elev = solar_pos["elevation"].values
    
    # Calculate geometric power baseline
    phys_power = capacity_mw * eta_stc * (ghi / 1000.0)
    
    # Apply standard fallbacks and clear-sky reduction factors
    simulated_pr = np.where(elev > 0, pr_fallback * (1.0 - cloud_attenuation), 0.0)
    mw_out = phys_power * simulated_pr
    
    # Hard clamp nighttime zero-bounds
    mw_out[elev <= 0] = 0.0
    return mw_out


class SolarPRModel:
    def __init__(self, capacity_mw: float = 65.0, eta_stc: float = 0.85, degradation: float = 0.005):
        self.capacity_mw = capacity_mw
        self.eta_stc = eta_stc
        self.degradation = degradation
        self._pr_fallback = 0.80
        self._fitted = False
        
        # State Dataframes for analysis
        self.merged: pd.DataFrame | None = None
        self.residual_pool: pd.DataFrame | None = None
        self.model = LGBMRegressor(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )
        
        self.features = [
            "GHI", "CLRSKY_GHI", "cloud_factor", "T2M", "RH2M", "WS2M",
            "elevation", "azimuth", "hour_sin", "hour_cos", "doy_sin", "doy_cos"
        ]

    def _prepare_training_frame(
        self,
        weather_df: pd.DataFrame,
        plant_df: pd.DataFrame,
        plant_mw_col: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocesses and aligns data matrices for training and residuals."""
        p_df = plant_df.copy()
        if p_df.index.tz is not None:
            p_df.index = p_df.index.tz_localize(None)

        merged = weather_df.join(p_df[[plant_mw_col]], how="inner").dropna()
        if len(merged) == 0:
            raise ValueError("Data alignment failed: 0 matching row indices. Check input time steps.")

        phys = self.capacity_mw * self.eta_stc * (merged["GHI"] / 1000.0)
        mask = (merged["elevation"] > 3.0) & (phys > 0.5)

        if mask.sum() == 0:
            raise RuntimeError("No valid daylight data after filtering indices.")

        return merged, mask

    def fit(self, weather_df: pd.DataFrame, plant_df: pd.DataFrame, plant_mw_col: str = "MW"):
        """
        Aligns weather data with local production records via naive inner joins.
        Builds and retains full diagnostic frame `self.merged` inside the instance.
        """
        print("  [SolarModel] Aligning datasets and training PR model...")
        
        merged, daylight_mask = self._prepare_training_frame(weather_df, plant_df, plant_mw_col)
        
        # Add tracking variables to the persistent frame for diagnostic plot reuse
        merged["physical_baseline_mw"] = self.capacity_mw * self.eta_stc * (merged["GHI"] / 1000.0)
        
        # Performance Ratio target variable
        merged["actual_pr"] = np.where(
            merged["physical_baseline_mw"] > 0.1,
            merged[plant_mw_col] / merged["physical_baseline_mw"],
            0.0
        )
        merged["actual_pr"] = merged["actual_pr"].clip(0.0, 1.2)
        
        train_slice = merged.loc[daylight_mask]
        
        # Train lightGBM to predict the performance ratio
        self.model.fit(train_slice[self.features], train_slice["actual_pr"])
        self._fitted = True
        
        # Track predictions and residuals on the total dataset
        merged["pred_pr"] = np.where(merged["elevation"] > 0, self.model.predict(merged[self.features]), 0.0)
        merged["pred_pr"] = merged["pred_pr"].clip(0.0, 1.2)
        
        # Generation reconstruction
        merged["pred_mw"] = merged["physical_baseline_mw"] * merged["pred_pr"]
        merged.loc[merged["elevation"] <= 0, "pred_mw"] = 0.0
        
        # Save complete diagnostic set inside the object state
        self.merged = merged
        
        # Build residual pool
        res_df = pd.DataFrame(index=train_slice.index)
        res_df["hour"] = train_slice["hour"]
        res_df["month"] = train_slice["month"]
        res_df["residual"] = train_slice["actual_pr"] - self.model.predict(train_slice[self.features])
        self.residual_pool = res_df
        
        print(f"  [SolarModel] Calibration complete. R² Score: {self.get_r2(plant_mw_col):.3f}")

    def get_r2(self, plant_mw_col: str) -> float:
        if not self._fitted or self.merged is None:
            return 0.0
        from sklearn.metrics import r2_score
        daylight = self.merged[self.merged["elevation"] > 3.0]
        return r2_score(daylight[plant_mw_col], daylight["pred_mw"])

    def predict_mw(self, df: pd.DataFrame) -> pd.Series:
        """Generates point estimations for an input weather array."""
        phys = self.capacity_mw * self.eta_stc * (df["GHI"] / 1000.0)
        if self._fitted:
            pr = np.where(df["elevation"] > 0, self.model.predict(df[self.features]), 0.0)
            pr = np.clip(pr, 0.0, 1.2)
        else:
            pr = np.where(df["elevation"] > 0, self._pr_fallback, 0.0)
            
        mw = pd.Series(phys * pr, index=df.index)
        mw[df["elevation"] <= 0] = 0.0
        return mw

    def simulate(self, n_steps: int, dt_hours: float, start_dt: pd.Timestamp, 
                 weather_df: pd.DataFrame, residual_pool: pd.DataFrame | None = None, seed: int | None = None) -> np.ndarray:
        """
        Generates a synthetic path mapping historical residual variants to future weather steps.
        """
        rng = np.random.default_rng(seed)
        
        # Ensure simulation window is naive local time
        sim_start = start_dt.tz_localize(None) if start_dt.tz is not None else start_dt
        sim_index = pd.date_range(start=sim_start, periods=n_steps, freq=f"{int(dt_hours*60)}min")
        
        # Reindex weather features to future path template
        wx_sim = weather_df.reindex(sim_index).interpolate("linear").ffill().bfill()
        
        phys_arr = (self.capacity_mw * self.eta_stc * (wx_sim["GHI"] / 1000.0)).values
        elev_arr = wx_sim["elevation"].values
        
        # Predict structural deterministic PR baseline
        if self._fitted:
            pr_det = np.where(elev_arr > 0, self.model.predict(wx_sim[self.features]), 0.0)
        else:
            pr_det = np.where(elev_arr > 0, self._pr_fallback, 0.0)
            
        # Apply stochastic residual noise if a calibration pool exists
        active_pool = residual_pool if residual_pool is not None else self.residual_pool
        if active_pool is not None and len(active_pool) > 0:
            # Vectorized bootstrap lookup via pre-grouped tables
            pool_dict = {
                (h, m): grp["residual"].values 
                for (h, m), grp in active_pool.groupby(["hour", "month"])
            }
            
            hours, months = wx_sim["hour"].values, wx_sim["month"].values
            res_noise = np.zeros(n_steps)
            
            for t in range(n_steps):
                key = (hours[t], months[t])
                if key in pool_dict and elev_arr[t] > 0:
                    res_noise[t] = rng.choice(pool_dict[key])
            pr_sim = np.clip(pr_det + res_noise, 0.0, 1.2)
        else:
            pr_sim = pr_det

        # Structural Output Generation
        mw_out = phys_arr * pr_sim
        mw_out[elev_arr <= 0] = 0.0 # Strict physical nighttime zero enforcement
        return mw_out


def plot_solar_diagnostics2(
    model: SolarPRModel,
    weather_df: pd.DataFrame,
    residual_pool: pd.DataFrame | None = None,
    plant_df: pd.DataFrame | None = None,
    plant_mw_col: str = "MW",
    n_sim_paths: int = 5,
    sim_days: int = 14,
    save_path: str = "solar_diagnostics.png"
) -> None:
    """Renders comprehensive six-panel diagnostic verification figures."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # ── Panel 1: Seasonal GHI Heatmap ────────────────────────────────────────
    ax = axes[0]
    pivot = (
        weather_df.assign(month=weather_df.index.month, hour=weather_df.index.hour)
        .groupby(["month", "hour"])["GHI"].mean().unstack("hour")
    )
    cmap = LinearSegmentedColormap.from_list("solar", ["#0d1b2a", "#f4a261", "#e9c46a"])
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8)
    ax.set_yticks(range(12))
    ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean GHI (W/m²)")
    ax.set_title("Mean GHI by Month & Hour")

    # ── Panel 2: Predicted vs Actual MW ──────────────────────────────────────
    ax = axes[1]
    if model._fitted and model.merged is not None:
        daylight = model.merged[model.merged["elevation"] > 3.0]
        actual = daylight[plant_mw_col].values
        predicted = daylight["pred_mw"].values
        
        hb = ax.hexbin(predicted, actual, gridsize=40, mincnt=1, cmap="YlOrRd", linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Count")
        lim = max(predicted.max(), actual.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", label="1:1 Match")
        ax.set_title(f"Predicted vs Actual MW (R²={model.get_r2(plant_mw_col):.3f})")
        ax.set_xlabel("Predicted Output (MW)")
        ax.set_ylabel("Observed Plant Output (MW)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No Calibration Data\nAvailable", ha="center", va="center", transform=ax.transAxes)

    # ── Panel 3: Performance Ratio hourly variations ─────────────────────────
    ax = axes[2]
    if model._fitted and model.merged is not None:
        daylight = model.merged[model.merged["elevation"] > 3.0]
        hours = sorted(daylight["hour"].unique())
        box_data = [daylight[daylight["hour"] == h]["pred_pr"].values for h in hours]
        ax.boxplot(box_data, positions=hours, widths=0.6, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor="#f4a261", alpha=0.7))
        ax.set_title("Model Performance Ratio by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("PR Ratio")
        ax.set_ylim(0, 1.2)

    # ── Panel 4: PR Residual Distribution ────────────────────────────────────
    ax = axes[3]
    active_pool = residual_pool if residual_pool is not None else model.residual_pool
    if active_pool is not None:
        res = active_pool["residual"].values
        ax.hist(res, bins=60, color="steelblue", edgecolor="white", density=True, alpha=0.8)
        ax.set_title("PR Calibration Residuals")
        ax.set_xlabel("Residual (Actual PR - Predicted PR)")

    # ── Panel 5: Horizon Monte Carlo Sampling vs Actual ─────────────────────
    ax = axes[4]
    n_steps = sim_days * 48
    sim_start = pd.Timestamp("2023-01-15")  # Naive local timestamp
    sim_idx = pd.date_range(start=sim_start, periods=n_steps, freq="30min")
    t_axis = np.arange(n_steps) / 48.0    
        
    # Plot real observed plant series if present
    if plant_df is not None:
        actual_series = plant_df[plant_mw_col].reindex(sim_idx).values
        ax.plot(t_axis, actual_series, color="crimson", linewidth=1.5, 
                label="Observed Production", zorder=5)
    # Base expected trend lines
    if weather_df is not None:
        wx_slice = weather_df.reindex(sim_idx).interpolate("linear").ffill().bfill()
        mw_deterministic = model.predict_mw(wx_slice).values
    else:
        mw_deterministic = _geometry_only_mw(
            sim_idx, model.capacity_mw, model.eta_stc, model._pr_fallback, 0.0, model.degradation
        )

    ax.fill_between(t_axis, 0, mw_deterministic, color="gold", 
                    alpha=0.15, label="Expected Forecast")
    
    # Simulation paths
    path_colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_sim_paths))
    for s in range(n_sim_paths):
        path = model.simulate(n_steps, 0.5, sim_start, weather_df, residual_pool=active_pool, seed=s)
        ax.plot(t_axis, path, color=path_colors[s], alpha=0.4, linewidth=0.8,
                label="Stochastic Path" if s == 0 else None)
        
        
    ax.set_title(f"Simulation Horizon ({sim_days} Days)")
    ax.set_xlabel("Days Elapsed")
    ax.set_ylabel("Generation (MW)")
    ax.set_ylim(0, model.capacity_mw * 1.1)
    ax.legend(fontsize=8, loc="upper right")

    # ── Panel 6: Monthly Realized Means comparison ───────────────────────────
    ax = axes[5]
    months = range(1, 13)
    if model.merged is not None:
        pred_monthly = model.merged.groupby("month")["pred_mw"].mean()
        ax.bar(np.array(months) - 0.2, pred_monthly.reindex(months).values, width=0.35, label="Model", color="steelblue", alpha=0.7)
        
        if plant_df is not None:
            act_monthly = model.merged.groupby("month")[plant_mw_col].mean()
            ax.bar(np.array(months) + 0.2, act_monthly.reindex(months).values, width=0.35, label="Observed", color="tab:orange", alpha=0.7)
    ax.set_xticks(months)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=7)
    ax.set_title("Monthly Mean Balance Summary")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  [SolarModel] System verification visual saved -> {save_path}")
    plt.show()


def get_solar_profile(plant_df: pd.DataFrame, plant_mw_col: str = "MW", force_refresh: bool = False):
    """Convenience pipeline utility mapping inputs into clean training spaces."""
    weather_raw = fetch_nasa_weather(force_refresh=force_refresh)
    weather_enriched = build_solar_features(weather_raw)
    
    model = SolarPRModel(capacity_mw=65.0)
    model.fit(weather_enriched, plant_df, plant_mw_col=plant_mw_col)
    
    return model, weather_enriched, model.residual_pool

def test_geometry_only_model(start = "2023-01-01", 
                            end = "2023-12-31 23:30:00", diagnostic_plot = False):
        # Create 1 year (per 30 min) artificial solar plant data for diagnostics
        plant_df = pd.DataFrame(
            {   "date": pd.date_range(start, end, freq="30min"),
                "MW": _geometry_only_mw(
                    pd.date_range(start, end, freq="30min"),
                    PLANT_CAPACITY_MW, PLANT_ETA_STC, 0.80, 0.0, ANNUAL_DEGRADATION,
                )
            }
        )
        n = len(plant_df)
        plant_df["MW"] *= np.random.uniform(size=n)  # add noise
        plant_df = plant_df.set_index('date')
        if diagnostic_plot:
            plt.plot(plant_df.index, plant_df["MW"])
            plt.title("Simulated Solar Plant Generation (Geometry-only)")
            plt.xlabel("Date")
            plt.ylabel("MW")
            plt.grid(True, alpha=0.3)
            plt.show()
        return plant_df

if __name__ == "__main__":
    
    def test_nasa_weather_fetch(force_refresh=True):
        df = fetch_nasa_weather(force_refresh)
        print(df.head())
        print(df.describe())
        # create nasa plots for GHI  CLRSKY_GHI    T2M   RH2M  WS2M
        plts = ["GHI", "CLRSKY_GHI", "T2M", "RH2M", "WS2M"]
        fig, axes = plt.subplots(len(plts), 1, figsize=(12, 10), sharex=True)
        for i, col in enumerate(plts):
            axes[i].plot(df.index, df[col], color="steelblue", linewidth=0.5)
            axes[i].set_title(col)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()  
    def test_solar_pr_model():
        try:
            raw = pd.read_excel(SOLAR_XLSX)
            raw["MW"] = pd.to_numeric(raw["MW"], errors="coerce")
            start = pd.Timestamp("2023-01-01 00:00:00")
            raw['Date'] = start + pd.to_timedelta(raw['Hours since 00:00 Jan 1'],
                                                  unit="h")
            plant_df = (
                raw.set_index("Date")
                .asfreq("30min")[["MW"]]
            )
            plant_df = plant_df.interpolate(method='linear')
            
            plant_df = plant_df[plant_df["MW"] > 0]
            print(':-)'*20)
            print(plant_df.head(25))
            print(f"  Actual plant data: {len(plant_df):,} bars")
        except:
            plant_df = test_geometry_only_model(diagnostic_plot=False)
        model, weather_df, residual_pool = get_solar_profile(plant_df=plant_df)
        plot_solar_diagnostics2(model, weather_df, residual_pool, plant_df)
    #test_geometry_only_model()
    #test_nasa_weather_fetch(force_refresh=False)
    test_solar_pr_model()

    #model, weather_df, residual_pool = get_solar_profile(plant_df = plant_df,   plant_mw_col="MW")
    #plot_solar_diagnostics(model, weather_df, residual_pool)