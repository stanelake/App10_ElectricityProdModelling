"""
PriceModel2.py
==============
MRSVCJ — Mean-Reverting Stochastic Volatility model with Co-Jumps.
Calibrated to a pandas DataFrame column via MCMC (MH-within-Gibbs).

Model
-----
dX_t = [η·(μ - X_t) - β·μ_J] dt + √V_t dW_t^X + J_t^X dN_t
dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V  + J_t^V dN_t
corr(dW^X, dW^V) = ρ

Jump sizes:
  J^V ~ Exp(μ_v)
  J^X | J^V ~ N(μ_x + ρ_J·J^V, σ_x²)

Parameters
----------
theta_params : η, μ, κ, θ, σ_v, ρ, β, μ_J   (structural / diffusion)
phi_params   : μ_v, μ_x, σ_x, ρ_J             (jump sizes)

Public API
----------
  loadPriceData(csv_path, log_U_params)   → (df, params)
  fit_log_U_mlr(df)                       → params dict
  add_deterministic_features(df)          → df with Fourier columns
  log_U(dt_input, n_bars, **params)       → np.ndarray
  run_full_mcmc(df, col, dt, ...)         → results dict
  summarise_posterior(results)            → pd.DataFrame
  simulate_mrsvcj(params, jump_params, T_steps, dt) → pd.DataFrame

Usage
-----
    import pandas as pd
    import PriceModel2 as pm2

    df, log_u_params = pm2.loadPriceData(log_U_params="fit")
    results = pm2.run_full_mcmc(df, col="Stochastic", dt=1/(48*365.25),
                                n_iter=10_000, burn_in=2_000)
    summary = pm2.summarise_posterior(results)
    print(summary)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from numba import njit, prange
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DETERMINISTIC LOG-PRICE COMPONENT  log U(t)
#
#  Gudkov & Ignatieva (2021) specification:
#
#    log U_t = α + β·t
#              + γ·sin((t + τ)·2π/365)
#              + Σ_{d=1}^{6}  φ_d · D_d(t)
#              + Σ_{m=1}^{11} ζ_m · M_m(t)
#
#  where
#    t       : elapsed days from the dataset origin (float)
#    τ       : phase-shift in days (nonlinear parameter, fitted by NLS)
#    D_d     : day-of-week dummy, d=1 Mon … d=6 Sat  (Sunday is baseline)
#    M_m     : month dummy, m=1 Jan … m=11 Nov       (December is baseline)
#
#  Estimation: nonlinear least squares via scipy.optimize.least_squares
#  with a soft-L1 (pseudo-Huber) loss to down-weight price spikes.
# ─────────────────────────────────────────────────────────────────────────────

# Training-set origin — first observation in the dataset.
_ORIGIN      = pd.Timestamp("2020-05-05 00:30:00")
_SECS_PER_DAY = 86_400.0
_BARS_PER_DAY = 48
_BARS_PER_YEAR = 48 * 365.25   # for dt consistency with MCMC


# ── helper: elapsed days (float) from origin ──────────────────────────────────

def elapsed_days(timestamps: pd.DatetimeIndex,
                 origin: pd.Timestamp = _ORIGIN) -> np.ndarray:
    """
    Return elapsed days (float) from *origin* for every timestamp.

    Using days (not years) keeps the linear trend coefficient β on a
    per-day scale, which is what the paper uses and avoids numerical
    precision issues from very small yearly increments.
    """
    return (timestamps - origin).total_seconds().values / _SECS_PER_DAY


# ── core model function ───────────────────────────────────────────────────────

def log_U(
    dt_input: pd.Timestamp,
    n_bars: int,
    alpha: float = 0.0,
    beta:  float = 0.0,
    gamma: float = 0.0,
    tau:   float = 0.0,
    phi:   "np.ndarray | list | None" = None,    # shape (6,)  Mon–Sat
    zeta:  "np.ndarray | list | None" = None,    # shape (11,) Jan–Nov
    psi:   "np.ndarray | list | None" = None,    # shape (47,) half-hour dummies
) -> np.ndarray:
    """
    Gudkov & Ignatieva (2021) deterministic log-price component,
    extended with intra-day half-hour dummies to capture the overnight
    low-price regime that the annual sinusoid alone cannot explain.

        log U_t = α + β·t
                  + γ·sin((t + τ)·2π/365)
                  + Σ_{d=1}^{6}  φ_d · D_d(t)
                  + Σ_{m=1}^{11} ζ_m · M_m(t)
                  + Σ_{h=1}^{47} ψ_h · H_h(t)      ← NEW

    Parameters
    ----------
    dt_input : pd.Timestamp
    n_bars   : int
    alpha    : float  — intercept
    beta     : float  — linear trend (per day)
    gamma    : float  — annual sinusoid amplitude
    tau      : float  — annual sinusoid phase shift (days)
    phi      : array-like (6,)  — Mon–Sat dummies  (Sunday = baseline)
    zeta     : array-like (11,) — Jan–Nov dummies  (December = baseline)
    psi      : array-like (47,) — half-hour-of-day dummies, bars 1–47
                                   (bar 0 = 00:30 is baseline)
                                   If None, intra-day dummies are omitted.

    Returns
    -------
    np.ndarray, shape (n_bars,)
    """
    phi  = np.zeros(6)  if phi  is None else np.asarray(phi,  dtype=float)
    zeta = np.zeros(11) if zeta is None else np.asarray(zeta, dtype=float)
    assert phi.shape  == (6,),  f"phi must have length 6,  got {phi.shape}"
    assert zeta.shape == (11,), f"zeta must have length 11, got {zeta.shape}"
    if psi is not None:
        psi = np.asarray(psi, dtype=float)
        assert psi.shape == (47,), f"psi must have length 47, got {psi.shape}"

    idx = pd.date_range(dt_input, periods=n_bars, freq="30min")
    t   = elapsed_days(idx)

    # ── (1) Linear trend ───────────────────────────────────────────────────
    trend = alpha + beta * t

    # ── (2) Annual sinusoid ────────────────────────────────────────────────
    seasonal = gamma * np.sin((t + tau) * 2.0 * np.pi / 365.0)

    # ── (3) Day-of-week dummies  (Mon=0…Sun=6; Sunday is baseline) ────────
    dow        = idx.dayofweek.values
    dow_effect = np.zeros(n_bars)
    for d in range(6):
        dow_effect += phi[d] * (dow == d).astype(float)

    # ── (4) Month dummies  (Jan=1…Dec=12; December is baseline) ───────────
    month        = idx.month.values
    month_effect = np.zeros(n_bars)
    for m in range(1, 12):
        month_effect += zeta[m - 1] * (month == m).astype(float)

    # ── (5) Intra-day half-hour dummies  (bar 0 = 00:00 midnight is baseline)
    # bar index: 00:00=0, 00:30=1, 01:00=2, … 23:30=47
    intraday_effect = np.zeros(n_bars)
    if psi is not None:
        bar_of_day = (idx.hour * 2 + idx.minute // 30).values
        for h in range(1, 48):
            intraday_effect += psi[h - 1] * (bar_of_day == h).astype(float)

    return trend + seasonal + dow_effect + month_effect + intraday_effect


# ── NLS fitting ───────────────────────────────────────────────────────────────

def fit_log_U_nls(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Fit the Gudkov & Ignatieva log_U model by two-stage nonlinear least squares.

    Why two-stage?
    --------------
    With ~18% of bars being price spikes (±5–8 log-units), even a robust
    loss function like soft-L1 is overwhelmed at f_scale=1.0 and finds the
    wrong phase τ.  The fix is:

      Stage 1 — spike detection via a rolling-median filter (1-week window).
                 Any bar deviating more than 3 MADs from its local median is
                 excluded from the NLS fit entirely.  This is more reliable
                 than relying solely on a robust loss because the spikes are
                 so extreme relative to seasonal variation that they dominate
                 even soft-L1.

      Stage 2 — NLS on the cleaned ~82% of bars with soft-L1 loss and a
                 tight f_scale=0.25 (transition at ~28% price move) for any
                 residual outliers that survived stage 1.

    A multi-start grid over τ ∈ [0, 365) in 30-day steps is used to escape
    local minima — the sinusoid likelihood surface is multimodal in τ.

    Model
    -----
        log U_t = α + β·t + γ·sin((t+τ)·2π/365) + φ·D + ζ·M

    Parameter vector: [α, β, γ, τ, φ₁…φ₆, ζ₁…ζ₁₁]  →  20 params

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and ``logPrice`` column.
    verbose : bool
        Print full parameter table (default True).

    Returns
    -------
    dict with keys: alpha, beta, gamma, tau,
                    phi (np.ndarray shape 6), zeta (np.ndarray shape 11)
    """
    try:
        from scipy.optimize import least_squares
        from sklearn.metrics import r2_score, mean_squared_error
    except ImportError as e:
        raise ImportError("scipy and scikit-learn are required.") from e

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex.")
    if "logPrice" not in df.columns:
        raise ValueError("DataFrame must contain a 'logPrice' column.")

    sub   = df[["logPrice"]].dropna()
    idx   = sub.index
    y_all = sub["logPrice"].values.astype(float)
    t_all = elapsed_days(idx)

    # ── Stage 1: spike detection ──────────────────────────────────────────
    # Strategy: per-half-hour-bin outlier detection using the Tukey fence
    # (Q1 - k*IQR, Q3 + k*IQR).  This preserves the overnight low-price
    # shape (structural, repeats every night) and only removes genuine
    # scarcity spikes that are extreme within their own hour-of-day peer group.
    # k=3.0 corresponds to "far outliers" in the Tukey sense.
    TUKEY_K        = 3.0
    bar_of_day_all = (idx.hour * 2 + idx.minute // 30).values
    clean_mask     = np.ones(len(y_all), dtype=bool)

    for h in range(48):
        hmask = bar_of_day_all == h
        if hmask.sum() < 10:
            continue
        hvals = y_all[hmask]
        q1, q3 = np.percentile(hvals, [25, 75])
        iqr    = q3 - q1
        if iqr == 0:
            continue   # perfectly constant bin — no spikes possible
        lo = q1 - TUKEY_K * iqr
        hi = q3 + TUKEY_K * iqr
        clean_mask[hmask] = (hvals >= lo) & (hvals <= hi)

    n_spike = int((~clean_mask).sum())

    y     = y_all[clean_mask]
    t     = t_all[clean_mask]
    idx_c = idx[clean_mask]

    # ── Dummy matrices on clean subset ────────────────────────────────────
    dow        = idx_c.dayofweek.values
    month      = idx_c.month.values
    bar_of_day = (idx_c.hour * 2 + idx_c.minute // 30).values

    D  = np.zeros((len(y), 6),  dtype=float)   # Mon–Sat
    M  = np.zeros((len(y), 11), dtype=float)   # Jan–Nov
    H  = np.zeros((len(y), 47), dtype=float)   # half-hour bars 1–47

    for d in range(6):
        D[:, d] = (dow == d).astype(float)
    for m in range(1, 12):
        M[:, m - 1] = (month == m).astype(float)
    for h in range(1, 48):
        H[:, h - 1] = (bar_of_day == h).astype(float)

    # Parameter vector: [α, β, γ, τ, φ₁…φ₆, ζ₁…ζ₁₁, ψ₁…ψ₄₇]  →  67 params
    def residuals(p):
        y_hat = (p[0] + p[1] * t
                 + p[2] * np.sin((t + p[3]) * 2.0 * np.pi / 365.0)
                 + D @ p[4:10]
                 + M @ p[10:21]
                 + H @ p[21:68])
        return y - y_hat

    # ── Stage 2: multi-start NLS over τ grid [0, 365) ────────────────────
    alpha0 = float(np.median(y))
    beta0  = float(np.polyfit(t, y, 1)[0])
    gamma0 = float(np.std(y)) * 0.5

    best_cost, best_result = np.inf, None
    for tau_init in np.arange(0, 365, 30, dtype=float):
        # [α, β, γ, τ, φ×6, ζ×11, ψ×47] = 67 params
        p0 = np.concatenate([[alpha0, beta0, gamma0, tau_init],
                              np.zeros(6), np.zeros(11), np.zeros(47)])
        try:
            res = least_squares(
                residuals, p0,
                loss="soft_l1",
                f_scale=0.25,
                method="trf",
                max_nfev=5_000,   # more iterations for larger param vector
                verbose=0,
            )
            if res.cost < best_cost:
                best_cost, best_result = res.cost, res
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("NLS optimisation failed for all τ starting values.")

    p_fit   = best_result.x
    alpha_f = float(p_fit[0])
    beta_f  = float(p_fit[1])
    gamma_f = float(p_fit[2])
    tau_f   = float(p_fit[3])
    phi_f   = p_fit[4:10].copy()
    zeta_f  = p_fit[10:21].copy()
    psi_f   = p_fit[21:68].copy()

    # ── Diagnostics on full series ────────────────────────────────────────
    dow_all       = idx.dayofweek.values
    month_all     = idx.month.values
    bar_all       = (idx.hour * 2 + idx.minute // 30).values

    D_all = np.zeros((len(y_all), 6),  dtype=float)
    M_all = np.zeros((len(y_all), 11), dtype=float)
    H_all = np.zeros((len(y_all), 47), dtype=float)
    for d in range(6):
        D_all[:, d] = (dow_all == d).astype(float)
    for m in range(1, 12):
        M_all[:, m - 1] = (month_all == m).astype(float)
    for h in range(1, 48):
        H_all[:, h - 1] = (bar_all == h).astype(float)

    y_hat_all = (alpha_f + beta_f * t_all
                 + gamma_f * np.sin((t_all + tau_f) * 2.0 * np.pi / 365.0)
                 + D_all @ phi_f
                 + M_all @ zeta_f
                 + H_all @ psi_f)

    resid_full  = y_all - y_hat_all
    mad_full    = float(np.median(np.abs(resid_full - np.median(resid_full))))
    non_spike   = np.abs(resid_full) < 3.0 * mad_full
    r2   = r2_score(y_all[non_spike], y_hat_all[non_spike])
    rmse = np.sqrt(mean_squared_error(y_all[non_spike], y_hat_all[non_spike]))
    spike_pct_report = 100.0 * (~non_spike).mean()

    if verbose:
        print(f"\nlog_U NLS fit:  R² = {r2:.4f}   RMSE = {rmse:.4f}  "
              f"({100 - spike_pct_report:.1f}% non-spike rows; "
              f"{spike_pct_report:.1f}% flagged as spikes)")
        print(f"  Stage-1 spike removal: {n_spike:,} / {len(y_all):,} bars "
              f"({100*n_spike/len(y_all):.1f}%) removed before NLS")
        print(f"  alpha = {alpha_f:+.6f}   beta = {beta_f:+.8f}   "
              f"gamma = {gamma_f:+.6f}   tau = {tau_f:+.4f} days")
        dow_names   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov"]
        print("  Day-of-week (baseline = Sunday):")
        for name, v in zip(dow_names, phi_f):
            print(f"    phi_{name}  = {v:+.6f}")
        print("  Monthly (baseline = December):")
        for name, v in zip(month_names, zeta_f):
            print(f"    zeta_{name} = {v:+.6f}")
        # Print intra-day dummies as a compact table
        print("  Intra-day half-hour dummies (baseline = 00:30):")
        for h in range(47):
            hh   = (h + 1) * 30
            hstr = f"{hh//60:02d}:{hh%60:02d}"
            print(f"    psi_{hstr} = {psi_f[h]:+.6f}")

    return dict(
        alpha=alpha_f, beta=beta_f, gamma=gamma_f, tau=tau_f,
        phi=phi_f, zeta=zeta_f, psi=psi_f,
    )


def loadPriceData(
    csv_path: str = "DataSource/Wholesale_Prices/Wholesale_price_trends_20260505202423.csv",
    max_fill_gap: int = 4,
    log_U_params: "dict | None | str" = None,
) -> "tuple[pd.DataFrame, dict | None]":
    """
    Load, clean, and featurise half-hourly wholesale electricity price data.

    Steps
    -----
    1. Parse CSV, rename columns, enforce 30-min frequency.
    2. Warn if gaps longer than ``max_fill_gap`` bars are filled.
    3. Clip near-zero prices before log (NZ floor = $1/MWh).
    4. Fit or apply the Gudkov & Ignatieva log_U model.
    5. Compute ``Stochastic = logPrice − log_U``.

    Parameters
    ----------
    csv_path : str
    max_fill_gap : int
        Warn when forward-fill runs exceed this many bars (default 4 = 2 hr).
    log_U_params : None | 'fit' | dict
        ``None``  → skip log_U.
        ``'fit'`` → run ``fit_log_U_nls`` and attach results.
        ``dict``  → use supplied params directly.

    Returns
    -------
    (df, fitted_params)
        df : pd.DataFrame with DatetimeIndex
        fitted_params : dict | None
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    raw = pd.read_csv(csv_path, parse_dates=False)
    raw["PeriodEnd"] = pd.to_datetime(raw["Period end"], format="%d/%m/%Y %H:%M")
    raw = raw.rename(columns={"Price ($/MWh)": "Price"})
    raw["Price"] = pd.to_numeric(raw["Price"], errors="coerce")
    drop_cols = [c for c in ["Region ID", "Region", "Period end"] if c in raw.columns]
    raw = raw.drop(columns=drop_cols)
    raw = raw.set_index("PeriodEnd").sort_index()
    raw.index = pd.to_datetime(raw.index)

    # ── 2. Enforce 30-min grid & fill gaps ───────────────────────────────────
    raw = raw.asfreq("30min")
    n_missing = raw["Price"].isna().sum()
    if n_missing:
        na_runs     = raw["Price"].isna().astype(int)
        run_lengths = na_runs.groupby((na_runs != na_runs.shift()).cumsum()).sum()
        long_runs   = run_lengths[run_lengths > max_fill_gap]
        if not long_runs.empty:
            warnings.warn(
                f"Price series has {len(long_runs)} gap(s) longer than "
                f"{max_fill_gap} bars (>{max_fill_gap*0.5:.1f} hr). "
                "Forward-filling regardless.",
                UserWarning, stacklevel=2,
            )
        raw["Price"] = raw["Price"].ffill().bfill()

    # ── 3. Log price ──────────────────────────────────────────────────────────
    # NZ allows negative prices (floor = -$600/MWh).  A hard clip at $1
    # corrupts ~7% of bars (the overnight low-price regime) and destroys the
    # stochastic residual.  Instead we use a data-adaptive floor: the 2nd
    # percentile of *positive* prices.  This clips only genuine near-zero
    # outliers while preserving the overnight low-price structure.
    pos_prices   = raw["Price"][raw["Price"] > 0]
    price_floor  = float(np.percentile(pos_prices, 2)) if len(pos_prices) > 0 else 1.0
    price_floor  = max(price_floor, 0.01)   # absolute minimum safety net
    n_below      = (raw["Price"] < price_floor).sum()
    if n_below > 0:
        warnings.warn(
            f"{n_below} prices below data-adaptive floor "
            f"{price_floor:.3f} $/MWh ({100*n_below/len(raw):.1f}% of bars) "
            "clipped before log.  Raw 'Price' column unchanged.",
            UserWarning, stacklevel=2,
        )
    raw["logPrice"] = np.log(raw["Price"].clip(lower=price_floor))
    df = raw.copy()

    # ── 4 & 5. log_U and stochastic residual ─────────────────────────────────
    fitted_params = None
    if log_U_params == "fit":
        fitted_params = fit_log_U_nls(df)
    elif isinstance(log_U_params, dict):
        fitted_params = log_U_params

    if fitted_params is not None:
        df["log_U"] = log_U(
            dt_input=df.index[0],
            n_bars=len(df),
            **fitted_params,
        )
        df["Stochastic"] = df["logPrice"] - df["log_U"]

        # ── 6. Winsorise the stochastic residual before MCMC ─────────────────
        # Winsorise per half-hour bin so the overnight low-price structure is
        # preserved.  Only genuine spikes (extreme relative to same-hour peers)
        # are clipped.  The raw column is kept as StochasticRaw.
        WINSOR_MAD   = 5.0
        stoch        = df["Stochastic"].values.copy()
        bar_of_day   = (df.index.hour * 2 + df.index.minute // 30).values
        n_clipped    = 0

        for h in range(48):
            hmask = bar_of_day == h
            if hmask.sum() < 5:
                continue
            hvals  = stoch[hmask]
            h_med  = float(np.median(hvals))
            h_mad  = float(np.median(np.abs(hvals - h_med)))
            if h_mad == 0:
                continue
            lo = h_med - WINSOR_MAD * h_mad
            hi = h_med + WINSOR_MAD * h_mad
            clipped             = np.clip(hvals, lo, hi)
            n_clipped          += int(np.sum(hvals != clipped))
            stoch[hmask]        = clipped

        df["StochasticRaw"] = df["Stochastic"].copy()

        # ── 6. Filter to trading hours for MCMC ──────────────────────────────
        # The overnight low-price regime (roughly 22:00–07:00 NZ time) is
        # structural — prices hit the market floor every night due to must-run
        # geothermal with no load.  This creates Stochastic residuals of −8
        # log-units that occur deterministically, not as random jumps.
        #
        # Feeding these to the SV-Jump MCMC causes:
        #   - sigma_v → 0  (variance frozen to explain large constant residuals)
        #   - kappa → slow (can't identify fast reversion with frozen variance)
        #   - jump rate → 96%  (only explanation for the -8 residuals)
        #
        # Solution: mark overnight bars with NaN in Stochastic so load_X_from_df
        # drops them before MCMC.  The overnight price distribution is stored in
        # OvernightStochastic for separate empirical simulation.
        #
        # Trading hours: 07:00–22:00 (bars 14–44 in 0-indexed 30-min scheme)
        # Adjust TRADING_START/END to match your market's active period.
        TRADING_START = 7    # hour, inclusive
        TRADING_END   = 22   # hour, exclusive

        bar_hour = df.index.hour
        overnight_mask = (bar_hour < TRADING_START) | (bar_hour >= TRADING_END)

        df["OvernightStochastic"] = np.where(overnight_mask,
                                             df["Stochastic"], np.nan)
        df.loc[overnight_mask, "Stochastic"] = np.nan

        # Per-bin Tukey winsorisation on trading-hours residuals only
        # (removes genuine scarcity spikes, not the overnight floor)
        stoch      = df["Stochastic"].values.copy()
        bar_of_day = (df.index.hour * 2 + df.index.minute // 30).values
        n_clipped  = 0
        # TUKEY_K=10: pass genuine electricity scarcity spikes through.
        # The SV-Jump model absorbs them via the jump component.
        TUKEY_K    = 10.0

        for h in range(48):
            hmask = (bar_of_day == h) & ~overnight_mask
            if hmask.sum() < 10:
                continue
            hvals = stoch[hmask]
            finite = hvals[np.isfinite(hvals)]
            if len(finite) < 10:
                continue
            q1, q3 = np.percentile(finite, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - TUKEY_K * iqr
            hi = q3 + TUKEY_K * iqr
            clipped = np.clip(hvals, lo, hi)
            n_clipped += int(np.sum(np.isfinite(hvals) & (hvals != clipped)))
            stoch[hmask] = clipped

        df["Stochastic"] = stoch

        # ── 7. Demean trading-hours residual ─────────────────────────────────
        trading_vals = df["Stochastic"].dropna().values
        s_med        = float(np.median(trading_vals))
        s_std        = float(np.std(trading_vals))
        s_mad        = float(np.median(np.abs(trading_vals - s_med)))

        if abs(s_med) > 0.005:
            fitted_params          = fitted_params.copy()
            fitted_params["alpha"] = fitted_params["alpha"] + s_med
            df["log_U"]            = df["log_U"] + s_med
            df["Stochastic"]       = df["logPrice"] - df["log_U"]
            df.loc[overnight_mask, "Stochastic"] = np.nan
            trading_vals = df["Stochastic"].dropna().values
            s_med_after  = float(np.median(trading_vals))
            print(f"  Recentred: absorbed median {s_med:+.4f} into alpha. "
                  f"New median = {s_med_after:+.6f}")

        n_overnight  = int(overnight_mask.sum())
        n_trading    = int((~overnight_mask).sum())
        print(f"\nStochastic residual diagnostics:")
        print(f"  Trading hours ({TRADING_START}:00–{TRADING_END}:00): "
              f"{n_trading:,} bars  ({100*n_trading/len(df):.1f}%)")
        print(f"  Overnight (excluded from MCMC): {n_overnight:,} bars "
              f"({100*n_overnight/len(df):.1f}%)")
        print(f"  Trading residual — median={float(np.median(trading_vals)):+.4f}  "
              f"MAD={s_mad:.4f}  std={s_std:.4f}")
        print(f"  Trading spikes clipped: {n_clipped} ({100*n_clipped/n_trading:.1f}%)")

    return df, fitted_params


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_X_from_df(df: pd.DataFrame, col: str = "X") -> np.ndarray:
    """Extract a clean float64 array from a DataFrame column."""
    series = df[col].dropna().astype(np.float64)
    if len(series) < 10:
        raise ValueError(f"Column '{col}' has fewer than 10 non-null values.")
    return series.values


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def initialize_state_space(
    X: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust initialisation of latent states (V, Z, JX, JV).

    V units
    -------
    The Euler scheme is  X[t] = X[t-1] + drift·dt + sqrt(V[t-1]·dt)·ε
    so V_t has units of variance **per year** (annualised).
    We compute  V_init = var(dX_non_jump) / dt  so that
    sqrt(V·dt) ≈ std(dX_non_jump) as required.

    Jump detection
    --------------
    Bars where |dX| > 3.5·MAD(dX) are flagged as jump arrivals.
    JX is initialised to the full increment for those bars.
    JV is initialised to a small positive fraction of V_init.
    """
    T  = X.shape[0]
    dX = np.diff(X)

    med = np.median(dX)
    mad = np.median(np.abs(dX - med))
    robust_std = 1.4826 * mad if mad > 0 else float(np.std(dX))
    jump_threshold = 3.5 * robust_std

    jump_mask = np.abs(dX) > jump_threshold
    non_jump  = dX[~jump_mask]

    # V is annualised variance: var(dX_non_jump) / dt
    per_step_var = float(np.var(non_jump)) if len(non_jump) > 1 else float(np.var(dX))
    V_init       = max(per_step_var / dt, 1e-8)

    Z   = np.zeros(T, dtype=np.int8)
    JX  = np.zeros(T, dtype=np.float64)
    JV  = np.zeros(T, dtype=np.float64)

    Z[1:]  = jump_mask.astype(np.int8)
    JX[1:] = np.where(jump_mask, dX, 0.0)
    # JV small relative to V_init — large JV blows up the CIR process
    JV[1:] = np.where(jump_mask, 0.01 * V_init, 0.0)

    V = np.full(T, V_init)
    return V, Z, JX, JV


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NUMBA-ACCELERATED LIKELIHOODS
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _transition_loglik_nb(
    X: np.ndarray,
    V: np.ndarray,
    Z: np.ndarray,
    JX: np.ndarray,
    JV: np.ndarray,
    dt: float,
    eta: float, mu: float, kappa: float, theta_v: float,
    sigma_v: float, rho: float, beta: float, mu_J: float,
) -> float:
    """Vectorised (loop) transition log-likelihood compiled by Numba.

    Discrete Euler-Maruyama of the directly-specified OU-SV model:
        dX_t = [η·(μ - X_t) - β·μ_J] dt + √V_t dW^X + J^X dN
        dV_t = κ(θ - V_t) dt + σ_v √V_t dW^V  + J^V dN

    Note: NO Itô -½V_t correction.  X_t is the de-seasonalised log-price
    specified directly as an OU process — it is not derived from a price
    process via Itô's lemma, so no correction applies.
    """
    psi   = rho * sigma_v
    Omega = (1.0 - rho * rho) * sigma_v * sigma_v
    if Omega <= 0.0:
        return -1e300

    const  = -np.log(2.0 * np.pi) - 0.5 * np.log(Omega)
    loglik = 0.0
    T      = X.shape[0]

    for t in range(1, T):
        Vp = V[t - 1] if V[t - 1] > 0.0 else 1e-8

        # Direct OU drift: η·(μ - X_{t-1}) - β·μ_J
        # No -½V Itô correction: X_t is the state variable directly (de-seasonalised
        # log-price), not derived via Itô from a price process. Adding -½V would
        # impose a downward drift of ~80 log-units/year at V≈160.
        ou_drift = eta * (mu - X[t - 1]) - beta * mu_J
        mu_X = X[t - 1] + ou_drift * dt + JX[t] * Z[t]
        mu_V = kappa * theta_v * dt + (1.0 - kappa * dt) * Vp + JV[t] * Z[t]

        eX = X[t] - mu_X
        eV = V[t] - mu_V

        scale = Vp * dt
        quad  = ((Omega + psi * psi) * eX * eX
                 - 2.0 * psi * eX * eV
                 + eV * eV) / (2.0 * scale * Omega)

        loglik += const - np.log(scale) - quad

    return loglik


@njit(cache=True)
def _jump_size_loglik_nb(
    Z: np.ndarray,
    JX: np.ndarray,
    JV: np.ndarray,
    mu_v: float, mu_x: float, sigma_x: float, rho_J: float,
) -> float:
    """Log-likelihood of jump sizes: J^V ~ Exp(μ_v), J^X|J^V ~ N(μ_x+ρ_J J^V, σ_x²)."""
    if mu_v <= 0.0 or sigma_x <= 0.0:
        return -1e300

    log_two_pi_sx2 = np.log(2.0 * np.pi * sigma_x * sigma_x)
    log_mu_v       = np.log(mu_v)
    loglik         = 0.0
    T              = Z.shape[0]

    for t in range(T):
        if Z[t] == 1:
            jv = JV[t]
            jx = JX[t]
            if jv <= 0.0:
                return -1e300
            loglik += -log_mu_v - jv / mu_v
            diff    = jx - (mu_x + rho_J * jv)
            loglik += -0.5 * log_two_pi_sx2 - 0.5 * diff * diff / (sigma_x * sigma_x)

    return loglik


@njit(cache=True)
def _log_prior_nb(
    eta: float, mu: float, kappa: float, theta_v: float,
    sigma_v: float, rho: float, beta: float, mu_J: float,
    mu_v: float, mu_x: float, sigma_x: float, rho_J: float,
    prior_scale_theta_v: float, prior_scale_sigma_v: float,
    prior_scale_mu_x: float,    prior_scale_sigma_x: float,
    prior_scale_mu_v: float,
) -> float:
    """
    Weakly-informative priors scaled to the data magnitude.

    theta_v prior
    -------------
    Half-normal with scale = prior_scale_theta_v (set to 3×V_init).
    This allows theta_v to be large if the data supports it, but penalises
    the explosion to 400+ seen when the MCMC has no upper guidance.

    kappa prior
    -----------
    Log-normal with log-mean=log(5), log-std=1.  This places the median
    at 5 yr^-1 (half-life ~73 days) with wide tails covering 0.5–50.
    The previous prior had median e≈2.7 which was too slow for intraday V.

    sigma_v prior
    -------------
    Half-normal with scale = prior_scale_sigma_v (set to 0.5×sqrt(V_init)).
    sigma_v is vol-of-vol and should be O(sqrt(V)), not O(V).
    """
    if sigma_v <= 0.0 or sigma_x <= 0.0 or mu_v <= 0.0:
        return -1e300
    if rho <= -1.0 or rho >= 1.0 or rho_J <= -1.0 or rho_J >= 1.0:
        return -1e300
    if beta <= 0.0 or kappa <= 0.0 or eta <= 0.0:
        return -1e300
    # Hard upper bounds: eta > 500 yr⁻¹ (half-life < 12 hrs) makes diffusion
    # absorb all spikes, killing jump identification. beta > 200/yr is
    # physically implausible for trading-hours electricity scarcity events.
    if eta > 500.0 or beta > 200.0:
        return -1e300

    lp = 0.0

    # N(0, 0.1²) on mu — Stochastic is recentred so mu should be near 0
    # mu_J is no longer sampled (derived from mu_x, sigma_x) so no prior needed
    lp += -0.5 * mu   * mu   / 0.01
    lp += -0.5 * mu_x * mu_x / (prior_scale_mu_x * prior_scale_mu_x)

    # Half-normal on positive scale params
    lp += -0.5 * (theta_v  / prior_scale_theta_v) ** 2
    lp += -0.5 * (sigma_v  / prior_scale_sigma_v) ** 2
    lp += -0.5 * (sigma_x  / prior_scale_sigma_x) ** 2
    lp += -0.5 * (mu_v     / prior_scale_mu_v)    ** 2

    # Log-normal on eta: median=100 yr⁻¹, log-std=0.75
    # Half-life = log(2)/100 ≈ 2.5 days.
    # Compromise between:
    #   - spike observation (2–10 hr half-life → eta 700–3000)
    #   - jump identification (eta < 500 needed so beta doesn't collapse)
    # At eta=100, a +3 log-unit spike (×20 price) takes ~2.5 days to halve,
    # which is the fastest reversion the model can achieve while still
    # allowing beta to identify a meaningful jump rate.
    # P5=35, P95=290 yr⁻¹  →  half-lives 0.9–7.2 days.
    le  = np.log(eta)
    lp += -0.5 * (le - np.log(100.0)) ** 2 / 0.5625

    # Log-normal on kappa: median=50 yr⁻¹, log-std=1.0
    # Variance half-life = log(2)/50 ≈ 5 days.
    lk  = np.log(kappa)
    lp += -0.5 * (lk - np.log(50.0)) ** 2 / 1.0

    # Log-normal on beta: median=20 jumps/yr, log-std=0.75
    # ~20 trading-hours spikes/year = ~1.7/month. Tight prior prevents
    # collapse to near-zero when eta is large enough to absorb spikes.
    lb  = np.log(beta)
    lp += -0.5 * (lb - np.log(20.0)) ** 2 / 0.5625

    return lp


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PYTHON WRAPPERS  (dict ↔ flat arrays for Numba)
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_theta(p: dict) -> tuple:
    return (p['eta'], p['mu'], p['kappa'], p['theta'],
            p['sigma_v'], p['rho'], p['beta'], p['mu_J'])

def _unpack_phi(p: dict) -> tuple:
    return (p['mu_v'], p['mu_x'], p['sigma_x'], p['rho_J'])

def _derive_mu_J(phi: dict) -> float:
    """
    Compute the jump compensator from the jump-size parameters.

    For J^X ~ N(mu_x, sigma_x²):
        mu_J = E[exp(J^X) - 1] = exp(mu_x + ½σ_x²) - 1

    This ensures the jump component is a martingale, i.e. jumps produce no
    net drift in X_t.  mu_J must NOT be sampled freely — doing so allows
    the MCMC to compensate a wrong mu_J with a biased mu, which is exactly
    why mu was drifting to -0.058 despite recentring.
    """
    return float(np.exp(phi['mu_x'] + 0.5 * phi['sigma_x'] ** 2) - 1.0)


def full_logpost(
    X, V, Z, JX, JV, dt,
    theta: dict, phi: dict,
    prior_scales: dict,
) -> float:
    # Always derive mu_J from phi before evaluating likelihood/prior
    theta = theta.copy()
    theta['mu_J'] = _derive_mu_J(phi)
    ll = _transition_loglik_nb(X, V, Z, JX, JV, dt, *_unpack_theta(theta))
    ll += _jump_size_loglik_nb(Z, JX, JV, *_unpack_phi(phi))
    ll += _log_prior_nb(
        *_unpack_theta(theta), *_unpack_phi(phi),
        prior_scales['theta_v'], prior_scales['sigma_v'],
        prior_scales['mu_x'],    prior_scales['sigma_x'],
        prior_scales['mu_v'],
    )
    return ll


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PROPOSALS  (random-walk MH)
# ─────────────────────────────────────────────────────────────────────────────

_LOG_SCALE_PARAMS = {"eta", "kappa", "beta"}   # propose on log-scale

def _rw_propose(params: dict, steps: dict, rng: np.random.Generator) -> dict:
    new = params.copy()
    for k, s in steps.items():
        if k in _LOG_SCALE_PARAMS and params[k] > 0:
            # Multiplicative random walk: new = old × exp(s × N(0,1))
            new[k] = params[k] * np.exp(s * rng.standard_normal())
        else:
            new[k] = params[k] + s * rng.standard_normal()
    return new

def _clip_theta(p: dict) -> dict:
    p['sigma_v'] = abs(p['sigma_v']) if p['sigma_v'] <= 0 else p['sigma_v']
    p['beta']    = abs(p['beta'])    if p['beta']    <= 0 else p['beta']
    p['rho']     = float(np.clip(p['rho'], -0.999, 0.999))
    # Hard upper bounds matching the prior constraint
    p['eta']     = float(np.clip(abs(p['eta']),  1e-3, 500.0))
    p['beta']    = float(np.clip(p['beta'],       1e-3, 200.0))
    p['kappa']   = abs(p['kappa']) if p['kappa'] <= 0 else p['kappa']
    return p

def _clip_phi(p: dict) -> dict:
    p['mu_v']    = abs(p['mu_v'])    if p['mu_v']    <= 0 else p['mu_v']
    p['sigma_x'] = abs(p['sigma_x']) if p['sigma_x'] <= 0 else p['sigma_x']
    p['rho_J']   = float(np.clip(p['rho_J'], -0.999, 0.999))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# 6.  JUMP STATE UPDATE  (vectorised over t where possible)
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _update_jumps_nb(
    X, V, Z, JX, JV, dt,
    eta, mu, kappa, theta_v, sigma_v, rho, beta, mu_J,
    mu_v, mu_x, sigma_x, rho_J,
    u_flip, u_jv, u_jx, u_accept,
):
    """
    Single-site MH update for (Z_t, JX_t, JV_t) at every t.
    All random numbers are pre-drawn for reproducibility and speed.
    """
    T = X.shape[0]
    for t in range(1, T):
        z_old  = Z[t]
        jx_old = JX[t]
        jv_old = JV[t]

        z_new = 1 - z_old
        if z_new == 1:
            jv_new = -mu_v * np.log(1.0 - u_jv[t])          # inverse-CDF Exp
            jx_new = mu_x + rho_J * jv_new + sigma_x * u_jx[t]
        else:
            jv_new = 0.0
            jx_new = 0.0

        # --- local log-posterior difference (only t and t+1 transitions matter) ---
        def _trans_t(xx, vv, zz, jxx, jvv, tp1):
            """Single-step transition log-density at step tp1 given state at tp1-1."""
            psi   = rho * sigma_v
            Omega = (1.0 - rho * rho) * sigma_v * sigma_v
            if Omega <= 0.0:
                return -1e300
            Vp   = vv[tp1 - 1] if vv[tp1 - 1] > 0.0 else 1e-8
            ou_drift = eta * (mu - xx[tp1 - 1]) - beta * mu_J
            mX   = xx[tp1 - 1] + ou_drift * dt + jxx[tp1] * zz[tp1]
            mV   = kappa * theta_v * dt + (1.0 - kappa * dt) * Vp + jvv[tp1] * zz[tp1]
            eX   = xx[tp1] - mX
            eV   = vv[tp1]  - mV
            sc   = Vp * dt
            q    = ((Omega + psi * psi) * eX * eX
                    - 2.0 * psi * eX * eV
                    + eV * eV) / (2.0 * sc * Omega)
            return -np.log(2.0 * np.pi) - 0.5 * np.log(Omega) - np.log(sc) - q

        def _jump_t(zz, jxx, jvv, tp1):
            if zz[tp1] == 0:
                return 0.0
            jv = jvv[tp1];  jx = jxx[tp1]
            if jv <= 0.0:
                return -1e300
            ll  = -np.log(mu_v) - jv / mu_v
            d   = jx - (mu_x + rho_J * jv)
            ll += -0.5 * np.log(2.0 * np.pi * sigma_x * sigma_x) - 0.5 * d * d / (sigma_x * sigma_x)
            return ll

        ll_old = _trans_t(X, V, Z,  JX,  JV,  t) + _jump_t(Z,  JX,  JV,  t)
        # temporarily substitute new values
        Z[t]  = z_new;  JX[t] = jx_new;  JV[t] = jv_new
        ll_new = _trans_t(X, V, Z, JX, JV, t) + _jump_t(Z, JX, JV, t)
        # also include next step if within bounds
        if t + 1 < T:
            ll_old += _trans_t(X, V, Z, JX, JV, t + 1)   # uses updated Z already
            # undo temp change for old
            Z[t] = z_old;  JX[t] = jx_old;  JV[t] = jv_old
            ll_old_next = _trans_t(X, V, Z, JX, JV, t + 1)
            ll_old += ll_old_next - _trans_t(X, V, Z, JX, JV, t + 1)  # cancel (same state)
            Z[t]  = z_new;  JX[t] = jx_new;  JV[t] = jv_new

        log_alpha = ll_new - ll_old
        if np.log(u_accept[t]) >= log_alpha:
            # reject – restore
            Z[t]  = z_old
            JX[t] = jx_old
            JV[t] = jv_old

    return Z, JX, JV


def update_jumps(X, V, Z, JX, JV, dt, theta, phi, rng):
    T = len(X)
    u_flip   = rng.random(T)   # not used but kept for future proposal variants
    u_jv     = rng.random(T)   # uniform for inverse-CDF Exp
    u_jx     = rng.standard_normal(T)
    u_accept = rng.random(T)
    return _update_jumps_nb(
        X, V, Z, JX, JV, dt,
        *_unpack_theta(theta), *_unpack_phi(phi),
        u_flip, u_jv, u_jx, u_accept,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ADAPTIVE STEP-SIZE  (Robbins-Monro target 0.234)
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_step(step: float, accept_rate: float, target: float = 0.234, factor: float = 1.05) -> float:
    if accept_rate > target:
        return step * factor
    return step / factor


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN MCMC LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_full_mcmc(
    df: pd.DataFrame,
    col: str = "X",
    dt: float = 1 / (48 * 365.25),
    n_iter: int = 10_000,
    burn_in: int = 2_000,
    adapt_every: int = 200,
    init_theta: dict | None = None,
    init_phi: dict | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full MCMC calibration pipeline.

    Parameters
    ----------
    df          : DataFrame containing the log-price / log-return series.
    col         : Column name to use.
    dt          : Time step in years.  ``1/(48*365.25)`` = 30-min bar, 24/7 market.
                  Use ``1/(13*252)`` for 30-min equity bars.
    n_iter      : Total MCMC iterations (including burn-in).
    burn_in     : Number of warm-up iterations discarded.
    adapt_every : Adapt MH step sizes every this many iterations.
    init_theta  : Initial structural parameters (optional).
    init_phi    : Initial jump-size parameters (optional).
    seed        : RNG seed for reproducibility.
    verbose     : Print progress every 10 % of iterations.

    Returns
    -------
    dict with keys:
        theta_samples   list[dict]  – posterior draws of structural params
        phi_samples     list[dict]  – posterior draws of jump-size params
        acceptance      dict        – MH acceptance rates
        X, V, Z, JX, JV            – final latent-state arrays
    """
    rng = np.random.default_rng(seed)
    X   = load_X_from_df(df, col)
    V, Z, JX, JV = initialize_state_space(X, dt)

    # ── Data-adaptive initialisation ─────────────────────────────────────────
    # V_t has units of annualised variance (variance per year).
    # The Euler step is  sqrt(V·dt)·ε  so  sqrt(V·dt) ≈ std(dX_non_jump).
    # Therefore:  V_init = var(dX_non_jump) / dt  — same as initialize_state_space.
    # DO NOT use baseline_var for mu or mu_J — the stochastic residual
    # is centred near zero by construction (log_U absorbed the level),
    # so mu should be initialised at the sample median of X, clamped to ±0.5.
    dX         = np.diff(X)
    mad_dX     = np.median(np.abs(dX - np.median(dX)))
    robust_std = 1.4826 * mad_dX if mad_dX > 0 else float(np.std(dX))
    non_jump   = dX[np.abs(dX) <= 3.5 * robust_std]
    per_step_var = float(np.var(non_jump)) if len(non_jump) > 1 else float(np.var(dX))
    baseline_var = max(per_step_var / dt, 1e-8)   # annualised, matches V_init

    # mu: after recentring the Stochastic residual its mean is ~0.
    # Use nanmean since overnight bars are NaN.
    mu_init = float(np.clip(np.nanmean(X), -0.05, 0.05))

    if init_theta is None:
        init_theta = dict(
            eta     = 100.0,               # half-life ~2.5 days
            mu      = mu_init,
            kappa   = 50.0,                # variance half-life ~5 days
            theta   = baseline_var,
            sigma_v = 0.5 * np.sqrt(baseline_var),
            rho     = -0.3,
            beta    = 20.0,
            mu_J    = 0.0,
        )
    if init_phi is None:
        init_phi = dict(
            mu_v    = 0.1  * baseline_var,
            mu_x    = 0.0,
            sigma_x = float(robust_std),
            rho_J   = -0.2,
        )

    theta = {k: float(v) for k, v in init_theta.items()}
    phi   = {k: float(v) for k, v in init_phi.items()}
    # Derive mu_J from initial phi — keep consistent throughout
    theta['mu_J'] = _derive_mu_J(phi)

    # Proposal steps: log-scale for eta/kappa (multiplicative moves work
    # better when parameters span orders of magnitude)
    step_theta = dict(
        eta     = 0.3,                                     # log-scale step
        mu      = max(0.02 * abs(theta['mu']), 0.002),
        kappa   = 0.3,                                     # log-scale step
        theta   = 0.05 * theta['theta'],
        sigma_v = 0.05 * theta['sigma_v'],
        rho     = 0.03,
        beta    = 0.3,                                     # log-scale step
    )
    step_phi = dict(
        mu_v    = 0.1  * phi['mu_v'],
        mu_x    = max(0.05 * abs(phi['mu_x']), 0.001),
        sigma_x = 0.05 * phi['sigma_x'],
        rho_J   = 0.03,
    )

    # ── Prior scales derived from data ───────────────────────────────────────
    prior_scales = dict(
        theta_v = max(3.0  * theta['theta'],             1e-6),
        sigma_v = max(2.0  * theta['sigma_v'],           1e-6),  # O(sqrt(V))
        mu_x    = max(3.0  * abs(phi['mu_x']) + 0.01,   0.01),
        sigma_x = max(3.0  * phi['sigma_x'],             1e-6),
        mu_v    = max(3.0  * phi['mu_v'],                1e-6),
    )

    samples_theta: list[dict] = []
    samples_phi:   list[dict] = []
    accept_theta = 0
    accept_phi   = 0
    accept_counts_theta: list[float] = []
    accept_counts_phi:   list[float] = []

    logpost = full_logpost(X, V, Z, JX, JV, dt, theta, phi, prior_scales)

    # ── Warm up Numba JIT on first call ─────────────────────────────────────
    if verbose:
        print("Compiling Numba kernels (first call) …", flush=True)
    _ = _transition_loglik_nb(X[:5], V[:5], Z[:5], JX[:5], JV[:5], dt, *_unpack_theta(theta))
    if verbose:
        n_init_jumps = int(Z.sum())
        print(f"Data diagnostics:")
        print(f"  robust_std(dX)   = {robust_std:.5g}   "
              f"per_step_var = {per_step_var:.5g}   "
              f"baseline_var (annualised) = {baseline_var:.5g}")
        print(f"  init_jumps = {n_init_jumps} ({100*n_init_jumps/len(X):.1f}%)")
        print(f"  Init θ: { {k: f'{v:.4g}' for k,v in theta.items()} }")
        print(f"  Init φ: { {k: f'{v:.4g}' for k,v in phi.items()} }")
        print(f"\nStarting MCMC: {n_iter} iterations, burn-in = {burn_in}\n")

    milestone = max(1, n_iter // 10)

    for it in range(1, n_iter + 1):

        # ── Block 1: latent jump states ──────────────────────────────────────
        Z, JX, JV = update_jumps(X, V, Z, JX, JV, dt, theta, phi, rng)

        # ── Block 2: structural parameters θ (excluding mu_J) ────────────────
        theta_cand = _clip_theta(_rw_propose(theta, step_theta, rng))
        theta_cand['mu_J'] = _derive_mu_J(phi)   # always consistent
        lp_cand    = full_logpost(X, V, Z, JX, JV, dt, theta_cand, phi, prior_scales)
        if np.log(rng.random()) < (lp_cand - logpost):
            theta   = theta_cand
            logpost = lp_cand
            accept_theta += 1

        # ── Block 3: jump-size parameters φ ─────────────────────────────────
        phi_cand = _clip_phi(_rw_propose(phi, step_phi, rng))
        lp_cand  = full_logpost(X, V, Z, JX, JV, dt, theta, phi_cand, prior_scales)
        if np.log(rng.random()) < (lp_cand - logpost):
            phi          = phi_cand
            theta['mu_J'] = _derive_mu_J(phi)   # update mu_J when phi changes
            logpost = lp_cand
            accept_phi += 1

        # ── Adaptive step sizes ──────────────────────────────────────────────
        if it % adapt_every == 0:
            ar_t = accept_theta / adapt_every
            ar_p = accept_phi   / adapt_every
            accept_counts_theta.append(ar_t)
            accept_counts_phi.append(ar_p)
            for k in step_theta:
                step_theta[k] = _adapt_step(step_theta[k], ar_t)
            for k in step_phi:
                step_phi[k]   = _adapt_step(step_phi[k],   ar_p)
            accept_theta = 0
            accept_phi   = 0

        # ── Store post burn-in ───────────────────────────────────────────────
        if it > burn_in:
            samples_theta.append(theta.copy())
            samples_phi.append(phi.copy())

        if verbose and it % milestone == 0:
            pct = 100 * it // n_iter
            n_jumps = int(Z.sum())
            print(f"  {pct:3d}%  iter={it:6d}  logpost={logpost:+.2f}  "
                  f"jumps={n_jumps}  "
                  f"ar_θ={np.mean(accept_counts_theta or [0]):.2f}  "
                  f"ar_φ={np.mean(accept_counts_phi or [0]):.2f}")

    overall_ar_theta = np.mean(accept_counts_theta) if accept_counts_theta else float("nan")
    overall_ar_phi   = np.mean(accept_counts_phi)   if accept_counts_phi   else float("nan")

    if verbose:
        print(f"\nDone. Collected {len(samples_theta)} posterior samples.")
        print(f"  Mean acceptance rate θ: {overall_ar_theta:.3f}")
        print(f"  Mean acceptance rate φ: {overall_ar_phi:.3f}\n")

    return dict(
        theta_samples=samples_theta,
        phi_samples=samples_phi,
        acceptance=dict(theta=overall_ar_theta, phi=overall_ar_phi),
        X=X, V=V, Z=Z, JX=JX, JV=JV,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9.  POSTERIOR SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summarise_posterior(results: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with mean, std, and credible intervals
    for all calibrated parameters.
    """
    rows = []
    all_params = {**results["theta_samples"][0], **results["phi_samples"][0]}

    for key in all_params:
        vals_theta = [s[key] for s in results["theta_samples"] if key in s]
        vals_phi   = [s[key] for s in results["phi_samples"]   if key in s]
        vals       = vals_theta if vals_theta else vals_phi
        arr        = np.array(vals)
        rows.append(dict(
            parameter=key,
            mean=np.mean(arr),
            std=np.std(arr),
            q2_5=np.percentile(arr, 2.5),
            median=np.median(arr),
            q97_5=np.percentile(arr, 97.5),
        ))

    return pd.DataFrame(rows).set_index("parameter").round(6)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

def simulate_mrsvcj(
    params: dict,
    jump_size_params: dict,
    T_steps: int,
    dt: float = 1 / (48 * 365.25),
    seed: int | None = None,
    overnight_stoch: "np.ndarray | None" = None,
    trading_start_hour: int = 7,
    trading_end_hour:   int = 22,
) -> pd.DataFrame:
    """
    Simulate a path from the MRSVCJ model for trading hours only, with
    overnight bars filled from an empirical distribution.

        dX_t = [η·(μ - X_t) - β·μ_J] dt + √V_t dW_t^X + J_t^X dN_t
        dV_t = κ(θ - V_t) dt + σ_v √V_t dW_t^V  + J_t^V dN_t

    Parameters
    ----------
    params : dict
        Structural parameters: eta, mu, kappa, theta, sigma_v, rho, beta, mu_J.
    jump_size_params : dict
        Jump-size parameters: mu_v, mu_x, sigma_x, rho_J.
    T_steps : int
        Total number of 30-min bars (including overnight).
    dt : float
        Time step in years.
    seed : int | None
    overnight_stoch : np.ndarray | None
        Empirical overnight stochastic residuals drawn from the historical
        OvernightStochastic column.  If supplied, overnight bars are filled
        by bootstrapping from this array.  If None, overnight bars use the
        SV-Jump model throughout (original behaviour).
    trading_start_hour : int
        First trading hour (default 7 → 07:00).
    trading_end_hour : int
        First non-trading hour (default 22 → 22:00).

    Returns
    -------
    pd.DataFrame  — Columns: X, V, Jump_Arrival, Price_Jump_Size, Vol_Jump_Size.
    """
    rng = np.random.default_rng(seed)

    eta     = params['eta']
    mu      = params['mu']
    kappa   = params['kappa']
    theta_v = params['theta']
    sigma_v = params['sigma_v']
    rho     = params['rho']
    beta    = params['beta']
    mu_J    = params['mu_J']

    mu_v    = jump_size_params['mu_v']
    mu_x    = jump_size_params['mu_x']
    sigma_x = jump_size_params['sigma_x']
    rho_J   = jump_size_params['rho_J']

    X  = np.zeros(T_steps)
    V  = np.zeros(T_steps)
    Z  = np.zeros(T_steps, dtype=np.int8)
    JX = np.zeros(T_steps)
    JV = np.zeros(T_steps)

    X[0] = mu
    V[0] = theta_v

    eps1  = rng.standard_normal(T_steps)
    eps_X = eps1
    eps_V = rho * eps1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(T_steps)

    Z[:] = (rng.random(T_steps) < beta * dt).astype(np.int8)
    Z[0] = 0

    # Bar index → hour of day (for 30-min bars anchored at 00:30)
    # bar 0 = 00:30, bar 1 = 01:00, …, bar 47 = 00:00
    # hour = (bar_index * 30 + 30) // 60 % 24
    bars_per_hour = 2
    n_overnight_pool = len(overnight_stoch) if overnight_stoch is not None else 0

    for t in range(1, T_steps):
        bar_hour = ((t % 48) * 30 + 30) // 60 % 24
        is_overnight = (bar_hour < trading_start_hour or
                        bar_hour >= trading_end_hour)

        if is_overnight and overnight_stoch is not None:
            # Bootstrap from empirical overnight distribution
            X[t] = overnight_stoch[rng.integers(n_overnight_pool)]
            V[t] = max(V[t - 1], 1e-8)   # carry variance forward silently
            continue

        Vp    = max(V[t - 1], 1e-8)
        sv_dt = np.sqrt(Vp * dt)

        if Z[t]:
            JV[t] = rng.exponential(scale=mu_v)
            JX[t] = rng.normal(loc=mu_x + rho_J * JV[t], scale=sigma_x)

        V[t] = max(
            kappa * theta_v * dt
            + (1.0 - kappa * dt) * Vp
            + JV[t] * Z[t]
            + sigma_v * sv_dt * eps_V[t],
            1e-8,
        )

        ou_drift = eta * (mu - X[t - 1]) - beta * mu_J
        X[t] = X[t - 1] + ou_drift * dt + sv_dt * eps_X[t] + JX[t] * Z[t]

    return pd.DataFrame({
        "X":               X,
        "V":               V,
        "Jump_Arrival":    Z,
        "Price_Jump_Size": JX,
        "Vol_Jump_Size":   JV,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 11.  QUICK-START DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PriceModel2 — Self-test Demo")
    print("=" * 60)

    # ── (A) Test log_U with known NLS-style params ────────────────────────
    import numpy as np
    test_phi  = np.array([0.21, 0.32, 0.38, 0.36, 0.28, 0.08])
    test_zeta = np.array([0.92, 1.03, 0.91, 0.35,-0.09,-0.34,
                          -0.30,-0.01, 0.01, 0.22, 0.58])
    lu = log_U(
        dt_input=_ORIGIN, n_bars=48*7,
        alpha=4.51, beta=-3.27e-4, gamma=0.98, tau=49.9,
        phi=test_phi, zeta=test_zeta,
    )
    print(f"\nlog_U test:  shape={lu.shape}  "
          f"min={lu.min():.3f}  max={lu.max():.3f}  mean={lu.mean():.3f}")

    # ── (B) Simulate stochastic component and MCMC-calibrate ─────────────
    true_params = dict(
        eta=2.0, mu=0.0, kappa=3.0, theta=0.04,
        sigma_v=0.3, rho=-0.7, beta=5.0, mu_J=0.0,
    )
    true_phi_sv = dict(mu_v=0.02, mu_x=-0.01, sigma_x=0.03, rho_J=-0.5)

    df_sim = simulate_mrsvcj(
        true_params, true_phi_sv,
        T_steps=2_000, dt=1/(48*365.25), seed=42,
    )
    print(f"\nSimulated stochastic path:  shape={df_sim.shape}  "
          f"X mean={df_sim['X'].mean():.4f}  "
          f"jumps={df_sim['Jump_Arrival'].sum()}")

    print(f"\nRunning MCMC (2000 obs, 4000 iter) …")
    results = run_full_mcmc(
        df_sim, col="X",
        dt=1/(48*365.25), n_iter=4_000, burn_in=1_000,
        seed=0, verbose=True,
    )

    summary = summarise_posterior(results)
    print("\nPosterior Parameter Summary")
    print("─" * 60)
    print(summary.to_string())