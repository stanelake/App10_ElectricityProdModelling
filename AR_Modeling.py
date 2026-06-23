import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def ar_model(data, order=(1, 0, 0)):
    """"
    Fit an ARIMA model to the data and recover the fitted values.
    Parameters:
    data: Time series data to be modeled.
    order: Tuple specifying the (p, d, q) order of the ARIMA model (default is (1, 0, 0) for AR(1)).
    Returns:
    fitted_model: The fitted ARIMA model.
    """
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

def OU_parameters(ar1_model, dt=0.5, series=None):
    """"
    Convert AR(1) parameters to OU process parameters.

    Formulae (Euler-Maruyama discretisation):
        X_t = c + φ·X_{t-1} + ε,   ε ~ N(0, σ²_ε)
        ↓
        kappa = -log(φ) / dt
        theta = c / (1 - φ)
        sigma = sqrt(2·σ²_ε·kappa / (1 - φ²))

    Near-unit-root guard
    --------------------
    When φ is very close to 1 (common on short dt like 0.5 hr), kappa and
    sigma blow up due to log(φ) ≈ 0 amplification.  In that case we fall
    back to method-of-moments on the original series:
        kappa = -log(autocorr(lag=1)) / dt
        theta = mean(series)
        sigma = std(series) * sqrt(2*kappa)

    Parameters
    ----------
    ar1_model : fitted ARIMA(1,0,0) model from statsmodels
    dt        : time step in the same units as kappa (e.g. 0.5 for hours)
    series    : original data Series — required for MOM fallback

    Returns
    -------
    kappa, theta, sigma
    """
    phi    = float(ar1_model.params['ar.L1'])
    c      = float(ar1_model.params['const'])
    sig2_e = float(ar1_model.params.sigma2)

    # Stability check: AR(1) must be stationary (|φ| < 1) and positive
    # for OU interpretation (negative φ means oscillatory, not mean-reverting)
    USE_MOM = (phi <= 0.0 or phi >= 1.0 or not np.isfinite(phi))

    if not USE_MOM:
        kappa = -np.log(phi) / dt
        theta = c / (1.0 - phi)           # FIXED: was phi/(1-phi)
        denom = 1.0 - phi ** 2
        sigma = np.sqrt(2.0 * sig2_e * kappa / denom) if denom > 0 else np.nan

        # Secondary guard: if any result is non-finite or physically unreasonable
        if not all(np.isfinite([kappa, theta, sigma])) or kappa > 500:
            USE_MOM = True

    if USE_MOM:
        if series is None:
            raise ValueError(
                "AR(1) coefficient is outside (0,1) and no 'series' was "
                "supplied for method-of-moments fallback."
            )
        s      = pd.Series(series).dropna().values.astype(float)
        theta  = float(np.mean(s))
        ac1    = float(pd.Series(s).autocorr(lag=1))
        ac1    = np.clip(ac1, 1e-6, 1 - 1e-6)
        kappa  = -np.log(ac1) / dt
        sigma  = float(np.std(s)) * np.sqrt(2.0 * kappa)
        print(f"  [OU fallback to MOM: phi={phi:.6f} outside stable range]")

    return kappa, theta, sigma


def calibrate_ou_process(data, dt=0.5):
    """"
    Calibrate an OU process to the given time series data.
    Parameters:
    data: Time series data to be modeled.
    dt: Time step size (default is 0.5 for half-hourly data).
    Returns:
    kappa: Speed of mean reversion.
    theta: Long-term mean level.
    sigma: Volatility of the process.
    """
    ar1_model = ar_model(data, order=(1, 0, 0))
    kappa, theta, sigma = OU_parameters(ar1_model, dt, series=data)
    print(f"Calibrated OU parameters for demand process:")
    print(f"  kappa (mean reversion speed): \t{kappa:.4f}")
    print(f"  theta (long-term mean): \t\t{theta:.4f}")
    print(f"  sigma (volatility): \t\t\t{sigma:.4f}")

    return kappa, theta, sigma


def ornstein_uhlenbeck_process(
    n_steps,
    x0 = None,
    kappa=0.7,
    theta=0.0,
    sigma=0.3,
    dt=1.0,
    isClipped = False,
    clipParams = (None, None),
    seed=42
):
    """
    Simulate an Ornstein-Uhlenbeck (OU) mean-reverting process.

    The OU process is commonly used for:
    - electricity prices
    - energy demand
    - market imbalance
    - stochastic environmental variables

    Mathematical form:
        dX_t = kappa * (theta - X_t) dt + sigma dW_t

    Parameters
    ----------
    n_steps : int
        Number of simulation timesteps.

    kappa : float
        Mean reversion speed.

    theta : float
        Long-term equilibrium value.

    sigma : float
        Volatility parameter.

    dt : float
        Timestep size.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated OU process.
    """

    rng = np.random.default_rng(seed)

    x = np.zeros(n_steps)
    x[0] = theta if x0 is None else x0

    dW1 = np.random.standard_normal(n_steps-1) * np.sqrt(dt)  # random numbers

    for i in range(1, n_steps):

        # Mean-reverting drift
        drift = kappa * (theta - x[i - 1]) * dt

        # Random diffusion
        diffusion = sigma * dW1[i - 1]

        # Process update
        x[i] = x[i - 1] + drift + diffusion
        if isClipped:
            x[i] = np.clip(x[i], clipParams[0], clipParams[1])

    return x
