"""
capm_valuation.py

Utilities to compute CAPM expected return and perform simple valuations:
- fetch_price_data: fetch adjusted close price series via yfinance
- compute_beta_from_prices: compute beta by regressing stock returns vs market returns
- get_beta_from_yahoo: read Yahoo-provided beta
- capm_expected_return: CAPM formula
- gordon_growth_value: Gordon Growth Model
- dcf_valuation: simple DCF with terminal value

Requires:
    pip install yfinance pandas numpy scipy
"""
from typing import List, Optional
import numpy as np
import pandas as pd

# Import yfinance lazily so tests can patch/monkeypatch it
try:
    import yfinance as yf
except Exception:
    yf = None  # tests should monkeypatch/inject

def fetch_price_data(ticker: str, start: str, end: Optional[str] = None, interval: str = "1mo") -> pd.Series:
    """
    Fetch adjusted close price series for ticker between start and end (inclusive).
    interval examples: "1d", "1wk", "1mo"
    Returns a pandas Series of adjusted close prices indexed by date.
    """
    if yf is None:
        raise RuntimeError("yfinance is not available. Install it with `pip install yfinance`.")
    df = yf.download(ticker, start=start, end=end or None, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].rename(ticker)

def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Compute simple returns (period-over-period) from a price series.
    """
    return prices.pct_change().dropna()

def compute_beta_from_prices(ticker: str,
                             market_ticker: str = "^GSPC",
                             start: str = "2018-01-01",
                             end: Optional[str] = None,
                             freq: str = "1mo") -> float:
    """
    Compute beta by regressing stock returns on market returns:
        R_i = alpha + beta * R_m + eps
    Returns beta (float).
    """
    stock_prices = fetch_price_data(ticker, start=start, end=end, interval=freq)
    market_prices = fetch_price_data(market_ticker, start=start, end=end, interval=freq)

    data = pd.concat([stock_prices, market_prices], axis=1).dropna()
    if data.shape[0] < 6:
        raise ValueError("Not enough overlapping price points to compute beta. Try a longer range or different freq.")
    stock_returns = compute_returns(data.iloc[:, 0])
    market_returns = compute_returns(data.iloc[:, 1])
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    df.columns = ["stock", "market"]

    cov = df["stock"].cov(df["market"])
    var_m = df["market"].var()
    if var_m == 0:
        raise ValueError("Market returns have zero variance.")
    beta = cov / var_m
    return float(beta)

def get_beta_from_yahoo(ticker: str) -> Optional[float]:
    """
    Attempt to read Yahoo Finance's beta from yfinance.Ticker.info.
    Returns None if not present or if yfinance is not available.
    """
    if yf is None:
        return None
    t = yf.Ticker(ticker)
    info = getattr(t, "info", None) or {}
    b = info.get("beta")
    try:
        return None if b is None else float(b)
    except Exception:
        return None

def capm_expected_return(risk_free_rate: float, beta: float, market_return: float) -> float:
    """
    CAPM: expected return = rf + beta * (market_return - rf)
    Inputs are decimals (e.g., 0.03 for 3%)
    """
    return float(risk_free_rate + beta * (market_return - risk_free_rate))

def gordon_growth_value(D0: float, g: float, r: float, use_D1: bool = True) -> float:
    """
    Gordon Growth Model:
    If use_D1 True: D1 = D0 * (1 + g) used normally, value = D1 / (r - g)
    If use_D1 False: value = D0 / (r - g) (if D0 is next-period dividend already)
    """
    if r <= g:
        raise ValueError("Discount rate r must be greater than growth rate g for a finite value.")
    D1 = D0 * (1 + g) if use_D1 else D0
    return float(D1 / (r - g))

def dcf_valuation(cashflows: List[float], r: float, terminal_growth: float = 0.02) -> float:
    """
    Simple DCF valuation per share using explicit cashflows for years 1..N and a terminal value (perpetuity).
    Terminal value (TV) at year N = CF_N * (1 + terminal_growth) / (r - terminal_growth)
    Returns present value (float)
    """
    if len(cashflows) == 0:
        raise ValueError("cashflows must contain at least one period.")
    if r <= terminal_growth:
        raise ValueError("Discount rate r must be greater than terminal growth rate for a finite terminal value.")
    pv = 0.0
    for t, cf in enumerate(cashflows, start=1):
        pv += cf / ((1 + r) ** t)
    last_cf = cashflows[-1]
    tv = last_cf * (1 + terminal_growth) / (r - terminal_growth)
    pv += tv / ((1 + r) ** len(cashflows))
    return float(pv)