import builtins
import pandas as pd
import numpy as np
import types
import pytest

import capm_valuation as cv

# --- Unit tests for pure functions ---

def test_capm_expected_return_basic():
    rf = 0.02
    beta = 1.5
    market = 0.08
    expected = rf + beta * (market - rf)
    assert cv.capm_expected_return(rf, beta, market) == pytest.approx(expected)

def test_gordon_growth_value_and_errors():
    D0 = 1.0
    g = 0.03
    r = 0.08
    v = cv.gordon_growth_value(D0, g, r, use_D1=True)
    assert v == pytest.approx((D0 * (1 + g)) / (r - g))
    with pytest.raises(ValueError):
        cv.gordon_growth_value(D0, 0.05, 0.04)  # r <= g should raise

def test_dcf_valuation_basic_and_errors():
    cashflows = [1.0, 1.1, 1.21]
    r = 0.1
    tg = 0.02
    v = cv.dcf_valuation(cashflows, r, terminal_growth=tg)
    # compute expected manually
    pv = sum(cashflows[i] / ((1 + r) ** (i + 1)) for i in range(len(cashflows)))
    tv = cashflows[-1] * (1 + tg) / (r - tg)
    pv += tv / ((1 + r) ** len(cashflows))
    assert v == pytest.approx(pv)
    with pytest.raises(ValueError):
        cv.dcf_valuation([], r, tg)
    with pytest.raises(ValueError):
        cv.dcf_valuation(cashflows, 0.01, terminal_growth=0.02)  # r <= terminal_growth

# --- Tests that mock yfinance behavior for deterministic control ---

class DummyTicker:
    def __init__(self, info):
        self.info = info

def test_get_beta_from_yahoo_monkeypatch(monkeypatch):
    dummy = DummyTicker({"beta": "1.23"})
    monkeypatch.setattr(cv, "yf", types.SimpleNamespace(Ticker=lambda t: dummy))
    assert cv.get_beta_from_yahoo("FAKE") == 1.23

    dummy2 = DummyTicker({})
    monkeypatch.setattr(cv, "yf", types.SimpleNamespace(Ticker=lambda t: dummy2))
    assert cv.get_beta_from_yahoo("FAKE2") is None

def make_price_df(symbol, dates, prices):
    df = pd.DataFrame({"Adj Close": prices}, index=pd.to_datetime(dates))
    return df

def test_compute_beta_from_prices_monkeypatch(monkeypatch):
    # Create small synthetic price series (monthly)
    dates = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"]
    # stock moves 1.5x market plus small noise
    market_prices = [100, 102, 98, 105, 107, 110]
    stock_prices = [50, 53, 47, 59, 62, 65]  # approx 1.5x returns
    def fake_download(ticker, start, end, interval, progress):
        if ticker == "^GSPC":
            return make_price_df(ticker, dates, market_prices)
        return make_price_df(ticker, dates, stock_prices)
    monkeypatch.setattr(cv, "yf", types.SimpleNamespace(download=fake_download, Ticker=lambda t: DummyTicker({})))
    beta = cv.compute_beta_from_prices("FAKE", market_ticker="^GSPC", start="2020-01-01", freq="1mo")
    assert isinstance(beta, float)
    # Beta should be positive and reasonably > 0.5 for this synthetic series
    assert beta > 0.5