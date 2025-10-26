```python
# CAPM Valuation Example (paste into a Jupyter notebook cell-by-cell)

This is a short, runnable example showing:
- compute beta from prices (monthly)
- compute CAPM expected return
- compute Gordon Growth value and a simple DCF

Cell 1: Imports
```python
import pandas as pd
from capm_valuation import compute_beta_from_prices, get_beta_from_yahoo, capm_expected_return, gordon_growth_value, dcf_valuation
```

Cell 2: Parameters and beta retrieval
```python
ticker = "AAPL"
rf = 0.035  # 3.5%
market_return = 0.09  # 9%
# Try Yahoo beta first
beta = get_beta_from_yahoo(ticker)
if beta is None:
    beta = compute_beta_from_prices(ticker, market_ticker="^GSPC", start="2018-01-01", freq="1mo")
print("Beta:", beta)
r = capm_expected_return(rf, beta, market_return)
print("CAPM expected return:", r)
```

Cell 3: Gordon Growth
```python
D0 = 0.90
g = 0.03
try:
    gordon_val = gordon_growth_value(D0, g, r)
    print("Gordon Growth intrinsic value:", gordon_val)
except Exception as e:
    print("Gordon error:", e)
```

Cell 4: Simple DCF
```python
cashflows = [5.0, 5.5, 6.0, 6.6, 7.2]
terminal_growth = 0.02
try:
    dcf_val = dcf_valuation(cashflows, r, terminal_growth)
    print("DCF intrinsic value:", dcf_val)
except Exception as e:
    print("DCF error:", e)
```