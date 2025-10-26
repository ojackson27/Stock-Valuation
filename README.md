
# Stock Valuation using CAPM (Python)

This repository contains a small Python module to compute CAPM expected returns and to value stocks using:
- Gordon Growth Model (for dividend-paying stocks)
- Simple DCF (explicit cash flows + terminal value)

Included:
- capm_valuation.py — core functions (fetch prices, compute beta, CAPM expected return, valuations)
- examples/capm_notebook.md — runnable step-by-step example (code cells you can paste into a Jupyter notebook)
- tests/test_capm_valuation.py — pytest unit tests (including mocks for yfinance)
- .github/workflows/ci.yml — GitHub Actions workflow to run tests

Requirements
- Python 3.8+
- pip install -r requirements.txt

Quick start
1. Install dependencies:
   pip install -r requirements.txt

2. Run tests:
   pytest -q

3. Try the example (open examples/capm_notebook.md and paste code cells into a Jupyter notebook or run them interactively).

Notes and cautions
- Ensure r > g for any model that uses a perpetuity (Gordon / terminal value).
- CAPM output depends on beta and your chosen market return and risk-free rate.
- The module uses `yfinance` for price data; data quality depends on Yahoo Finance availability.
```
