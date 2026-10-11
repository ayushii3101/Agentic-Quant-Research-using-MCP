import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch


def get_stock_data(ticker: str, period: str = "2y") -> dict:
    """Download historical stock data."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {
            "ticker": ticker,
            "error": "No data found."
        }

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()

    return {
        "ticker": ticker,
        "period": period,
        "rows": len(data),
        "start_date": str(data.index[0].date()),
        "end_date": str(data.index[-1].date()),
        "latest_price": float(data["Close"].iloc[-1]),
    }


def calculate_returns(ticker: str, period: str = "2y") -> dict:
    """Calculate daily simple and logarithmic returns."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()

    simple_returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    return {
        "ticker": ticker,
        "period": period,
        "observations": len(simple_returns),
        "mean_daily_return": float(simple_returns.mean()),
        "mean_daily_log_return": float(log_returns.mean()),
        "total_return": float((prices.iloc[-1] / prices.iloc[0]) - 1),
    }


def calculate_statistics(ticker: str, period: str = "2y") -> dict:
    """Calculate statistical properties of daily returns."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    return {
        "ticker": ticker,
        "mean": float(returns.mean()),
        "median": float(returns.median()),
        "std": float(returns.std()),
        "skewness": float(stats.skew(returns)),
        "kurtosis": float(stats.kurtosis(returns)),
        "minimum": float(returns.min()),
        "maximum": float(returns.max()),
    }


def calculate_volatility(
    ticker: str,
    period: str = "2y",
    annualization_factor: int = 252
) -> dict:
    """Calculate annualized historical volatility."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(
        annualization_factor
    )

    return {
        "ticker": ticker,
        "period": period,
        "daily_volatility": float(daily_volatility),
        "annualized_volatility": float(annualized_volatility),
    }


def calculate_sharpe(
    ticker: str,
    period: str = "2y",
    risk_free_rate: float = 0.0
) -> dict:
    """Calculate annualized Sharpe ratio."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    if excess_returns.std() == 0:
        return {
            "ticker": ticker,
            "sharpe_ratio": None,
            "error": "Standard deviation is zero."
        }

    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(252)
    )

    return {
        "ticker": ticker,
        "period": period,
        "risk_free_rate": risk_free_rate,
        "sharpe_ratio": float(sharpe),
    }


def calculate_drawdown(ticker: str, period: str = "2y") -> dict:
    """Calculate maximum drawdown."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()

    running_max = prices.cummax()
    drawdown = (prices / running_max) - 1

    max_drawdown = drawdown.min()

    return {
        "ticker": ticker,
        "period": period,
        "maximum_drawdown": float(max_drawdown),
    }


def calculate_autocorrelation(
    ticker: str,
    period: str = "2y",
    lag: int = 1
) -> dict:
    """Calculate autocorrelation of daily returns."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    autocorrelation = returns.autocorr(lag=lag)

    return {
        "ticker": ticker,
        "period": period,
        "lag": lag,
        "autocorrelation": float(autocorrelation),
    }


def run_adf_test(
    ticker: str,
    period: str = "2y"
) -> dict:
    """Run Augmented Dickey-Fuller test on daily prices."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()

    result = adfuller(prices)

    return {
        "ticker": ticker,
        "period": period,
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "lags_used": int(result[2]),
        "observations": int(result[3]),
    }


def run_kpss_test(
    ticker: str,
    period: str = "2y"
) -> dict:
    """Run KPSS test on daily prices."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()

    statistic, p_value, lags, critical_values = kpss(
        prices,
        regression="c",
        nlags="auto"
    )

    return {
        "ticker": ticker,
        "period": period,
        "kpss_statistic": float(statistic),
        "p_value": float(p_value),
        "lags_used": int(lags),
        "critical_values": {
            key: float(value)
            for key, value in critical_values.items()
        },
    }


def run_arch_test(
    ticker: str,
    period: str = "2y",
    lags: int = 10
) -> dict:
    """Run Engle's ARCH test on daily returns."""

    data = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return {"error": "No data found."}

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    prices = data["Close"].dropna()
    returns = prices.pct_change().dropna()

    statistic, p_value, _, _ = het_arch(
        returns,
        nlags=lags
    )

    return {
        "ticker": ticker,
        "period": period,
        "lags": lags,
        "arch_statistic": float(statistic),
        "p_value": float(p_value),
    }