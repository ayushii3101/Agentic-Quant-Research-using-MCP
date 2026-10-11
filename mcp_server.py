from mcp.server.fastmcp import FastMCP

from quant_tools import (
    get_stock_data,
    calculate_returns,
    calculate_statistics,
    calculate_volatility,
    calculate_sharpe,
    calculate_drawdown,
    calculate_autocorrelation,
    run_adf_test,
    run_kpss_test,
    run_arch_test,
)


# Create the MCP server
mcp = FastMCP("Quant Research Server")


# =========================
# Market Data
# =========================

@mcp.tool()
def stock_data(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Get historical stock data information.

    Args:
        ticker: Stock ticker symbol, e.g. AAPL, MSFT, NVDA.
        period: Historical period such as 1y, 2y, or 5y.
    """
    return get_stock_data(ticker, period)


# =========================
# Return Analysis
# =========================

@mcp.tool()
def stock_returns(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Calculate historical stock returns.
    """
    return calculate_returns(ticker, period)


# =========================
# Statistical Analysis
# =========================

@mcp.tool()
def stock_statistics(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Calculate statistical properties of stock returns.
    """
    return calculate_statistics(ticker, period)


# =========================
# Volatility
# =========================

@mcp.tool()
def stock_volatility(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Calculate daily and annualized historical volatility.
    """
    return calculate_volatility(ticker, period)


# =========================
# Sharpe Ratio
# =========================

@mcp.tool()
def stock_sharpe(
    ticker: str,
    period: str = "2y",
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate the annualized Sharpe ratio.
    """
    return calculate_sharpe(
        ticker,
        period,
        risk_free_rate
    )


# =========================
# Drawdown
# =========================

@mcp.tool()
def stock_drawdown(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Calculate maximum historical drawdown.
    """
    return calculate_drawdown(ticker, period)


# =========================
# Autocorrelation
# =========================

@mcp.tool()
def stock_autocorrelation(
    ticker: str,
    period: str = "2y",
    lag: int = 1
) -> dict:
    """
    Calculate autocorrelation of stock returns.
    """
    return calculate_autocorrelation(
        ticker,
        period,
        lag
    )


# =========================
# ADF Test
# =========================

@mcp.tool()
def adf_test(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Run the Augmented Dickey-Fuller test.
    """
    return run_adf_test(ticker, period)


# =========================
# KPSS Test
# =========================

@mcp.tool()
def kpss_test(
    ticker: str,
    period: str = "2y"
) -> dict:
    """
    Run the KPSS stationarity test.
    """
    return run_kpss_test(ticker, period)


# =========================
# ARCH Test
# =========================

@mcp.tool()
def arch_test(
    ticker: str,
    period: str = "2y",
    lags: int = 10
) -> dict:
    """
    Run Engle's ARCH test for volatility clustering.
    """
    return run_arch_test(
        ticker,
        period,
        lags
    )


# =========================
# Start MCP Server
# =========================

if __name__ == "__main__":
    mcp.run()