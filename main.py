import asyncio

from quant_tools import (
    calculate_statistics,
    calculate_volatility,
    calculate_sharpe,
    calculate_drawdown,
    calculate_autocorrelation,
)

from rag import search_research


# -------------------------------------------------
# Quantitative Analysis
# -------------------------------------------------

def run_quant_analysis(ticker: str):
    """
    Run the quantitative analysis tools
    for a given ticker.
    """

    print("\n" + "=" * 70)
    print(f"QUANTITATIVE ANALYSIS: {ticker}")
    print("=" * 70)

    statistics = calculate_statistics(ticker)
    volatility = calculate_volatility(ticker)
    sharpe = calculate_sharpe(ticker)
    drawdown = calculate_drawdown(ticker)
    autocorrelation = calculate_autocorrelation(ticker)

    print("\nStatistics:")
    print(statistics)

    print("\nAnnualized Volatility:")
    print(volatility)

    print("\nSharpe Ratio:")
    print(sharpe)

    print("\nMaximum Drawdown:")
    print(drawdown)

    print("\nReturn Autocorrelation:")
    print(autocorrelation)


# -------------------------------------------------
# Research Analysis
# -------------------------------------------------

def run_research_search(query: str):
    """
    Search academic research related to a
    quantitative finance question.
    """

    print("\n" + "=" * 70)
    print("ACADEMIC RESEARCH")
    print("=" * 70)

    try:

        results = search_research(
            query=query,
            limit=3,
        )

        print(results)

    except RuntimeError as e:

        print("\nResearch search unavailable:")
        print(e)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    ticker = "AAPL"

    research_question = (
        "regime switching volatility "
        "models quantitative finance"
    )

    # ---------------------------------------------
    # Step 1: Quantitative analysis
    # ---------------------------------------------

    run_quant_analysis(ticker)

    # ---------------------------------------------
    # Step 2: Academic research
    # ---------------------------------------------

    run_research_search(research_question)


if __name__ == "__main__":
    main()