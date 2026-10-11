import asyncio
import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# Load environment variables from .env
load_dotenv()


async def main():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )


    client = MultiServerMCPClient(
        {
            "quant": {
                "command": "python",
                "args": ["mcp_server.py"],
                "transport": "stdio",
            }
        }
    )


    # -------------------------------------------------
    # 3. Discover MCP tools
    # -------------------------------------------------

    tools = await client.get_tools()

    print("\nAvailable MCP tools:")
    
    for tool in tools:
        print(f"- {tool.name}")


    # -------------------------------------------------
    # 4. Create LangChain agent
    # -------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=tools,
    )


    # -------------------------------------------------
    # 5. Ask the agent a question
    # -------------------------------------------------

    question = """
    Analyze AAPL using the available quantitative tools.

    Calculate:
    - historical returns
    - annualized volatility
    - Sharpe ratio
    - maximum drawdown
    - return autocorrelation

    Explain the results clearly.
    """

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    print("\n" + "=" * 70)
    print("QUANT RESEARCH AGENT")
    print("=" * 70)

    final_message = result["messages"][-1]

    print(final_message.content)


if __name__ == "__main__":
    asyncio.run(main())