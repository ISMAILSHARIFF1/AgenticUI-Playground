from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.playground import Playground, serve_playground_app

from dotenv import load_dotenv

load_dotenv()

web_search_agent = Agent(
    name="Web Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instrcutions="Always include sources",
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

#web_search_agent.print_response("What are top technologies for 2025?",stream=True)

finance_agent=Agent(
    name="Finance Agent",
    description="Your task is to find the finance information",
    model=Groq(id="llama-3.3-70b-versatile"),
    #model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True,company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)

#finance_agent.print_response("Summarize aanalyst recommendations for NVDA", stream=True)

app = Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)