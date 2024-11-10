from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

ANTHROPIC = False
@tool
def multiply(x: float, y:float) -> float:
    return x*y

def toolcalling():
    print("Hello Tool Calling")
    llm = None
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you're a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearchResults(), multiply]
    if not ANTHROPIC:
        llm = ChatOpenAI(model="gpt-4-utrbo")
    else:
        llm =ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    res = agent_executor.invoke(
        {
            "input": "what is the weather in poland right now? compare it with Mexico, output should be in celsious",
        }
    )

    print(res)