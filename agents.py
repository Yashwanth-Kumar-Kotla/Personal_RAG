from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import create_agent


load_dotenv()

search_tool = DuckDuckGoSearchRun()

llm = ChatOpenAI()

agent = create_agent(
    model = llm,
    tools=[search_tool],
    system_prompt=
    '''You are a ReAct-style agent.

    When solving problems:
    1. Think about what information is needed.
    2. Use tools whenever necessary.
    3. Observe tool outputs.
    4. Continue until you can answer.
    5. Return only the final answer to the user.
    '''
)

response = agent.invoke({
    "messages" : [
        {
            "role" : "user",
            "content" : "what is the capital of france and what is its population?"
        }
    ]
}
)

print(response["messages"][-1].content)