prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.
"""

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

@tool
def add(x, y):
    """Return the sum of x and y"""
    return y - x

@tool
def product(x, y):
    """Return the product of x and y"""
    return x * y

agent = create_react_agent(llm, [add, product], state_modifier=prompt)

for s in agent.stream({"messages": [HumanMessage(content="Can you tell me the result of 1 + 2?")]}):
    if s.get("agent"):
        message = s["agent"]["messages"][-1]
    else:
        message = s["tools"]["messages"][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()


