"""
Tool-calling AI agent using Llama 3.2 (served locally via Ollama) + LangGraph.

Architecture
------------
StateGraph with two nodes:
  - "agent": calls the LLM (bound with tools). The LLM decides whether to
             respond directly or request one or more tool calls.
  - "tools": executes any requested tool calls and feeds results back.

A conditional edge loops agent -> tools -> agent until the LLM returns a
plain answer with no further tool calls, at which point the graph ends.

Requirements
------------
1. Install Ollama: https://ollama.com/download
2. Pull a tool-calling-capable Llama 3.2 model:
       ollama pull llama3.2          # 3B, supports tool calling
       # or: ollama pull llama3.2:1b (smaller, weaker tool-calling)
3. pip install -r requirements.txt
4. python agent.py
"""

from datetime import datetime
import json

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------------------------------------------------------------------
# 1. Define tools
# ---------------------------------------------------------------------------
# Each @tool-decorated function becomes callable by the model. The docstring
# and type hints are turned into the tool's JSON schema, so write them
# carefully -- the model relies on them to decide when/how to call the tool.

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.

    Args:
        expression: A valid Python arithmetic expression using
            numbers and + - * / ( ) only.
    """
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data for demo purposes).

    Args:
        city: Name of the city, e.g. 'Paris' or 'Tokyo'.
    """
    # Replace this with a real API call (e.g. OpenWeatherMap) in production.
    mock_db = {
        "paris": "15°C, cloudy",
        "tokyo": "22°C, clear skies",
        "new york": "18°C, light rain",
    }
    return mock_db.get(city.lower(), f"No weather data available for {city}.")


TOOLS = [calculator, get_current_time, get_weather]


# ---------------------------------------------------------------------------
# 2. Set up the LLM
# ---------------------------------------------------------------------------
# ChatOllama talks to a locally running Ollama server (default localhost:11434).
# temperature=0 makes tool-calling decisions more deterministic.
llm = ChatOllama(model="llama3.2", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# 3. Define the graph
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools: calculator, "
    "get_current_time, and get_weather. Use a tool whenever it would give "
    "a more accurate or up-to-date answer than you could give from memory. "
    "If no tool is needed, answer directly and concisely."
)


def agent_node(state: MessagesState):
    """Call the LLM with the running message history."""
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(TOOLS))

graph_builder.add_edge(START, "agent")
# tools_condition inspects the last AIMessage: if it has tool_calls, route
# to "tools"; otherwise route to END.
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

agent = graph_builder.compile()


# ---------------------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------------------
def run_agent(user_input: str, verbose: bool = True):
    result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    if verbose:
        for msg in result["messages"]:
            role = msg.type
            if role == "ai" and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    print(f"[tool call] {call['name']}({json.dumps(call['args'])})")
            elif role == "tool":
                print(f"[tool result:{msg.name}] {msg.content}")
            elif role == "ai":
                print(f"[assistant] {msg.content}")
            elif role == "human":
                print(f"[user] {msg.content}")
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Llama 3.2 + LangGraph tool-calling agent. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        print()
        answer = run_agent(user_input)
        print(f"\nFinal answer: {answer}\n")
