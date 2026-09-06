# Llama 3.2 + LangGraph Tool-Calling Agent

A minimal but complete tool-calling agent: Llama 3.2 runs locally via
[Ollama](https://ollama.com), and [LangGraph](https://langchain-ai.github.io/langgraph/)
orchestrates the agent ↔ tool loop.

## How it works

```
        ┌────────┐   has tool_calls?   ┌────────┐
 START →│ agent  │ ───────────────────→│ tools  │
        │ (LLM)  │←──────────────────── │(execute)│
        └────────┘    no tool_calls     └────────┘
             │
             ▼
            END
```

1. `agent` node sends the conversation to Llama 3.2 (bound with tool schemas).
2. If the model responds with one or more `tool_calls`, LangGraph routes to
   the `tools` node, which executes each tool and appends the result as a
   `ToolMessage`.
3. Control returns to `agent`, which sees the tool output and either calls
   another tool or gives a final answer.
4. Once the model replies with no tool calls, the graph ends.

## Setup

1. **Install Ollama** (runs Llama 3.2 locally): https://ollama.com/download

2. **Pull a tool-calling-capable model:**
   ```bash
   ollama pull llama3.2        # 3B parameters, supports tool calling
   ```
   (`llama3.2:1b` also works but is noticeably weaker at deciding when/how
   to call tools — use the 3B model if your hardware allows.)

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run:**
   ```bash
   python agent.py
   ```

## Example session

```
You: What's 2847 * 193, and what time is it?

[tool call] calculator({"expression": "2847 * 193"})
[tool result:calculator] 549471
[tool call] get_current_time({})
[tool result:get_current_time] 2026-09-06 14:32:07
[assistant] 2847 * 193 = 549,471. The current time is 2026-09-06 14:32:07.

Final answer: 2847 * 193 = 549,471. The current time is 2026-09-06 14:32:07.
```

## Included tools

| Tool | Purpose |
|---|---|
| `calculator` | Evaluates basic arithmetic expressions |
| `get_current_time` | Returns current date/time |
| `get_weather` | Mock weather lookup (swap in a real API for production) |

## Extending it

To add your own tool, write a function decorated with `@tool` in `agent.py`,
give it a clear docstring (the model uses this to decide when to call it),
and add it to the `TOOLS` list. LangGraph's `ToolNode` and `tools_condition`
handle execution and routing automatically — no other changes needed.

To swap in a different model, change:
```python
llm = ChatOllama(model="llama3.2", temperature=0)
```
to any other Ollama-served model name, or swap `ChatOllama` for another
LangChain chat model class (e.g. `ChatGroq`, `ChatOpenAI`) if you'd rather
call Llama 3.2 through a hosted API instead of running it locally.

## Notes on Llama 3.2 tool calling

- Tool calling works reliably on the **3B instruct model**; the 1B model
  is much less consistent about emitting correctly formatted tool calls.
- Ollama must be running (`ollama serve`, usually started automatically)
  and listening on `localhost:11434` (the `ChatOllama` default).
- If the model ignores a tool it should be using, tightening the tool's
  docstring or the system prompt usually helps more than raising
  `temperature`.
