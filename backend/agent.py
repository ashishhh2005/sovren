"""
Sovren agent layer — the LLM decides whether to ANSWER from the knowledge base
or CALL A TOOL to take an action. This is the "agentic" part.

How it works (standard tool-calling loop):
  1. We describe our tools to the model and send the user's message.
  2. The model either replies directly, or asks to call a tool (e.g. raise_ticket).
  3. We run the tool, hand the result back, and the model writes the final answer.
  4. Every retrieval and tool call is recorded in a 'trace' — the traceable layer.
"""

import json
import os

# --- The tools the agent can use ---------------------------------------------
# Each is a plain Python function. In a real system these hit a ticketing API,
# a database, etc. Here they're simple stand-ins so the loop is easy to follow.

TICKETS: list[dict] = []  # in-memory store so you can see tickets accumulate


def raise_ticket(summary: str, category: str = "general") -> dict:
    """Create an IT/HR support ticket."""
    ticket = {"id": len(TICKETS) + 1, "summary": summary, "category": category}
    TICKETS.append(ticket)
    return {"created": ticket}


def check_leave_balance(employee: str) -> dict:
    """Look up remaining leave for an employee (hard-coded sample data)."""
    balances = {"ashish": 12, "priya": 7}
    return {"employee": employee, "days_left": balances.get(employee.lower(), 18)}


# Map tool name -> (function, JSON schema the LLM sees).
TOOLS = {
    "raise_ticket": (
        raise_ticket,
        {
            "type": "function",
            "function": {
                "name": "raise_ticket",
                "description": "Create an IT or HR support ticket when the user needs an action taken.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        },
    ),
    "check_leave_balance": (
        check_leave_balance,
        {
            "type": "function",
            "function": {
                "name": "check_leave_balance",
                "description": "Look up how many leave days an employee has left.",
                "parameters": {
                    "type": "object",
                    "properties": {"employee": {"type": "string"}},
                    "required": ["employee"],
                },
            },
        },
    ),
}


def run_agent(question: str, retrieved: list[dict]) -> dict:
    """
    Run one agent turn. Returns the answer plus a 'trace' of what happened.
    Needs an LLM key; without one we can't do tool-calling, so we say so.
    """
    trace: list[dict] = [
        {"step": "retrieve", "sources": [r["source"] for r in retrieved]}
    ]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "(no LLM key set — set OPENAI_API_KEY to enable the agent)",
            "trace": trace,
        }

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sovren, an internal assistant. Answer from the provided "
                "policy context. Use a tool only when the user asks to take an action "
                "(raise a ticket, check a balance). Policy rule: never raise a ticket "
                "for a request that policy says is self-service — tell them to self-serve."
            ),
        },
        {"role": "user", "content": f"Policy context:\n{context}\n\nUser: {question}"},
    ]
    tool_schemas = [schema for (_, schema) in TOOLS.values()]

    # First call: let the model answer or request a tool.
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=tool_schemas
    )
    msg = resp.choices[0].message

    # If the model asked for tools, run them and record each in the trace.
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            fn, _ = TOOLS[call.function.name]
            args = json.loads(call.function.arguments)
            result = fn(**args)
            trace.append({"step": "tool", "name": call.function.name, "args": args, "result": result})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
        # Second call: model writes the final answer using the tool results.
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        msg = resp.choices[0].message

    trace.append({"step": "answer"})
    return {"answer": msg.content, "trace": trace}
