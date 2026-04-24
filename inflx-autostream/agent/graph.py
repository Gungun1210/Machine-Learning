"""
graph.py
--------
Builds and compiles the LangGraph StateGraph for the AutoStream agent.

Graph topology:
  [START]
     │
     ▼
  intent_node          ← classify user intent
     │
     ├─(LEAD_INFO or collecting)──► extract_node ─┐
     │                                             │
     │                                             ├─(all 3 fields ready)──► capture_node
     │                                             │                              │
     │                                             └─(fields missing)─────────────┤
     │                                                                            │
     └─(GREETING / PRODUCT_INQUIRY / OTHER)──────────────────────────────────────┤
                                                                                  │
                                                                                  ▼
                                                                           response_node
                                                                                  │
                                                                                [END]

State is retained across all turns via the AgentState TypedDict.
The `add_messages` reducer appends to history rather than replacing it,
giving the agent full multi-turn memory.
"""

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from agent.state import AgentState, initial_state
from agent.nodes import intent_node, extract_node, capture_node, response_node


# ─────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    After intent detection, decide next node:
    - If user is in lead collection flow OR intent is LEAD_INFO → extract fields
    - Otherwise → generate response directly
    """
    intent = state.get("intent", "OTHER")
    collecting = state.get("collecting_lead", False)

    if intent in ("LEAD_INFO", "HIGH_INTENT") or collecting:
        return "extract_node"
    return "response_node"


def route_after_extract(state: AgentState) -> str:
    """
    After extracting fields, decide next node:
    - All 3 fields present and tool not yet fired → trigger capture
    - Otherwise → generate response (ask for next missing field)
    """
    all_collected = (
        state.get("lead_name")
        and state.get("lead_email")
        and state.get("lead_platform")
    )
    already_captured = state.get("lead_captured", False)

    if all_collected and not already_captured:
        return "capture_node"
    return "response_node"


# ─────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct, wire, and compile the LangGraph StateGraph.

    Returns:
        CompiledGraph: Ready-to-invoke agent graph
    """
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("intent_node", intent_node)
    g.add_node("extract_node", extract_node)
    g.add_node("capture_node", capture_node)
    g.add_node("response_node", response_node)

    # Entry point
    g.set_entry_point("intent_node")

    # Conditional routing from intent detection
    g.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "extract_node": "extract_node",
            "response_node": "response_node",
        }
    )

    # Conditional routing from field extraction
    g.add_conditional_edges(
        "extract_node",
        route_after_extract,
        {
            "capture_node": "capture_node",
            "response_node": "response_node",
        }
    )

    # After capture, always generate response
    g.add_edge("capture_node", "response_node")

    # Response is always the terminal node
    g.add_edge("response_node", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────
# SESSION MANAGER
# ─────────────────────────────────────────────────────────────

class ConversationSession:
    """
    Wraps a compiled LangGraph agent and maintains per-user state
    across multiple conversation turns.

    Usage:
        session = ConversationSession()
        result  = session.chat("Hi, tell me about your pricing")
        result  = session.chat("I want the Pro plan for my YouTube channel")
    """

    def __init__(self):
        self.graph = build_graph()
        self.state: AgentState = initial_state()

    def chat(self, user_message: str) -> dict:
        """
        Process one user message through the full agent graph.

        Args:
            user_message (str): Raw text from the user

        Returns:
            dict: {
                response    : str   - Agent's reply
                intent      : str   - Detected intent label
                lead_info   : dict  - Collected lead fields
                lead_captured: bool - Whether tool was fired
                capture_result: dict - Tool result payload (if fired)
            }
        """
        # Append user message to history
        self.state["messages"] = list(self.state["messages"]) + [
            HumanMessage(content=user_message)
        ]

        # Run through the LangGraph pipeline
        result = self.graph.invoke(self.state)
        self.state = result

        return {
            "response": self.state["last_response"],
            "intent": self.state["intent"],
            "lead_captured": self.state["lead_captured"],
            "capture_result": self.state.get("capture_result"),
            "lead_info": {
                "name": self.state.get("lead_name"),
                "email": self.state.get("lead_email"),
                "platform": self.state.get("lead_platform"),
            },
        }

    def reset(self):
        """Clear all conversation state and start fresh."""
        self.state = initial_state()

    @property
    def message_count(self) -> int:
        return len(self.state.get("messages", []))