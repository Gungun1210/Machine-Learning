"""
state.py
--------
LangGraph AgentState definition for the AutoStream conversational agent.

State is the single source of truth passed between every node in the graph.
LangGraph uses TypedDict + Annotated to manage how state fields are updated.
"""

from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Complete state carried across every turn of the conversation.

    Fields:
        messages        : Full conversation history (HumanMessage + AIMessage).
                          Uses LangGraph's `add_messages` reducer — new messages
                          are APPENDED, not replaced, ensuring multi-turn memory.

        intent          : Last detected intent. One of:
                          GREETING | PRODUCT_INQUIRY | HIGH_INTENT | LEAD_INFO | OTHER

        collecting_lead : True once user has shown HIGH_INTENT.
                          Keeps the agent in lead-collection mode across turns.

        lead_name       : Extracted prospect name (None until collected)
        lead_email      : Extracted prospect email (None until collected)
        lead_platform   : Extracted creator platform (None until collected)

        lead_captured   : True after mock_lead_capture() has been fired.
                          Prevents duplicate tool calls.

        last_response   : The agent's most recent reply text.
                          Used by the Flask API to return just the new message.

        capture_result  : Payload returned by mock_lead_capture() on success.
    """

    messages: Annotated[List, add_messages]

    intent: str
    collecting_lead: bool

    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool

    last_response: str
    capture_result: Optional[dict]


def initial_state() -> AgentState:
    """Return a clean initial AgentState for a new conversation session."""
    return AgentState(
        messages=[],
        intent="",
        collecting_lead=False,
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        last_response="",
        capture_result=None,
    )