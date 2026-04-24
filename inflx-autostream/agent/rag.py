"""
rag.py
------
RAG (Retrieval-Augmented Generation) pipeline for AutoStream.

Loads the local JSON knowledge base and converts it into a
formatted context string injected into every LLM prompt.

In a production system this would use vector embeddings + FAISS/Chroma
for semantic search over thousands of documents. For this assignment,
the KB is small enough that full-context injection is accurate and efficient.
"""

import json
from pathlib import Path
from functools import lru_cache


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


@lru_cache(maxsize=1)
def load_knowledge_base() -> dict:
    """Load and cache the JSON knowledge base from disk."""
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {KB_PATH}")
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def format_kb_as_context() -> str:
    """
    Convert the JSON knowledge base into a clean plain-text context block
    that the LLM can reason over accurately.

    Returns:
        str: Formatted knowledge base ready to inject into system prompt
    """
    kb = load_knowledge_base()
    sections = []

    # --- Company Overview ---
    c = kb["company"]
    sections.append(
        f"## COMPANY OVERVIEW\n"
        f"Name: {c['name']}\n"
        f"Description: {c['description']}\n"
        f"Tagline: {c['tagline']}"
    )

    # --- Pricing Plans ---
    plans_text = "## PRICING PLANS\n"
    for plan in kb["plans"]:
        features = "\n  - ".join(plan["features"])
        plans_text += (
            f"\n### {plan['name']}\n"
            f"  Monthly Price: ${plan['price_monthly']}/month\n"
            f"  Yearly Price : ${plan['price_yearly']}/year (save {round((1 - plan['price_yearly']/(plan['price_monthly']*12))*100)}%)\n"
            f"  Features:\n  - {features}\n"
            f"  Best for: {plan['ideal_for']}\n"
        )
    sections.append(plans_text)

    # --- Policies ---
    policy_text = "## COMPANY POLICIES\n"
    for p in kb["policies"]:
        policy_text += f"\n- {p['policy']}: {p['detail']}"
    sections.append(policy_text)

    # --- FAQs ---
    faq_text = "## FREQUENTLY ASKED QUESTIONS\n"
    for faq in kb["faqs"]:
        faq_text += f"\nQ: {faq['question']}\nA: {faq['answer']}\n"
    sections.append(faq_text)

    return "\n\n".join(sections)


def retrieve(query: str) -> str:
    """
    Retrieve relevant knowledge for a given user query.

    For this assignment, returns the full formatted KB.
    In production, this would do semantic similarity search
    and return only the top-k most relevant chunks.

    Args:
        query (str): User's question or message

    Returns:
        str: Relevant knowledge context
    """
    # Full-context retrieval (appropriate for small KB)
    return format_kb_as_context()


# Pre-load KB at import time so first request is fast
KNOWLEDGE_CONTEXT: str = format_kb_as_context()