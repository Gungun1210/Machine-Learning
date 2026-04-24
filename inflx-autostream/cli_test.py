"""
cli_test.py
-----------
Command-line tester for the AutoStream agent.
Runs a scripted conversation that covers ALL evaluation criteria:

  1. Greeting
  2. Product/pricing inquiry (RAG)
  3. High-intent detection
  4. Lead collection (name → email → platform)
  5. Tool execution (mock_lead_capture)

Usage:
    python cli_test.py                  # scripted demo
    python cli_test.py --interactive    # manual REPL
"""

import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from agent.graph import ConversationSession

RESET  = "\033[0m"
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"


def print_turn(turn: int, user_msg: str, result: dict):
    print(f"\n{'─'*60}")
    print(f"{BLUE}{BOLD}[Turn {turn}] USER:{RESET} {user_msg}")
    print(f"{YELLOW}  Intent  : {result['intent']}{RESET}")

    li = result['lead_info']
    print(f"{CYAN}  Lead    : name={li['name']} | email={li['email']} | platform={li['platform']}{RESET}")

    if result['lead_captured']:
        print(f"{GREEN}{BOLD}  ✅ TOOL FIRED: mock_lead_capture() — Lead captured!{RESET}")

    print(f"\n{GREEN}ARIA:{RESET} {result['response']}")


def run_scripted_demo():
    """Run a full scripted conversation to demonstrate all agent capabilities."""
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  AutoStream AI Agent — Scripted Demo{RESET}")
    print(f"  Covers: Greeting → RAG → High Intent → Lead Capture")
    print(f"{BOLD}{'='*60}{RESET}")

    session = ConversationSession()

    script = [
        "Hi! Tell me about AutoStream.",
        "What are your pricing plans? How does Basic compare to Pro?",
        "What's your refund policy?",
        "That sounds great! I want to sign up for the Pro plan for my YouTube channel.",
        "Ravi Sharma",
        "ravi.sharma@gmail.com",
        "YouTube",
    ]

    for i, msg in enumerate(script, 1):
        result = session.chat(msg)
        print_turn(i, msg, result)

    print(f"\n{'='*60}")
    print(f"{GREEN}{BOLD}  Demo complete. Lead captured: {session.state['lead_captured']}{RESET}")
    print(f"  Check leads.json for the saved lead record.")
    print(f"{'='*60}\n")


def run_interactive():
    """Interactive REPL for manual testing."""
    print(f"\n{BOLD}AutoStream Agent — Interactive Mode{RESET}")
    print("Type 'quit' to exit, 'reset' to start a new conversation.\n")

    session = ConversationSession()

    while True:
        try:
            user_input = input(f"{BLUE}You: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'reset':
            session.reset()
            print(f"{YELLOW}[Session reset]{RESET}\n")
            continue

        result = session.chat(user_input)
        print(f"\n{YELLOW}[Intent: {result['intent']}]{RESET}")

        if result['lead_captured']:
            print(f"{GREEN}[✅ Tool fired: mock_lead_capture()]{RESET}")

        print(f"{GREEN}Aria: {RESET}{result['response']}\n")


if __name__ == "__main__":
    if "--interactive" in sys.argv or "-i" in sys.argv:
        run_interactive()
    else:
        run_scripted_demo()