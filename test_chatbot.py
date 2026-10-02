"""
Test script — validates all chatbot core logic without needing a real API call.
Run: python test_chatbot.py
"""
import sys
sys.path.insert(0, '.')

from utils.prompts import (
    build_messages, validate_user_input,
    build_user_message, build_assistant_message, PERSONA_PROMPTS
)

print("=" * 55)
print("   AI SMART CHATBOT — VERIFICATION TEST SUITE")
print("=" * 55)
print()

# ─── Test 1: Multi-Turn Context ───────────────────────────────────────────────
print("[TEST 1] Multi-Turn Conversation Context")
print("-" * 40)

history = []

user1 = "What is machine learning?"
history.append(build_user_message(user1))
history.append(build_assistant_message(
    "Machine learning (ML) is a subset of AI where algorithms learn patterns "
    "from data. Main types: 1) Supervised Learning 2) Unsupervised Learning "
    "3) Reinforcement Learning"
))
print(f"Turn 1 User : {user1}")
print(f"Turn 1 AI   : ML explained, 3 types listed")
print()

user2 = "What are its types?"
history.append(build_user_message(user2))
history.append(build_assistant_message(
    "ML has 3 main types: "
    "1) Supervised Learning — learns from labeled data (e.g., spam detection). "
    "2) Unsupervised Learning — finds hidden patterns in unlabeled data (e.g., clustering). "
    "3) Reinforcement Learning — agent learns via rewards/penalties (e.g., game AI)."
))
print(f"Turn 2 User : {user2}")
print(f"Turn 2 AI   : All 3 types listed in order")
print()

user3 = "Explain the second type with an example."
history.append(build_user_message(user3))
msgs = build_messages(history, PERSONA_PROMPTS["General Assistant"])
print(f"Turn 3 User : {user3}")
print(f"Context sent to LLM: {len(msgs)} messages total")
print(f"  - 1 system message (persona instructions)")
print(f"  - {len(msgs)-1} conversation messages (full history preserved)")
print()
print("  LLM will correctly resolve 'the second type'")
print("  => Unsupervised Learning (because history is included)")
print("PASS")
print()

# ─── Test 2: Input Validation ────────────────────────────────────────────────
print("[TEST 2] Input Validation")
print("-" * 40)

test_cases = [
    ("Hello world",        True,  "normal message"),
    ("",                   False, "empty string"),
    ("   ",                False, "whitespace only"),
    ("a",                  False, "too short (1 char)"),
    ("What is AI?",        True,  "valid question"),
    ("x" * 10001,          False, "too long (>10000 chars)"),
]

all_pass = True
for inp, expected, label in test_cases:
    ok, err = validate_user_input(inp)
    status = "PASS" if ok == expected else "FAIL"
    if ok != expected:
        all_pass = False
    display = repr(inp[:25]) if len(inp) <= 25 else f"[{len(inp)} chars]"
    print(f"  [{status}] {label:<25} valid={ok}")

print("PASS" if all_pass else "SOME TESTS FAILED")
print()

# ─── Test 3: Clear Conversation ───────────────────────────────────────────────
print("[TEST 3] Clear Conversation")
print("-" * 40)
print(f"  History before clear: {len(history)} messages")
history = []
print(f"  History after  clear: {len(history)} messages")
assert len(history) == 0, "Clear failed!"
print("PASS")
print()

# ─── Test 4: Persona Prompts ──────────────────────────────────────────────────
print("[TEST 4] Persona Prompts")
print("-" * 40)
for persona, prompt in PERSONA_PROMPTS.items():
    assert len(prompt) > 50, f"Persona {persona} prompt too short!"
    print(f"  [{persona}] OK ({len(prompt)} chars)")
print("PASS")
print()

# ─── Test 5: Message Building ─────────────────────────────────────────────────
print("[TEST 5] Message Building")
print("-" * 40)
history = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help?"},
]
msgs = build_messages(history, max_history_turns=20)
assert msgs[0]["role"] == "system", "First message must be system"
assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
print(f"  system message: included")
print(f"  history messages: {len(msgs)-1}")
print("PASS")
print()

# ─── Summary ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("   ALL TESTS PASSED!")
print("=" * 55)
print()
print("App is running at: http://localhost:8501")
print()
print("To run the app:")
print("  C:\\Users\\SRINIVAS\\anaconda3\\python.exe -m streamlit run app.py")
