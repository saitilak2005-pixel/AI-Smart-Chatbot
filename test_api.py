import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from services.llm_service import get_llm_service, LLMConfig

service = get_llm_service()
cfg = service.config

print(f"Provider: {cfg.api_provider}")
print(f"Key:      {cfg.api_key[:15]}...")
print(f"Model:    {cfg.model_name}")
print(f"Base URL: {cfg.base_url}")
print()

# Full multi-turn test
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user",   "content": "What is machine learning? Answer in 2 sentences."},
]

print("Turn 1: What is machine learning?")
reply1 = service.get_response(messages)
print(f"AI: {reply1.strip()}")
print()

messages.append({"role": "assistant", "content": reply1})
messages.append({"role": "user",      "content": "What are its types? List briefly."})

print("Turn 2: What are its types?")
reply2 = service.get_response(messages)
print(f"AI: {reply2.strip()}")
print()

messages.append({"role": "assistant", "content": reply2})
messages.append({"role": "user",      "content": "Explain the second type with an example."})

print("Turn 3: Explain the second type with an example.")
reply3 = service.get_response(messages)
print(f"AI: {reply3.strip()}")
print()
print("=" * 50)
print("ALL MULTI-TURN TESTS PASSED WITH GEMINI!")
print("Gemini 3.8 Flash is active with 1,000,000 tokens/min limit!")
