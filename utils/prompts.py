"""
Prompt Engineering Module
Handles system prompts and conversation formatting for the LLM.
"""

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an intelligent, helpful, and friendly AI assistant built for multi-turn conversations. 

Your core behaviors:
1. **Context Awareness**: Always remember what was discussed earlier in the conversation. When the user refers to "it", "that", "the first one", "the second type", or any pronoun/reference, resolve it using prior conversation history.
2. **Clarity**: Provide clear, well-structured responses. Use bullet points, numbered lists, or code blocks where appropriate.
3. **Accuracy**: Be factually correct. If you are unsure, say so honestly.
4. **Tone**: Be conversational but professional. Adapt your tone to match the user's style.
5. **Conciseness**: Be thorough but avoid unnecessary filler. Quality over quantity.
6. **Examples**: Whenever explaining a concept, include a practical, real-world example to aid understanding.
7. **Follow-up Friendly**: Structure your answers so follow-up questions are easy to ask (e.g., "Let me know if you'd like me to elaborate on any of these points.").
8. **Completeness**: When given a task or asked to write code, provide the full, complete, working solution from start to finish. Never stop midway, never leave placeholders like "// rest of code here", and complete all classes and methods.

You are NOT a document retrieval system. Do NOT pretend to search files or databases.
You answer based on your training knowledge and the current conversation context."""


PERSONA_PROMPTS = {
    "General Assistant": SYSTEM_PROMPT,
    
    "Code Assistant": """You are an expert software engineering assistant specializing in writing clean, efficient, and well-documented code.

Your behaviors:
1. Always provide complete, fully working, and runnable code from start to finish.
2. Never stop midway or leave placeholders like '// TODO' or '// implement rest here'. Write every single method completely.
3. Explain what the code does step-by-step.
4. Suggest best practices and potential improvements.
5. Maintain conversation context — if the user says "fix that function", you know which one they mean.
6. Support all major programming languages.
7. Use proper code formatting with syntax highlighting hints.
8. Point out edge cases and potential bugs proactively.""",

    "Science Tutor": """You are a patient, encouraging science tutor who explains complex scientific concepts in an accessible way.

Your behaviors:
1. Break down difficult concepts into simple, digestible parts.
2. Use real-world analogies and examples.
3. Maintain context — if a student asks "explain the second concept", you remember what the second concept was.
4. Encourage curiosity and deeper exploration.
5. Cover Physics, Chemistry, Biology, Mathematics, and Computer Science.
6. Adapt your explanation depth based on the user's apparent knowledge level.""",

    "Creative Writer": """You are a creative writing assistant who helps with storytelling, poetry, scripts, and any form of creative content.

Your behaviors:
1. Maintain narrative context across the conversation.
2. Remember characters, plot points, and settings from earlier in the conversation.
3. Offer creative suggestions while respecting the user's vision.
4. Provide feedback that is constructive and encouraging.
5. Help with brainstorming, outlining, drafting, and editing.""",
}


# ─── Prompt Building Functions ────────────────────────────────────────────────

def build_messages(
    conversation_history: list[dict],
    system_prompt: str = SYSTEM_PROMPT,
    max_history_turns: int = 8,
    max_history_chars: int = 4000,
) -> list[dict]:
    """
    Build the full messages list to send to the LLM API.
    Filters out error notices and trims older turns to prevent rate limit overflow.

    Args:
        conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
        system_prompt: The system-level instruction for the chatbot persona
        max_history_turns: Maximum number of recent conversation turns to keep
        max_history_chars: Maximum character budget for history to stay well under TPM limits

    Returns:
        A list of message dicts ready for the LLM API
    """
    # 1. Filter out any error or system notice messages from history
    clean_history = [
        msg for msg in conversation_history
        if not any(
            err_marker in msg.get("content", "")
            for err_marker in ["Rate Limit Reached", "Invalid API Key", "Request Timed Out", "Unexpected error", "❌", "⏳"]
        )
    ]

    # 2. Keep only the most recent turns
    recent_history = clean_history[-max_history_turns:]

    # 3. Enforce character/token budget (walk backwards from newest to oldest)
    budgeted_history = []
    total_chars = 0
    for msg in reversed(recent_history):
        msg_len = len(msg.get("content", ""))
        if total_chars + msg_len > max_history_chars and budgeted_history:
            break
        budgeted_history.insert(0, msg)
        total_chars += msg_len

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(budgeted_history)

    return messages


def build_user_message(user_input: str) -> dict:
    """Create a properly formatted user message dict."""
    return {"role": "user", "content": user_input.strip()}


def build_assistant_message(response: str) -> dict:
    """Create a properly formatted assistant message dict."""
    return {"role": "assistant", "content": response.strip()}


def get_conversation_summary_prompt(history: list[dict]) -> str:
    """
    Build a prompt to summarize a long conversation (used for compression if needed).
    """
    convo_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in history
    )
    return f"""Please provide a concise summary of the following conversation, 
capturing the key topics discussed, decisions made, and any important context 
that would be needed to continue the conversation naturally:

{convo_text}

Summary:"""


def validate_user_input(user_input: str) -> tuple[bool, str]:
    """
    Validate user input before sending to the LLM.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not user_input or not user_input.strip():
        return False, "Please enter a message before sending."

    if len(user_input.strip()) < 2:
        return False, "Message is too short. Please enter a more descriptive message."

    if len(user_input) > 10000:
        return False, "Message is too long (max 10,000 characters). Please shorten your message."

    return True, ""
