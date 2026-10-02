"""
AI-Powered Smart Chatbot
========================
Main Streamlit application — Week 4 Generative AI Internship Project.

Architecture:
  User → Streamlit Chat UI → Session State / Conversation History
       → Prompt Builder → LLM API → AI Response
       → Update Conversation History → Display Response
"""

import streamlit as st
import time
from datetime import datetime

# Internal modules
from services.llm_service import get_llm_service, LLMService, LLMServiceError, LLMConfig
from utils.prompts import (
    PERSONA_PROMPTS,
    build_messages,
    build_user_message,
    build_assistant_message,
    validate_user_input,
)

# ─── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Smart Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Root Design System ── */
  :root {
    --primary:        #7C3AED;
    --primary-light:  #A78BFA;
    --primary-glow:   rgba(124, 58, 237, 0.4);
    --accent:         #06B6D4;
    --accent-glow:    rgba(6, 182, 212, 0.4);
    --bg-dark:        #0D0B18;
    --surface-1:      #161226;
    --surface-2:      #1E1935;
    --surface-3:      #2A234A;
    --text-primary:   #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted:     #64748B;
    --border-subtle:  rgba(124, 58, 237, 0.25);
    --border-bright:  rgba(6, 182, 212, 0.5);
    --success:        #10B981;
  }

  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
  }

  /* ── Main Container ── */
  .main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 6.5rem !important;
    max-width: 920px !important;
  }

  /* ── Ambient Background Glow ── */
  .ambient-glow {
    position: relative;
    width: 100%;
    height: 0;
  }
  .ambient-glow::before {
    content: '';
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    width: 500px;
    height: 180px;
    background: radial-gradient(ellipse at center, rgba(124, 58, 237, 0.3) 0%, rgba(6, 182, 212, 0.15) 45%, transparent 70%);
    filter: blur(50px);
    pointer-events: none;
    z-index: 0;
  }

  /* ── Header Branding ── */
  .chat-header {
    text-align: center;
    padding: 1rem 0 0.5rem;
    position: relative;
    z-index: 1;
  }
  .chat-header h1 {
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #FFFFFF 0%, #C4B5FD 50%, #06B6D4 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.3rem !important;
    animation: titleGlow 4s ease-in-out infinite alternate;
  }
  @keyframes titleGlow {
    0% { filter: drop-shadow(0 0 16px rgba(124, 58, 237, 0.3)); }
    100% { filter: drop-shadow(0 0 28px rgba(6, 182, 212, 0.5)); }
  }
  .chat-header p {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    font-weight: 400;
  }

  /* ── Persona Badge ── */
  .persona-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 1.1rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
  }
  .persona-pill:hover {
    transform: translateY(-1px) scale(1.03);
  }

  /* ── Welcome Hero ── */
  .welcome-hero {
    background: linear-gradient(135deg, rgba(30, 25, 53, 0.85) 0%, rgba(22, 18, 38, 0.85) 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: 24px;
    padding: 2.25rem 2rem;
    text-align: center;
    margin: 1.25rem 0 1.5rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    animation: heroEntrance 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  @keyframes heroEntrance {
    0% { opacity: 0; transform: translateY(16px) scale(0.98); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
  }
  .welcome-hero::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.04), transparent);
    animation: shineSweep 8s infinite linear;
  }
  @keyframes shineSweep {
    0% { transform: translateX(0); }
    100% { transform: translateX(100%); }
  }
  .hero-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.2);
    border: 1px solid rgba(124, 58, 237, 0.4);
    color: var(--primary-light);
    border-radius: 9999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.85rem;
  }
  .hero-title {
    color: var(--text-primary) !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
  }
  .hero-sub {
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.5;
  }

  /* ── Section Label ── */
  .section-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin: 1.5rem 0 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-subtle), transparent);
  }

  /* ── Recommended Prompt Buttons (Card Style & Micro-Motions) ── */
  div[data-testid="column"] .stButton > button {
    background: linear-gradient(145deg, rgba(28, 22, 54, 0.85), rgba(18, 14, 36, 0.85)) !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 18px !important;
    color: #F8FAFC !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    padding: 1.15rem 1.25rem !important;
    min-height: 96px !important;
    text-align: left !important;
    white-space: pre-wrap !important;
    line-height: 1.45 !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px) !important;
    position: relative !important;
    overflow: hidden !important;
  }
  div[data-testid="column"] .stButton > button:hover {
    transform: translateY(-6px) scale(1.025) !important;
    border-color: #06B6D4 !important;
    box-shadow: 0 16px 36px -6px rgba(124, 58, 237, 0.55), 0 0 24px rgba(6, 182, 212, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    background: linear-gradient(145deg, rgba(46, 36, 85, 0.95), rgba(28, 22, 54, 0.95)) !important;
    color: #FFFFFF !important;
  }
  /* WHEN USER CLICKS RECOMMENDED PROMPT (Tactile Spring Bounce) */
  div[data-testid="column"] .stButton > button:active {
    transform: translateY(2px) scale(0.96) !important;
    box-shadow: 0 2px 10px rgba(6, 182, 212, 0.6), inset 0 0 18px rgba(124, 58, 237, 0.5) !important;
    border-color: #A78BFA !important;
    transition: all 0.08s ease-in-out !important;
  }

  /* ── Chat Messages (Fluid Entry Motion) ── */
  [data-testid="stChatMessage"] {
    background: rgba(30, 25, 53, 0.75) !important;
    border: 1px solid rgba(124, 58, 237, 0.28) !important;
    border-radius: 20px !important;
    padding: 1.15rem 1.25rem !important;
    margin-bottom: 1.1rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    animation: messageSlideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  }
  @keyframes messageSlideIn {
    0% {
      opacity: 0;
      transform: translateY(14px) scale(0.985);
    }
    100% {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  /* User Message Distinction */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
  [data-testid="stChatMessage"]:has([aria-label="chat message from user"]) {
    background: linear-gradient(135deg, rgba(61, 35, 115, 0.75) 0%, rgba(40, 22, 80, 0.75) 100%) !important;
    border: 1px solid rgba(167, 139, 250, 0.45) !important;
    border-right: 3px solid #A78BFA !important;
  }

  /* Assistant Message Distinction */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
  [data-testid="stChatMessage"]:has([aria-label="chat message from assistant"]) {
    background: linear-gradient(135deg, rgba(24, 20, 44, 0.85) 0%, rgba(18, 15, 34, 0.85) 100%) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-left: 3px solid #06B6D4 !important;
  }

  /* Code Block Styling inside Messages */
  pre {
    background: #090814 !important;
    border: 1px solid rgba(124, 58, 237, 0.35) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
  }
  code {
    font-family: 'JetBrains Mono', monospace !important;
    color: #38BDF8 !important;
  }

  /* ── Streaming Cursor Pulse ── */
  .streaming-cursor {
    display: inline-block;
    color: #06B6D4;
    font-size: 0.9em;
    vertical-align: middle;
    animation: cursorGlow 0.7s infinite alternate ease-in-out;
  }
  @keyframes cursorGlow {
    0% { opacity: 0.2; transform: scale(0.8); text-shadow: 0 0 2px #06B6D4; }
    100% { opacity: 1; transform: scale(1.25); text-shadow: 0 0 10px #06B6D4, 0 0 20px #7C3AED; }
  }

  /* ── Continue Button (Glowing Pulse) ── */
  div.continue-btn-container button {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
    border: 1px solid rgba(52, 211, 153, 0.5) !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    animation: continuePulse 2.8s infinite ease-in-out !important;
    box-shadow: 0 4px 18px rgba(16, 185, 129, 0.35) !important;
  }
  div.continue-btn-container button:hover {
    transform: translateY(-2px) scale(1.03) !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.55) !important;
  }
  div.continue-btn-container button:active {
    transform: translateY(0px) scale(0.98) !important;
  }
  @keyframes continuePulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.45); }
    50% { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
  }

  /* ── Bottom Chat Input (Luminous Interactive Bar & Typing Motion) ── */
  [data-testid="stChatInput"] {
    padding-bottom: 1.25rem !important;
    background: transparent !important;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  [data-testid="stChatInput"] > div {
    border-radius: 20px !important;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  [data-testid="stChatInput"] textarea {
    background: rgba(24, 20, 44, 0.88) !important;
    border: 1.5px solid rgba(124, 58, 237, 0.4) !important;
    border-radius: 20px !important;
    color: #F8FAFC !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.96rem !important;
    padding: 0.95rem 1.25rem !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(20px) !important;
    transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  /* WHEN USER TYPES MANUALLY / FOCUSES INPUT */
  [data-testid="stChatInput"] textarea:focus {
    border-color: #06B6D4 !important;
    background: rgba(32, 26, 60, 0.96) !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.35), 
                0 0 28px rgba(124, 58, 237, 0.45), 
                0 16px 40px rgba(0, 0, 0, 0.6) !important;
    transform: translateY(-3px) scale(1.004) !important;
    color: #FFFFFF !important;
  }

  /* ── Send Button Micro-interaction & Hit Motion ── */
  [data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #7C3AED 0%, #06B6D4 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #FFFFFF !important;
    width: 42px !important;
    height: 42px !important;
    transition: all 0.28s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 3px 14px rgba(124, 58, 237, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    cursor: pointer !important;
  }
  [data-testid="stChatInput"] button svg {
    transition: all 0.2s ease !important;
    fill: #FFFFFF !important;
  }
  /* HOVER MOTION */
  [data-testid="stChatInput"] button:hover {
    transform: scale(1.18) rotate(-8deg) !important;
    box-shadow: 0 0 26px rgba(6, 182, 212, 0.85), 0 6px 20px rgba(124, 58, 237, 0.6) !important;
    background: linear-gradient(135deg, #8B5CF6 0%, #22D3EE 100%) !important;
  }
  [data-testid="stChatInput"] button:hover svg {
    transform: scale(1.1) !important;
  }
  /* WHEN USER HITS THE SEND BUTTON (Spring Press + Burst Glow) */
  [data-testid="stChatInput"] button:active {
    transform: scale(0.84) rotate(4deg) !important;
    box-shadow: 0 0 36px #06B6D4, 0 0 14px #FFFFFF, inset 0 0 10px rgba(255, 255, 255, 0.5) !important;
    transition: all 0.08s cubic-bezier(0.1, 0.9, 0.2, 1) !important;
  }

  /* ── Sidebar Styling ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120F24 0%, #0B0818 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--primary-light) !important;
    font-weight: 700;
  }
  [data-testid="stMetric"] {
    background: rgba(30, 25, 53, 0.65) !important;
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    padding: 0.85rem 1rem !important;
    backdrop-filter: blur(8px) !important;
  }
  [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
  [data-testid="stMetricValue"] { color: var(--primary-light) !important; font-weight: 700 !important; }

  /* ── General Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, var(--primary), #5B21B6);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.55rem 1.25rem;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.3);
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
  }
  .stButton > button:active {
    transform: translateY(0);
  }

  /* ── Selectbox ── */
  .stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
  }

  /* ── Scrollbars ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-dark); }
  ::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.5); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────

def initialize_session():
    """Initialize all Streamlit session state variables."""
    defaults = {
        "conversation_history": [],          # Full chat history
        "messages_display": [],              # Messages shown in UI
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_messages": 0,
        "total_sessions": 1,
        "selected_persona": "General Assistant",
        "is_processing": False,
        "llm_service": None,
        "api_configured": False,
        "error_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_or_create_service():
    """Get the LLM service, always loading fresh config from .env."""
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=env_path, override=True)
        service = LLMService(LLMConfig())
        st.session_state.llm_service = service
        st.session_state.api_configured = service.is_available()
    except Exception as e:
        import logging
        logging.error(f"Error loading LLMService: {e}")
        st.session_state.api_configured = False
    return st.session_state.llm_service


# ─── Chat Actions ──────────────────────────────────────────────────────────────

def clear_conversation():
    """Clear current conversation, keeping session settings intact."""
    st.session_state.conversation_history = []
    st.session_state.messages_display = []
    st.session_state.is_processing = False
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


def new_conversation():
    """Start a completely fresh conversation and increment session count."""
    clear_conversation()
    st.session_state.total_sessions += 1


def send_message(user_input: str):
    """
    Main message processing pipeline:
      1. Validate input
      2. Add to history
      3. Build prompt with full context
      4. Call LLM API
      5. Update conversation history
      6. Display response
    """
    # Step 1: Check for continue shorthand
    is_continue_cmd = user_input.strip().lower() in ["continue", "keep going", "continue generating", "go on", "more", "next"]
    if is_continue_cmd:
        actual_input = "Please continue generating the rest of the response or code directly from where you stopped. Complete all remaining methods and classes until the task is 100% finished."
        display_input = "▶️ Continue generating"
    else:
        actual_input = user_input.strip()
        display_input = user_input.strip()

    # Step 1b: Validate
    is_valid, err_msg = validate_user_input(actual_input)
    if not is_valid:
        st.warning(err_msg)
        return

    # Step 2: Add user message to histories
    user_msg = build_user_message(actual_input)
    st.session_state.conversation_history.append(user_msg)
    st.session_state.messages_display.append({"role": "user", "content": display_input})
    st.session_state.total_messages += 1

    # Step 3: Build full prompt with context
    system_prompt = PERSONA_PROMPTS.get(
        st.session_state.selected_persona,
        PERSONA_PROMPTS["General Assistant"]
    )
    messages = build_messages(st.session_state.conversation_history, system_prompt)

    # Step 4 & 5: Call API with streaming
    service = get_or_create_service()

    if not service or not service.is_available():
        err_display = "⚙️ **API not configured.** Please add your `LLM_API_KEY` to the `.env` file and restart the app."
        assistant_msg = build_assistant_message(err_display)
        st.session_state.conversation_history.append(assistant_msg)
        st.session_state.messages_display.append({"role": "assistant", "content": err_display})
        return

    try:
        st.session_state.is_processing = True

        # Stream response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""

            with st.spinner(""):
                for chunk in service.get_streaming_response(messages):
                    full_response += chunk
                    # Show streaming text with glowing pulsating cursor effect
                    if full_response.count("```") % 2 == 1:
                        response_placeholder.markdown(full_response + "\n▌")
                    else:
                        response_placeholder.markdown(full_response + " <span class='streaming-cursor'>●</span>", unsafe_allow_html=True)

            # Final render without cursor
            response_placeholder.markdown(full_response)

        # Step 5: Persist assistant response
        assistant_msg = build_assistant_message(full_response)
        st.session_state.conversation_history.append(assistant_msg)
        st.session_state.messages_display.append({"role": "assistant", "content": full_response})
        st.session_state.total_messages += 1

    except LLMServiceError as e:
        error_text = e.user_friendly_message
        st.session_state.error_log.append({
            "time": datetime.now().isoformat(),
            "type": e.error_type,
            "message": str(e),
        })
        # Add error as assistant message so context is maintained
        st.session_state.conversation_history.append(build_assistant_message(error_text))
        st.session_state.messages_display.append({"role": "assistant", "content": error_text})
        st.error(error_text)

    except Exception as e:
        error_text = f"❗ **Unexpected error:** {str(e)}"
        st.session_state.conversation_history.append(build_assistant_message(error_text))
        st.session_state.messages_display.append({"role": "assistant", "content": error_text})
        st.error(error_text)

    finally:
        st.session_state.is_processing = False


# ─── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
          <div style='font-size:2.5rem;'>🤖</div>
          <h2 style='margin:0; color:#A78BFA; font-size:1.2rem;'>AI Smart Chatbot</h2>
          <p style='color:#6B6890; font-size:0.75rem; margin:0;'>Powered by LLM API</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # API Status
        service = get_or_create_service()
        if service and service.is_available():
            cfg = service.config
            st.success(f"✅ API Connected")
            st.caption(f"Provider: **{cfg.api_provider.upper()}**")
            st.caption(f"Model: **{cfg.model_name}**")
        else:
            st.error("❌ API Not Configured")
            st.caption("Add `LLM_API_KEY` to your `.env` file")

        st.divider()

        # Persona selector
        st.markdown("### 🎭 Chatbot Persona")
        persona = st.selectbox(
            "Select Persona",
            options=list(PERSONA_PROMPTS.keys()),
            index=list(PERSONA_PROMPTS.keys()).index(st.session_state.selected_persona),
            key="persona_selector",
            label_visibility="collapsed",
        )
        if persona != st.session_state.selected_persona:
            st.session_state.selected_persona = persona
            st.caption("Persona updated! Takes effect on next message.")

        st.divider()

        # Conversation controls
        st.markdown("### 💬 Conversation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear current chat"):
                clear_conversation()
                st.rerun()
        with col2:
            if st.button("✨ New", use_container_width=True, help="Start a new conversation"):
                new_conversation()
                st.rerun()

        st.divider()

        # Stats
        st.markdown("### 📊 Session Stats")
        msg_count = len([m for m in st.session_state.messages_display if m["role"] == "user"])
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", msg_count)
        with col2:
            st.metric("Sessions", st.session_state.total_sessions)

        st.divider()

        # Model config info
        st.markdown("### ⚙️ Configuration")
        st.caption("Set these in your `.env` file:")
        st.code("""LLM_PROVIDER=groq
LLM_API_KEY=your_key
LLM_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024""", language="bash")

        # Error log (collapsible)
        if st.session_state.error_log:
            with st.expander(f"⚠️ Errors ({len(st.session_state.error_log)})"):
                for err in st.session_state.error_log[-5:]:
                    st.caption(f"`{err['type']}` — {err['time'][:19]}")

        st.divider()
        st.caption("Week 4 | Gen AI Internship Project")


# ─── Welcome Screen ────────────────────────────────────────────────────────────

def render_welcome():
    st.markdown("""
    <div class="welcome-hero">
      <div class="hero-badge">⚡ GROQ + OPENAI GPT-OSS-120B ENGINE</div>
      <h2 class="hero-title">What would you like to explore today?</h2>
      <p class="hero-sub">
        Ask complex coding challenges, full application architectures, or deep concepts.<br>
        Click a curated prompt below or type your custom query below.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">
      <span>RECOMMENDED STARTERS & PROMPTS</span>
    </div>
    """, unsafe_allow_html=True)

    prompt_cards = [
        {
            "icon": "🐍",
            "tag": "PYTHON",
            "title": "Sort Dictionaries by Key",
            "prompt": "Write a clean Python function to sort a list of dictionaries by a given key with robust error handling and type hints.",
            "desc": "Clean code with typing & docstrings",
        },
        {
            "icon": "🎮",
            "tag": "GAME DEV",
            "title": "Java Swing Snake Game",
            "prompt": "Write a complete Java Swing Snake game class with all methods fully implemented, ready to compile and run.",
            "desc": "Full runnable code from start to finish",
        },
        {
            "icon": "🤖",
            "tag": "AI & ML",
            "title": "Machine Learning Overview",
            "prompt": "Explain what machine learning is and compare its 3 main paradigms with concrete real-world use cases.",
            "desc": "Supervised, unsupervised & RL",
        },
        {
            "icon": "🧠",
            "tag": "NEURAL NETS",
            "title": "Deep Learning Simply",
            "prompt": "Explain neural networks simply using an everyday kitchen or brain analogy for beginners.",
            "desc": "Weights, activations & backprop",
        },
        {
            "icon": "⚛️",
            "tag": "PHYSICS",
            "title": "Quantum Computing 101",
            "prompt": "Explain quantum computing and superposition simply as if I am 10 years old with fun examples.",
            "desc": "Qubits, entanglement & principles",
        },
        {
            "icon": "🚀",
            "tag": "API DEV",
            "title": "FastAPI Production API",
            "prompt": "Write a complete production-grade CRUD REST API using FastAPI and Pydantic with validation and error responses.",
            "desc": "CRUD endpoints with Swagger docs",
        },
    ]

    # Render interactive cards in 3 columns
    cols = st.columns(3)
    for i, item in enumerate(prompt_cards):
        with cols[i % 3]:
            btn_label = f"{item['icon']}  **{item['title']}**\n\n_{item['desc']}_"
            if st.button(btn_label, key=f"rec_card_{i}", use_container_width=True):
                send_message(item["prompt"])
                st.rerun()

    st.markdown("""
    <div style='text-align:center; margin-top:2.2rem; color:#6B6890; font-size:0.85rem; letter-spacing:0.02em;'>
      🔄 <strong style="color:#A78BFA;">Multi-turn memory</strong> &nbsp;•&nbsp; 
      ⚡ <strong style="color:#06B6D4;">High-speed streaming</strong> &nbsp;•&nbsp; 
      🎭 <strong style="color:#10B981;">Custom personas</strong> &nbsp;•&nbsp; 
      🚀 <strong style="color:#F59E0B;">Infinite continuations</strong>
    </div>
    """, unsafe_allow_html=True)


# ─── Main App ──────────────────────────────────────────────────────────────────

def main():
    initialize_session()
    render_sidebar()

    # Header
    st.markdown("""
    <div class="chat-header">
      <h1>🤖 AI Smart Chatbot</h1>
      <p>Intelligent multi-turn conversations powered by LLM APIs</p>
    </div>
    """, unsafe_allow_html=True)

    # Persona badge
    persona_colors = {
        "General Assistant": "#7C3AED",
        "Code Assistant":    "#0EA5E9",
        "Science Tutor":     "#10B981",
        "Creative Writer":   "#F59E0B",
    }
    color = persona_colors.get(st.session_state.selected_persona, "#7C3AED")
    st.markdown(
        f"<div style='text-align:center; margin-bottom:1rem;'>"
        f"<span style='background:{color}22; color:{color}; border:1px solid {color}55; "
        f"border-radius:24px; padding:0.3rem 1rem; font-size:0.85rem; font-weight:600;'>"
        f"🎭 {st.session_state.selected_persona}</span></div>",
        unsafe_allow_html=True
    )

    # ── Chat History Display ──
    if not st.session_state.messages_display:
        render_welcome()
    else:
        # Render all previous messages
        for message in st.session_state.messages_display:
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Quick Continue button if last message was from assistant
        if st.session_state.messages_display and st.session_state.messages_display[-1]["role"] == "assistant":
            col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
            with col_c2:
                st.markdown('<div class="continue-btn-container">', unsafe_allow_html=True)
                if st.button("▶️ Continue / Complete Response", key="btn_continue_gen", use_container_width=True):
                    send_message("continue")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Chat Input ──
    if prompt := st.chat_input(
        "Type your message here...",
        disabled=st.session_state.is_processing,
        key="chat_input",
    ):
        # Show user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process and stream AI response
        send_message(prompt)
        st.rerun()


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
