# 🤖 AI-Powered Smart Chatbot

**Week 4 — Generative AI Internship Project**

A production-ready, multi-turn AI chatbot built with Python and Streamlit, powered by LLM APIs (Groq / OpenAI).

---

## 📁 Project Structure

```
AI-Smart-Chatbot/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── .env.example            ← Environment variable template
├── .gitignore              ← Keeps API keys out of git
├── README.md               ← This file
├── services/
│   └── llm_service.py      ← LLM API integration layer
└── utils/
    └── prompts.py          ← Prompt engineering & validation
```

---

## 🚀 Quick Start

### 1. Clone / Navigate to the project
```bash
cd AI-Smart-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your API key
```bash
# Copy the template
copy .env.example .env

# Edit .env and add your key:
# LLM_API_KEY=your_actual_api_key_here
```

> **Get a FREE Groq API key** at https://console.groq.com (no credit card needed!)

### 4. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## ⚙️ Configuration (`.env`)

| Variable          | Default                  | Description                     |
|-------------------|--------------------------|---------------------------------|
| `LLM_PROVIDER`    | `groq`                   | `groq` or `openai`              |
| `LLM_API_KEY`     | *(required)*             | Your API key                    |
| `LLM_MODEL`       | `llama-3.1-8b-instant`   | Model name                      |
| `LLM_TEMPERATURE` | `0.7`                    | Creativity (0.0 – 1.0)          |
| `LLM_MAX_TOKENS`  | `1024`                   | Max response length              |
| `LLM_TIMEOUT`     | `30`                     | Request timeout (seconds)        |
| `LLM_MAX_RETRIES` | `2`                      | Retry attempts on failure        |

---

## 🧠 Architecture

```
User Input
    ↓
Streamlit Chat UI  (app.py)
    ↓
Input Validation   (utils/prompts.py → validate_user_input)
    ↓
Session State      (Streamlit st.session_state)
    ↓
Prompt Builder     (utils/prompts.py → build_messages)
    ↓
LLM API Call       (services/llm_service.py → get_streaming_response)
    ↓
Streamed Response  (token-by-token display)
    ↓
Update History     (conversation_history + messages_display)
    ↓
Display in Chat UI
```

---

## ✨ Features

| Feature | Status |
|---------|--------|
| Multi-turn context awareness | ✅ |
| Streaming responses | ✅ |
| Conversation history | ✅ |
| Clear conversation | ✅ |
| New conversation | ✅ |
| Multiple AI personas | ✅ |
| API error handling | ✅ |
| Loading indicators | ✅ |
| Input validation | ✅ |
| Secure API key (.env) | ✅ |
| Configurable via env vars | ✅ |
| Groq + OpenAI support | ✅ |

---

## 🔄 Multi-Turn Test

The chatbot correctly handles contextual follow-up questions:

```
You:  "What is machine learning?"
Bot:  [Explains machine learning]

You:  "What are its types?"
Bot:  [Lists supervised, unsupervised, reinforcement learning]

You:  "Explain the second type with an example."
Bot:  [Explains UNSUPERVISED learning with example — knows "second type" from context]
```

---

## 🎭 Available Personas

- **General Assistant** — Balanced, helpful responses
- **Code Assistant** — Expert software engineering help
- **Science Tutor** — Patient, example-driven explanations
- **Creative Writer** — Storytelling and creative content

---

## 🔒 Security

- API keys are **never hardcoded** in source code
- `.env` is excluded from git via `.gitignore`
- `.env.example` is safe to commit (contains no real keys)

---

## 🛠️ Supported LLM Providers

### Groq (Recommended — Free tier available)
```
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-8b-instant   # Fast, free
LLM_MODEL=llama-3.1-70b-versatile # More capable
```

### OpenAI
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini   # Cost-effective
LLM_MODEL=gpt-4o        # Most capable
```

---

*Built for Week 4 of the Generative AI Internship Program.*
