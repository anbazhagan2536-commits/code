import streamlit as st
import json
import numpy as np
import google.generativeai as genai
from gtts import gTTS
import tempfile
import os
import time

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Jabez AI", layout="wide")

# --------------------------------
# API KEY
# --------------------------------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --------------------------------
# LOAD DATASET
# --------------------------------
MEMORY_FILE = "dataset.json"

if not os.path.exists(MEMORY_FILE):
    st.error("dataset.json file not found.")
    st.stop()

with open(MEMORY_FILE, "r", encoding="utf-8") as f:
    memory_data = json.load(f)

# --------------------------------
# MEMORY EXTRACTION
# --------------------------------
def flatten_memory(data):
    texts = []

    for item in data.get("conversations", []):
        if isinstance(item, dict) and "dialogue" in item:
            texts.append(item["dialogue"])

    for item in data.get("chat_examples", []):
        if isinstance(item, dict):
            texts.append(item.get("user", ""))
            texts.append(item.get("bot", ""))

    for item in data.get("letters", []):
        if isinstance(item, dict) and "content" in item:
            texts.append(item["content"])

    for quote in data.get("quotes", []):
        texts.append(quote)

    if "love_story" in data:
        for v in data["love_story"].values():
            texts.append(v)

    return [t for t in texts if t.strip() != ""]

memory_texts = flatten_memory(memory_data)

# --------------------------------
# SIMPLE CONTEXT RETRIEVAL (NO HEAVY MODEL)
# --------------------------------
def retrieve_context(query, top_k=3):
    query = query.lower()

    scored = []
    for text in memory_texts:
        score = sum(1 for word in query.split() if word in text.lower())
        scored.append((score, text))

    scored.sort(reverse=True)
    return [t for s, t in scored[:top_k] if s > 0]

# --------------------------------
# SAVE MEMORY
# --------------------------------
def save_memory(user, ai):
    if "chat_examples" not in memory_data:
        memory_data["chat_examples"] = []

    memory_data["chat_examples"].append({
        "user": user,
        "bot": ai
    })

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)

# --------------------------------
# EMOTION DETECTION
# --------------------------------
def detect_emotion(text):
    t = text.lower()
    if any(w in t for w in ["sad", "miss", "lonely", "cry", "hurt"]):
        return "sad"
    if any(w in t for w in ["happy", "love", "excited", "great"]):
        return "happy"
    return "neutral"

# --------------------------------
# VOICE OUTPUT
# --------------------------------
def speak(text, emotion):
    slow = True if emotion == "sad" else False
    tts = gTTS(text, slow=slow)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# --------------------------------
# SAFE GENERATION
# --------------------------------
def generate_response(prompt):
    model = genai.GenerativeModel("models/gemini-2.5-flash")

    for i in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            time.sleep(3)

    return "⚠️ Server busy. Please try again later."

# --------------------------------
# DISCLAIMER
# --------------------------------
st.warning("""
⚠️ Research Prototype.
Jabez AI is a synthetic persona for study purposes.
It does NOT represent a real human.
""")

# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.title("🧠 Jabez Control Panel")

voice_on = st.sidebar.checkbox("🔊 Voice Output", value=True)

persona_mode = st.sidebar.radio(
    "Persona Mode",
    ["🧠 Memory Mode", "💬 Casual Talk", "🤍 Emotional Support"]
)

show_emotion = st.sidebar.checkbox("Show Emotion Debug")

# --------------------------------
# MAIN UI
# --------------------------------
st.title("🤖 Jabez AI")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"🧍 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Jabez:** {msg}")

st.markdown("---")
user_input = st.text_input("Talk to Jabez:")

if st.button("Send") and user_input.strip():

    st.session_state.chat.append(("user", user_input))

    # CONTEXT
    context = retrieve_context(user_input)
    context_text = "\n".join([c[:300] for c in context])
    context_text = context_text[:1000]

    # MODE
    if persona_mode == "🧠 Memory Mode":
        mode_instruction = "Use past memories strongly."
    elif persona_mode == "💬 Casual Talk":
        mode_instruction = "Respond short and casual."
    else:
        mode_instruction = "Respond warmly and supportively."

    # PROMPT
    prompt = f"""
You are Jabez.
You are an AI persona for academic research.
Never claim to be human.
Maintain ethical boundaries.

{mode_instruction}

Memory Context:
{context_text}

User: {user_input}
Jabez:
"""

    prompt = prompt[:3000]

    # GENERATE
    ai_text = generate_response(prompt)

    emotion = detect_emotion(ai_text)

    st.session_state.chat.append(("ai", ai_text))

    save_memory(user_input, ai_text)

    if voice_on:
        audio_file = speak(ai_text, emotion)
        st.audio(audio_file)

    if show_emotion:
        st.write("Detected Emotion:", emotion)
