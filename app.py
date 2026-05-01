import streamlit as st
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import json

# ================= CONFIG =================
st.set_page_config(page_title="VLSI AI DB", page_icon="⚡")

SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer.
Understand conversations and answer with practical insights.
"""

# ================= DATABASE =================
conn = sqlite3.connect("chat.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    user TEXT,
    messages TEXT
)
""")
conn.commit()

def load_chat(user):
    cursor.execute("SELECT messages FROM chats WHERE user=?", (user,))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0])
    return []

def save_chat(user, messages):
    data = json.dumps(messages)
    cursor.execute("DELETE FROM chats WHERE user=?", (user,))
    cursor.execute("INSERT INTO chats VALUES (?, ?)", (user, data))
    conn.commit()

# ================= LOAD KNOWLEDGE =================
@st.cache_resource
def load_data():
    try:
        with open("vlsi_knowledge.txt", "r") as f:
            text = f.read()
    except:
        text = ""
    return [c.strip() for c in text.split("\n\n") if c.strip()]

@st.cache_resource
def load_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return model, np.array(embeddings)

chunks = load_data()
embed_model, embeddings = load_embeddings(chunks)

# ================= RAG =================
def retrieve(query, k=3, threshold=0.35):
    if len(chunks) == 0:
        return ""

    query_vec = embed_model.encode([query])[0]

    norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_vec)

    scores = np.dot(embeddings, query_vec) / (norms * query_norm + 1e-8)

    top_indices = np.argsort(scores)[-k:][::-1]

    if scores[top_indices[0]] < threshold:
        return ""

    return "\n\n".join([chunks[i] for i in top_indices])

# ================= FOLLOW-UP =================
def is_follow_up(query):
    keywords = ["this", "that", "it", "how", "why", "where"]
    return any(word in query.lower() for word in keywords)

# ================= USER LOGIN =================
user = st.sidebar.text_input("Enter your name")

if not user:
    st.warning("Enter your name to continue")
    st.stop()

# ================= LOAD SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = load_chat(user)

# ================= SIDEBAR =================
with st.sidebar:
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if not api_key:
        api_key = st.text_input("Enter Groq API Key", type="password")

    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        save_chat(user, [])

# ================= MAIN =================
st.title(f"⚡ VLSI AI - {user}")

if not api_key:
    st.stop()

# ================= DISPLAY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask your VLSI question...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    save_chat(user, st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG logic
    if is_follow_up(user_input):
        context = ""
    else:
        context = retrieve(user_input)

    if context:
        final_prompt = f"Context:\n{context}\n\nQuestion:\n{user_input}"
    else:
        final_prompt = user_input

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    clean_messages = st.session_state.messages[-8:]

    messages_for_model = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *clean_messages,
        {"role": "user", "content": final_prompt}
    ]

    try:
        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model=model_choice,
                messages=messages_for_model,
                max_tokens=1024
            )

            reply = response.choices[0].message.content

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

            save_chat(user, st.session_state.messages)

            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        st.error(str(e))
