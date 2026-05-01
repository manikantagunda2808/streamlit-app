import streamlit as st
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="VLSI Conversational AI",
    page_icon="⚡",
    layout="wide"
)

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer.

You:
- Understand multi-turn conversations
- Track context across questions
- Answer follow-up questions correctly

Expert in:
- SystemVerilog, UVM
- AXI, PCIe
- Debugging & interviews

Rules:
- If user refers to previous answer ("this", "that"), use conversation context
- Use RAG context only when relevant
- Give practical explanations
- Provide code examples when needed
"""

# ================= LOAD KNOWLEDGE =================
@st.cache_resource
def load_data():
    try:
        with open("vlsi_knowledge.txt", "r") as f:
            text = f.read()
    except:
        text = ""

    return [c.strip() for c in text.split("\n\n") if c.strip()]

# ================= EMBEDDINGS =================
@st.cache_resource
def load_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return model, np.array(embeddings)

chunks = load_data()
embed_model, embeddings = load_embeddings(chunks)

# ================= SMART RAG =================
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

# ================= FOLLOW-UP DETECTION =================
def is_follow_up(query):
    keywords = ["this", "that", "it", "these", "those", "how", "why", "where"]
    return any(word in query.lower() for word in keywords)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚡ VLSI AI Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")

    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        index=0
    )

# ================= MAIN =================
st.title("⚡ VLSI Conversational AI (Smart RAG)")

if not api_key:
    st.warning("Enter API key to continue")
    st.stop()

st.caption(f"Using model: {model_choice}")

# ================= DISPLAY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask your VLSI question...")

if user_input and user_input.strip():
    user_input = user_input.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ================= RAG DECISION =================
    if is_follow_up(user_input):
        context = ""   # skip RAG for follow-ups
    else:
        context = retrieve(user_input)

    if context:
        final_prompt = f"""
Use this VLSI context:

{context}

Question:
{user_input}
"""
    else:
        final_prompt = user_input

    # ================= CLIENT =================
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    # ================= CLEAN HISTORY =================
    clean_messages = []
    for msg in st.session_state.messages:
        if (
            isinstance(msg, dict)
            and msg.get("role") in ["user", "assistant"]
            and isinstance(msg.get("content"), str)
            and msg["content"].strip() != ""
        ):
            clean_messages.append({
                "role": msg["role"],
                "content": msg["content"].strip()
            })

    clean_messages = clean_messages[-8:]

    # ================= FINAL MESSAGE =================
    messages_for_model = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *clean_messages,
        {"role": "user", "content": final_prompt}
    ]

    # ================= API CALL =================
    try:
        with st.spinner("Thinking... 🤔"):

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

            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        st.error(f"Error: {str(e)}")
