import streamlit as st
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client

# ================= CONFIG =================
st.set_page_config(page_title="VLSI AI", page_icon="⚡")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = "You are a senior VLSI verification engineer."

# ================= AUTH =================
if "user" not in st.session_state:
    st.session_state.user = None

def login(email, password):
    res = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
    return res

def signup(email, password):
    res = supabase.auth.sign_up({
        "email": email,
        "password": password
    })
    return res

# ================= LOGIN UI =================
if not st.session_state.user:
    st.title("🔐 Login / Signup")

    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            try:
                res = login(email, password)
                st.session_state.user = res.user
                st.rerun()
            except Exception as e:
                st.error("Login failed")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")

        if st.button("Signup"):
            try:
                signup(email, password)
                st.success("Signup successful. Now login.")
            except:
                st.error("Signup failed")

    st.stop()

# ================= USER INFO =================
user_id = st.session_state.user.id

# ================= DATABASE =================
def load_chat():
    res = supabase.table("Chats").select("*").eq("user", user_id).execute()
    if res.data:
        return res.data[0]["messages"]
    return []

def save_chat(messages):
    data = {"user": user_id, "messages": messages}
    supabase.table("Chats").upsert(data).execute()

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

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = load_chat()

# ================= SIDEBAR =================
with st.sidebar:
    st.write(f"👤 {st.session_state.user.email}")

    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        save_chat([])

# ================= MAIN =================
st.title("⚡ VLSI AI (Authenticated)")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_chat(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_input)

    context = retrieve(user_input)

    final_prompt = f"{context}\n\nQuestion:\n{user_input}" if context else user_input

    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *st.session_state.messages[-8:],
        {"role": "user", "content": final_prompt}
    ]

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model=model_choice,
            messages=messages,
            max_tokens=1024
        )

        reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})
        save_chat(st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(reply)
