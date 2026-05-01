import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ================= CONFIG =================
st.set_page_config(page_title="VLSI FAISS Assistant", page_icon="⚡")

SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer.

Use provided context carefully.
Give:
- Clear explanation
- SystemVerilog examples
- Practical debugging insights
"""

# ================= LOAD KNOWLEDGE =================
@st.cache_resource
def load_data():
    try:
        with open("vlsi_knowledge.txt", "r") as f:
            text = f.read()
    except:
        text = ""

    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks

# ================= EMBEDDING + INDEX =================
@st.cache_resource
def create_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return model, index

chunks = load_data()
embed_model, faiss_index = create_index(chunks)

# ================= SEARCH =================
def retrieve(query, k=3):
    query_vec = embed_model.encode([query])
    D, I = faiss_index.search(np.array(query_vec), k)

    results = []
    for idx in I[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return "\n\n".join(results)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚡ VLSI FAISS Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")

    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

# ================= MAIN =================
st.title("⚡ VLSI Assistant (FAISS RAG)")

if not api_key:
    st.warning("Enter API key")
    st.stop()

# ================= DISPLAY =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask VLSI question...")

if user_input and user_input.strip():
    user_input = user_input.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ===== FAISS RETRIEVAL =====
    context = retrieve(user_input)

    # ===== CLIENT =====
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    try:
        with st.spinner("Thinking..."):

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
Use this context:

{context}

Question:
{user_input}
"""
                    }
                ],
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
        st.error(str(e))
