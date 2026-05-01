import streamlit as st
from openai import OpenAI

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="VLSI RAG Assistant",
    page_icon="⚡",
    layout="wide"
)

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer with 10+ years experience.

Expert in:
- SystemVerilog, UVM
- AXI, PCIe
- Debugging and interview preparation

Instructions:
- Give clear explanations
- Provide SystemVerilog examples
- Mention common bugs/pitfalls
- Be practical, not theoretical
"""

# ================= LOAD KNOWLEDGE =================
def load_knowledge():
    try:
        with open("vlsi_knowledge.txt", "r") as f:
            return f.read()
    except:
        return ""

knowledge_base = load_knowledge()

# ================= SIMPLE RAG SEARCH =================
def get_relevant_context(query, knowledge):
    chunks = knowledge.split("\n\n")
    scored = []

    for chunk in chunks:
        score = 0
        for word in query.lower().split():
            if word in chunk.lower():
                score += 1
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for score, chunk in scored[:3]]

    return "\n\n".join(top_chunks)

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚡ VLSI Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")

    st.markdown("### ⚙️ Model Settings")
    model_choice = st.selectbox(
        "Select Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### 📚 Quick Topics")

    topics = {
        "SystemVerilog": "Explain blocking vs non-blocking assignments",
        "UVM": "Explain UVM phases",
        "AXI": "Explain AXI handshake",
        "PCIe": "Explain PCIe TLP flow",
        "Interview": "Ask me VLSI interview questions"
    }

    for topic, question in topics.items():
        if st.button(topic):
            st.session_state.messages = [{"role": "user", "content": question}]

# ================= MAIN =================
st.title("⚡ VLSI Engineering Assistant (RAG Enabled)")

if not api_key:
    st.warning("Enter Groq API key to continue")
    st.stop()

st.caption(f"Using model: {model_choice}")

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ================= USER INPUT =================
user_input = st.chat_input("Ask your VLSI question...")

if user_input and user_input.strip():
    user_input = user_input.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ================= RAG CONTEXT =================
    context = get_relevant_context(user_input, knowledge_base)

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

    clean_messages = clean_messages[-10:]

    # ================= API CALL =================
    try:
        with st.spinner("Thinking... 🤔"):

            response = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"""
Use this VLSI context if relevant:

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
        st.error(f"Error: {str(e)}")
