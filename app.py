import streamlit as st
from openai import OpenAI

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Vaaluka Engineering Assistant",
    page_icon="⚡",
    layout="wide"
)

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer with 10+ years experience.

Expert in:
- SystemVerilog, UVM
- AXI, PCIe
- Interview preparation

Answer style:
- Clear explanation
- Code examples
- Practical insights
- Avoid fluff
"""

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚡ VLSI Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")

    st.markdown("---")
    st.markdown("### Topics")

    topics = {
        "SystemVerilog": "Explain blocking vs non-blocking assignments",
        "UVM": "Explain UVM architecture",
        "AXI": "Explain AXI handshake",
        "PCIe": "Explain PCIe TLP flow",
        "Interview": "Common VLSI interview questions"
    }

    for t, q in topics.items():
        if st.button(t):
            st.session_state.messages = [{"role": "user", "content": q}]

# ================= SESSION =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= UI =================
st.title("⚡ Vaaluka Engineering Assistant")

if not api_key:
    st.warning("Enter Groq API key to start")
    st.stop()

# ================= SHOW CHAT =================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= INPUT =================
user_input = st.chat_input("Ask your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    with st.spinner("Thinking..."):
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages
            ],
            max_tokens=1024
        )

        reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)
