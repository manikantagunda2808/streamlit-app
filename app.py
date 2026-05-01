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
- Avoid unnecessary fluff
"""

# ================= SESSION STATE =================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚡ VLSI Assistant")

    api_key = st.text_input("Enter Groq API Key", type="password")

    st.markdown("---")
    st.markdown("### Topics")

    topics = {
        "SystemVerilog": "Explain blocking vs non-blocking assignments",
        "UVM": "Explain UVM architecture",
        "AXI": "Explain AXI handshake mechanism",
        "PCIe": "Explain PCIe TLP flow",
        "Interview": "Tell me important VLSI interview questions"
    }

    for topic, question in topics.items():
        if st.button(topic):
            st.session_state.messages = [{
                "role": "user",
                "content": question
            }]

# ================= MAIN UI =================
st.title("⚡ Vaaluka Engineering Assistant")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to start.")
    st.stop()

# ================= DISPLAY CHAT =================
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ================= USER INPUT =================
user_input = st.chat_input("Ask your VLSI question...")

if user_input and user_input.strip():
    user_input = user_input.strip()

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    # ================= CLEAN MESSAGE HISTORY =================
    clean_messages = []

    for msg in st.session_state.messages:
        if (
            isinstance(msg, dict)
            and "role" in msg
            and "content" in msg
            and msg["role"] in ["user", "assistant"]
            and isinstance(msg["content"], str)
            and msg["content"].strip() != ""
        ):
            clean_messages.append({
                "role": msg["role"],
                "content": msg["content"].strip()
            })

    # Limit history (avoid token overflow)
    clean_messages = clean_messages[-10:]

    # ================= API CALL =================
    try:
        with st.spinner("Thinking... 🤔"):

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *clean_messages
                ],
                max_tokens=1024
            )

            reply = response.choices[0].message.content

            # Save assistant reply
            st.session_state.messages.append({
                "role": "assistant",
                "content": reply
            })

            with st.chat_message("assistant"):
                st.markdown(reply)

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()
