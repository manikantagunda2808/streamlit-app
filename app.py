# ============================================================================
# VAALUKA ENGINEERING ASSISTANT - Streamlit App
# A VLSI Verification Training Assistant powered by Claude AI
# ============================================================================

# Import required libraries
import streamlit as st
import anthropic

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Configure the Streamlit page settings and appearance
st.set_page_config(
    page_title="Vaaluka Engineering Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    /* Make the app look more professional */
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    
    .topic-button {
        width: 100%;
        margin: 8px 0;
    }
    
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 14px;
    }
    
    .message-user {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .message-assistant {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SYSTEM PROMPT DEFINITION
# ============================================================================
# This prompt defines how the Claude AI assistant behaves
SYSTEM_PROMPT = """You are a Senior VLSI Verification Engineer at Vaaluka Solutions with 10+ years of industry experience.

You specialize in:
- SystemVerilog (SV) for verification and testbench development
- Universal Verification Methodology (UVM) frameworks and best practices
- AXI Protocol verification and implementation
- PCIe Protocol verification and debugging
- Interview preparation for VLSI engineers

Your communication style:
- Practical and direct, focusing on real-world solutions
- Always provide code examples when relevant and helpful
- Explain complex concepts in simple terms
- Reference industry best practices and design patterns
- Be concise but thorough in your explanations
- If asked about topics outside your expertise, politely redirect to your specialties

When answering:
1. Start with a clear explanation of the concept
2. Provide relevant code examples in appropriate languages (SystemVerilog, Python, etc.)
3. Highlight common pitfalls and how to avoid them
4. Suggest practical approaches based on industry experience
5. Ask clarifying questions if the question is ambiguous"""

# ============================================================================
# TOPIC DEFINITIONS
# ============================================================================
# Define all available topics and their starter questions
TOPICS = {
    "SystemVerilog": {
        "starter": "What are the key differences between blocking and non-blocking assignments in SystemVerilog, and when should I use each one?",
        "icon": "🔧"
    },
    "UVM": {
        "starter": "Can you explain the UVM testbench hierarchy and how sequences, drivers, and monitors work together?",
        "icon": "🏗️"
    },
    "AXI Protocol": {
        "starter": "What are the main channels in the AXI protocol and how does the handshake mechanism work?",
        "icon": "🔌"
    },
    "PCIe Protocol": {
        "starter": "Can you explain the PCIe transaction layer and how it handles request/completion cycles?",
        "icon": "💾"
    },
    "Interview Prep": {
        "starter": "What would you expect a good answer to be for this question: 'Describe a complex verification challenge you solved and how you approached it'?",
        "icon": "🎯"
    }
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Initialize Streamlit session state variables (these persist across app interactions)
if "messages" not in st.session_state:
    # Initialize empty conversation history
    st.session_state.messages = []

if "current_topic" not in st.session_state:
    # Track which topic is currently selected
    st.session_state.current_topic = None

if "api_key_valid" not in st.session_state:
    # Track whether the API key has been validated
    st.session_state.api_key_valid = False

# ============================================================================
# SIDEBAR SETUP
# ============================================================================
with st.sidebar:
    # Display company branding
    st.markdown("## 🏢 Vaaluka Solutions")
    st.markdown("---")
    
    # Display API Key input section
    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input(
        "Enter your Anthropic API Key",
        type="password",
        help="Your API key is used only in this session and not stored"
    )
    
    # Validate and store API key
    if api_key:
        st.session_state.api_key_valid = True
        api_key_to_use = api_key
    else:
        st.session_state.api_key_valid = False
        api_key_to_use = None
    
    st.markdown("---")
    
    # Display topic selection section
    st.markdown("### 📚 Learning Topics")
    
    # Create a button for each topic
    for topic_name, topic_info in TOPICS.items():
        # Use columns to align buttons properly
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.write(topic_info["icon"])
        
        with col2:
            # When a topic button is clicked:
            # 1. Set the current topic
            # 2. Clear conversation history (start fresh)
            # 3. Add the starter question to the conversation
            if st.button(
                topic_name,
                key=f"btn_{topic_name}",
                use_container_width=True,
                help=f"Learn about {topic_name}"
            ):
                st.session_state.current_topic = topic_name
                st.session_state.messages = []  # Clear chat history
                # Add the starter question to the conversation
                st.session_state.messages.append({
                    "role": "user",
                    "content": topic_info["starter"]
                })
                st.rerun()  # Refresh the app to show the new question
    
    st.markdown("---")
    
    # Display helpful information
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Click a topic to get started
    - The assistant will load a starter question
    - Ask follow-up questions to dive deeper
    - Code examples are provided when relevant
    """)
    
    st.markdown("---")
    
    # Display footer information
    st.markdown("### About")
    st.markdown("""
    **Vaaluka Engineering Assistant** v1.0
    
    Powered by Anthropic Claude
    """)

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Display the main title
st.markdown('<h1 class="main-title">⚡ Vaaluka Engineering Assistant</h1>', unsafe_allow_html=True)

# Display current topic
if st.session_state.current_topic:
    st.info(f"📚 Currently discussing: **{st.session_state.current_topic}**")
else:
    st.info("👈 Select a topic from the sidebar to get started!")

# Check if API key is provided
if not st.session_state.api_key_valid:
    st.warning("⚠️ Please enter your Anthropic API Key in the sidebar to begin.")
    st.markdown("""
    Don't have an API key? 
    1. Go to https://console.anthropic.com
    2. Sign up or log in
    3. Create a new API key
    4. Paste it in the sidebar
    """)
else:
    # ========================================================================
    # DISPLAY CONVERSATION HISTORY
    # ========================================================================
    # Show all previous messages in the conversation
    for message in st.session_state.messages:
        if message["role"] == "user":
            # Display user messages in a blue box
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            # Display assistant messages in a gray box
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(message["content"])
    
    # ========================================================================
    # CHAT INPUT SECTION
    # ========================================================================
    # Get user input from the chat interface
    user_input = st.chat_input(
        "Ask your VLSI verification question...",
        disabled=not st.session_state.api_key_valid
    )
    
    # Process user input when provided
    if user_input:
        # Add user message to conversation history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display the user's message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # ====================================================================
        # CALL CLAUDE API
        # ====================================================================
        # Initialize the Anthropic client with the provided API key
        client = anthropic.Anthropic(api_key=api_key_to_use)
        
        # Show a loading spinner while waiting for the response
        with st.spinner("Thinking... 🤔"):
            try:
                # Make the API call to Claude
                response = client.messages.create(
                    # Use Haiku model for cost efficiency (as requested)
                    model="claude-haiku-4-5-20251001",
                    # Set maximum tokens for the response (adjust if needed)
                    max_tokens=1024,
                    # Use the system prompt to define assistant behavior
                    system=SYSTEM_PROMPT,
                    # Pass the entire conversation history so Claude understands context
                    messages=st.session_state.messages
                )
                
                # Extract the assistant's response text
                assistant_message = response.content[0].text
                
                # Add the assistant's response to conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Display the assistant's response
                with st.chat_message("assistant", avatar="⚡"):
                    st.markdown(assistant_message)
                
            except anthropic.APIError as e:
                # Handle API errors gracefully
                st.error(f"❌ API Error: {str(e)}")
                st.markdown("""
                **Troubleshooting:**
                - Check that your API key is correct
                - Verify you have sufficient API credits
                - Ensure your internet connection is working
                """)

# ============================================================================
# FOOTER
# ============================================================================
# Display footer at the bottom of the page
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Powered by Vaaluka Solutions</strong></p>
    <p>Senior VLSI Verification Engineering Training Assistant</p>
    <p style="font-size: 12px; margin-top: 10px;">© 2024 Vaaluka Solutions. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
