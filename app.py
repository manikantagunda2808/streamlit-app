# ============================================================================
# VAALUKA VLSI AI - FIXED VERSION
# All bugs fixed, security improved, better error handling
# ============================================================================

import streamlit as st
from openai import OpenAI
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
import os
from datetime import datetime
import uuid

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Vaaluka VLSI AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CONFIGURATION =================
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL")
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    
    if not all([SUPABASE_URL, SUPABASE_KEY, GROQ_API_KEY]):
        st.error("❌ Missing required secrets. Please configure SUPABASE_URL, SUPABASE_KEY, and GROQ_API_KEY")
        st.stop()
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

SYSTEM_PROMPT = """You are a senior VLSI verification engineer at Vaaluka Solutions with 10+ years of experience.

You specialize in:
- SystemVerilog (SV) for verification and testbench development
- Universal Verification Methodology (UVM) frameworks
- AXI Protocol verification
- PCIe Protocol verification

When answering:
1. Provide clear, practical explanations
2. Include code examples when relevant
3. Highlight common pitfalls
4. Reference industry best practices
5. Be concise but thorough"""

# ================= HELPER FUNCTIONS =================

# Ensure knowledge base file exists
def ensure_knowledge_base():
    """Create sample vlsi_knowledge.txt if it doesn't exist"""
    if not os.path.exists("vlsi_knowledge.txt"):
        sample_content = """SystemVerilog Basics
SystemVerilog is a hardware description language (HDL) and hardware verification language (HVL).
It extends Verilog with object-oriented programming features, allowing for constrained randomization and functional coverage.
Key features include interfaces, classes, inheritance, and virtual tasks.

Blocking vs Non-Blocking Assignments
Blocking assignments (=) execute sequentially and complete before the next statement.
Non-blocking assignments (<=) schedule updates at the end of the time step, preserving delta cycle order.
Non-blocking assignments are preferred in sequential logic to avoid race conditions.

UVM Framework
The Universal Verification Methodology (UVM) is a standardized methodology for verifying integrated circuits.
It provides a base class library (uvm_object, uvm_component) for building verification components.
UVM testbenches consist of stimulus generators, drivers, monitors, scoreboards, and coverage collectors.

AXI Protocol Channels
The AXI protocol uses five separate channels: Write Address, Write Data, Write Response, Read Address, Read Data.
Each channel uses ready/valid handshaking for flow control.
This separation enables higher throughput and better pipelining compared to single-channel protocols.

PCIe Protocol Layers
PCIe uses a four-layer model: Physical, Data Link, Transaction, and Application layers.
The transaction layer handles request/completion cycles and manages split transactions.
PCIe uses packet-based communication with headers and data payloads.

Functional Coverage
Functional coverage monitors whether specific design behaviors have been exercised during simulation.
It's different from code coverage which just tracks which lines were executed.
Good functional coverage goals include all protocol transitions, error conditions, and edge cases."""
        
        with open("vlsi_knowledge.txt", "w") as f:
            f.write(sample_content)

# Create knowledge base on startup
ensure_knowledge_base()

# Load and cache the knowledge base
@st.cache_resource
def load_knowledge_base():
    """Load VLSI knowledge base with error handling"""
    try:
        with open("vlsi_knowledge.txt", "r") as f:
            text = f.read()
        
        if not text.strip():
            st.warning("⚠️ Knowledge base file is empty")
            return []
        
        # Split into chunks (paragraphs separated by double newline)
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        return chunks if chunks else []
    
    except FileNotFoundError:
        st.error("❌ vlsi_knowledge.txt not found")
        return []
    except Exception as e:
        st.error(f"❌ Error loading knowledge base: {e}")
        return []

# Load and cache embeddings
@st.cache_resource
def load_embeddings(chunks):
    """Load embeddings model and encode chunks"""
    try:
        if len(chunks) == 0:
            return None, None
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)
        return model, np.array(embeddings)
    
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {e}")
        return None, None

# Load knowledge base and embeddings
knowledge_chunks = load_knowledge_base()
embed_model, embeddings = load_embeddings(knowledge_chunks)

# ================= RAG RETRIEVAL (FIXED) =================
def retrieve(query, k=3, threshold=0.35):
    """Retrieve relevant knowledge chunks using RAG (FIXED VERSION)"""
    if len(knowledge_chunks) == 0:
        return ""
    
    try:
        # Encode the query
        query_vec = embed_model.encode([query])[0]
        
        # Compute cosine similarity properly
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([query_vec], embeddings)[0]
        
        # Filter by threshold FIRST (bug fix)
        valid_indices = np.where(scores >= threshold)[0]
        
        if len(valid_indices) == 0:
            return ""
        
        # Get top k from valid results
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[-k:][::-1]]
        
        # Return relevant chunks
        return "\n\n".join([knowledge_chunks[i] for i in sorted_indices])
    
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return ""

# ================= DATABASE FUNCTIONS (FIXED) =================
def load_chat(conversation_id=None):
    """Load messages from database (FIXED VERSION)"""
    try:
        if "user" not in st.session_state or not st.session_state.user:
            return [], None
        
        user_id = st.session_state.user.id
        query = supabase.table("Chats").select("*").eq("user", user_id)
        
        if conversation_id:
            query = query.eq("conversation_id", conversation_id)
        
        res = query.order("created_at", desc=True).limit(1).execute()
        
        if res.data:
            return res.data[0]["messages"], res.data[0]["conversation_id"]
        return [], None
    
    except Exception as e:
        st.error(f"❌ Database error loading chat: {e}")
        return [], None

def save_chat(messages, conversation_id=None):
    """Save messages to database (FIXED VERSION)"""
    try:
        if "user" not in st.session_state or not st.session_state.user:
            return False
        
        user_id = st.session_state.user.id
        
        # Generate conversation_id if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        data = {
            "user": user_id,
            "conversation_id": conversation_id,
            "messages": messages,
            "updated_at": datetime.now().isoformat()
        }
        
        # Use insert (Supabase handles conflicts with conversation_id)
        # Or use upsert with proper conflict target
        result = supabase.table("Chats").upsert(
            data,
            ignore_duplicates=False
        ).execute()
        
        return True
    
    except Exception as e:
        st.error(f"❌ Database error saving chat: {e}")
        return False

def get_conversation_list():
    """Get all conversations for the user"""
    try:
        if "user" not in st.session_state or not st.session_state.user:
            return []
        
        user_id = st.session_state.user.id
        res = supabase.table("Chats").select("conversation_id, created_at, messages").\
            eq("user", user_id).order("created_at", desc=True).execute()
        
        return res.data if res.data else []
    
    except Exception as e:
        st.error(f"❌ Error loading conversations: {e}")
        return []

# ================= AUTHENTICATION (FIXED) =================
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

def login(email, password):
    """Login user"""
    try:
        res = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return res
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return None

def signup(email, password):
    """Signup new user"""
    try:
        res = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        return res
    except Exception as e:
        st.error(f"Signup failed: {str(e)}")
        return None

# ================= LOGIN UI =================
if not st.session_state.user:
    st.title("🔐 Login / Signup")
    st.markdown("Welcome to Vaaluka VLSI AI Assistant")
    
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        st.markdown("### Login to Your Account")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", use_container_width=True, key="login_btn"):
            if email and password:
                res = login(email, password)
                if res:
                    st.session_state.user = res.user
                    st.success("✅ Login successful!")
                    st.rerun()
            else:
                st.error("Please enter email and password")
    
    with tab2:
        st.markdown("### Create New Account")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
        
        if st.button("Signup", use_container_width=True, key="signup_btn"):
            if not new_email or not new_password or not confirm_password:
                st.error("Please fill all fields")
            elif new_password != confirm_password:
                st.error("Passwords don't match")
            else:
                res = signup(new_email, new_password)
                if res:
                    st.success("✅ Account created! Please login now.")
    
    st.stop()

# ================= MAIN APP =================
user_id = st.session_state.user.id
user_email = st.session_state.user.email

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown(f"### 👤 {user_email}")
    st.markdown("---")
    
    # Model selection
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        help="Choose Groq model for response generation"
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more factual"
    )
    
    st.markdown("---")
    
    # Conversation management
    st.markdown("### 💬 Conversations")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()
    
    # Show past conversations
    conversations = get_conversation_list()
    if conversations:
        for conv in conversations:
            # Get preview of first message
            if conv["messages"]:
                preview = conv["messages"][0]["content"][:40]
                timestamp = datetime.fromisoformat(conv["created_at"]).strftime("%b %d %H:%M")
                
                if st.button(
                    f"📝 {preview}... ({timestamp})",
                    key=f"conv_{conv['conversation_id'][:8]}",
                    use_container_width=True
                ):
                    st.session_state.conversation_id = conv["conversation_id"]
                    st.session_state.messages = conv["messages"]
                    st.rerun()
    
    st.markdown("---")
    
    # Export and logout
    if st.button("📥 Export Chat", use_container_width=True):
        if st.session_state.messages:
            chat_text = "\n".join([
                f"{m['role'].upper()}: {m['content']}\n"
                for m in st.session_state.messages
            ])
            st.download_button(
                "Download Chat",
                chat_text,
                f"vlsi_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.messages = []
        st.session_state.conversation_id = None
        try:
            supabase.auth.sign_out()  # FIX: Properly sign out
        except:
            pass
        st.rerun()

# ================= MAIN CHAT INTERFACE =================
st.title("⚡ Vaaluka VLSI AI Assistant")

# Show conversation info
if st.session_state.conversation_id:
    st.info(f"💬 Conversation: {st.session_state.conversation_id[:8]}...")
else:
    st.info("📝 Start a new conversation!")

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================= CHAT INPUT =================
user_input = st.chat_input("Ask your VLSI verification question...")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # ================= RETRIEVE CONTEXT =================
    context = retrieve(user_input)
    
    # ================= BUILD API CALL (FIXED) =================
    try:
        # Build message history properly (FIX: don't duplicate user message)
        context_message = f"**KNOWLEDGE BASE CONTEXT:**\n{context}\n\n---\n\n" if context else ""
        
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            # Use all messages except the last one (which we just added)
            *st.session_state.messages[:-1],
            # Add the current query with context
            {
                "role": "user",
                "content": f"{context_message}**Question:** {user_input}"
            }
        ]
        
        # Initialize Groq client
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Show loading state
        with st.spinner("🤔 Thinking..."):
            response = client.chat.completions.create(
                model=model_choice,
                messages=api_messages,
                max_tokens=1024,
                temperature=temperature
            )
        
        # Extract response
        reply = response.choices[0].message.content
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply
        })
        
        # Save to database
        save_chat(st.session_state.messages, st.session_state.conversation_id)
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(reply)
    
    except Exception as e:
        st.error(f"❌ API Error: {str(e)}")
        st.markdown("""
        **Troubleshooting:**
        - Verify your Groq API key is correct
        - Check you have sufficient API credits
        - Ensure your internet connection is working
        """)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 20px;">
    <p><strong>Powered by Vaaluka Solutions</strong></p>
    <p>Senior VLSI Verification Engineering AI Assistant</p>
    <p>© 2024 Vaaluka Solutions. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
