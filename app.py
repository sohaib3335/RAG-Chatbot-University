"""
RAG Chatbot - Streamlit Web Interface
A user-friendly web interface for the RAG chatbot
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'is_initialized' not in st.session_state:
        st.session_state.is_initialized = False


def initialize_rag_chain(use_local: bool = False):
    """Initialize or load the RAG chain"""
    try:
        rag_chain = RAGChain(
            use_local_embeddings=use_local,
            use_local_llm=use_local
        )
        
        # Try to load existing store
        if rag_chain.load_existing_store():
            st.session_state.rag_chain = rag_chain
            st.session_state.is_initialized = True
            return True
        else:
            st.session_state.rag_chain = rag_chain
            st.session_state.is_initialized = False
            return False
            
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return False


def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        # University Logo/Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #c9a227; margin-bottom: 0;">🎓</h1>
            <h3 style="color: white; margin-top: 0.5rem;">Student Portal</h3>
            <p style="color: #a0a0a0; font-size: 0.9rem;">Your Campus Information Hub</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        st.markdown("**AI Model**")
        use_local = st.checkbox(
            "Use Local Models",
            value=False,
            help="Use local HuggingFace embeddings and Ollama LLM instead of OpenAI"
        )
        
        # Initialize button
        if st.button("🔄 Connect to Knowledge Base"):
            with st.spinner("Connecting to university database..."):
                if initialize_rag_chain(use_local):
                    st.success("✅ Connected successfully!")
                else:
                    st.warning("No documents found. Please add documents first.")
        
        st.divider()
        
        # Document ingestion
        st.markdown("### 📚 Add Resources")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt', 'pdf', 'docx', 'md', 'csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📥 Add to Knowledge Base"):
            ingest_uploaded_files(uploaded_files, use_local)
        
        st.divider()
        
        # System stats
        st.markdown("### 📊 Database Info")
        if st.session_state.rag_chain and st.session_state.is_initialized:
            stats = st.session_state.rag_chain.get_stats()
            vs_stats = stats.get('vector_store', {})
            if vs_stats:
                st.metric("📄 Documents Indexed", vs_stats.get('document_count', 'N/A'))
        else:
            st.info("Not connected")
        
        st.divider()
        
        # Quick links
        st.markdown("### 🔗 Quick Topics")
        st.markdown("""
        <div style="color: #a0a0a0; font-size: 0.85rem;">
        • Academic Policies<br>
        • Course Registration<br>
        • Financial Aid<br>
        • Student Services<br>
        • Campus Resources
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Clear chat history
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()


def ingest_uploaded_files(uploaded_files, use_local: bool):
    """Ingest uploaded files into the knowledge base"""
    from langchain.schema import Document
    
    with st.spinner("Processing documents..."):
        # Save uploaded files temporarily
        temp_dir = Path(Config.KNOWLEDGE_BASE_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
        
        # Initialize RAG chain if needed
        if st.session_state.rag_chain is None:
            st.session_state.rag_chain = RAGChain(
                use_local_embeddings=use_local,
                use_local_llm=use_local
            )
        
        # Ingest documents
        num_chunks = st.session_state.rag_chain.ingest_documents(source_path=temp_dir)
        
        if num_chunks > 0:
            st.session_state.is_initialized = True
            st.success(f"✅ Successfully ingested {num_chunks} chunks from {len(saved_files)} files!")
        else:
            st.error("❌ No documents were ingested. Check file formats.")


def render_chat_interface():
    """Render the main chat interface"""
    # University Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 University Student Assistant</h1>
        <p>Your AI-powered guide to campus policies, services, and resources</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if system is initialized
    if not st.session_state.is_initialized:
        # Welcome message for new users
        st.markdown("""
        <div class="welcome-box">
            <h3>👋 Welcome to the University Assistant!</h3>
            <p>I'm here to help you find information about:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Category cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **📚 Academic Policies**
            - Grading systems
            - Academic integrity
            - Grade appeals
            - Exam procedures
            """)
        with col2:
            st.markdown("""
            **💰 Financial Information**
            - Tuition & fees
            - Financial aid
            - Scholarships
            - Payment plans
            """)
        with col3:
            st.markdown("""
            **🏛️ Campus Services**
            - Course registration
            - Student services
            - Health center
            - Career services
            """)
        
        st.divider()
        
        st.info("""
        **🚀 Getting Started:**
        1. Click **'Connect to Knowledge Base'** in the sidebar
        2. Or run `python main.py ingest` if you haven't indexed documents yet
        3. Start asking questions!
        """)
        
        # Show quick start guide
        with st.expander("📖 Need Help? View Setup Guide"):
            st.markdown("""
            ### How to use the University Assistant:
            
            1. **Connect**: Click 'Connect to Knowledge Base' in the sidebar
            2. **Ask Questions**: Type your questions about university policies, procedures, or services
            3. **Get Answers**: I'll search through official university documents to help you
            
            ### Example Questions:
            - "What is the grading scale for undergraduate students?"
            - "How do I apply for financial aid?"
            - "What are the library hours?"
            - "How do I register for courses?"
            - "What is the policy on academic integrity?"
            
            ### Adding New Documents:
            - Upload PDF, TXT, DOCX, or MD files using the sidebar
            - Click 'Add to Knowledge Base' to index them
            """)
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available (with deduplicated, clean filenames)
            if message.get("sources"):
                with st.expander(f"📚 Sources ({len(message['sources'])} documents)"):
                    for source in message["sources"]:
                        display_name = source.get('filename', source['source'])
                        st.markdown(f"- **{display_name}**")
    
    # Helpful suggestions for first-time users
    if not st.session_state.chat_history:
        st.markdown("---")
        st.markdown("**💡 Try asking about:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("📝 Grading Policy"):
                st.session_state.suggested_query = "What is the grading policy?"
                st.rerun()
        with col2:
            if st.button("💳 Tuition & Fees"):
                st.session_state.suggested_query = "What are the tuition fees?"
                st.rerun()
        with col3:
            if st.button("📅 Registration"):
                st.session_state.suggested_query = "How do I register for courses?"
                st.rerun()
        with col4:
            if st.button("🏥 Health Services"):
                st.session_state.suggested_query = "What health services are available?"
                st.rerun()
    
    # Handle suggested query
    suggested = st.session_state.get('suggested_query', None)
    if suggested:
        prompt = suggested
        st.session_state.suggested_query = None
    else:
        prompt = st.chat_input("Ask me anything about university policies, services, or resources...")
    
    # Chat input
    if prompt:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating response..."):
                try:
                    result = st.session_state.rag_chain.query(prompt)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display unique sources with cleaner filenames
                    if result.get('sources'):
                        with st.expander(f"📚 Sources ({len(result['sources'])} documents)"):
                            for source in result['sources']:
                                # Use filename if available, otherwise extract from path
                                display_name = source.get('filename', source['source'])
                                st.markdown(f"- **{display_name}**")
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', [])
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Initialize suggested_query if not present
    if 'suggested_query' not in st.session_state:
        st.session_state.suggested_query = None
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎓 University Student Assistant | Powered by RAG Technology</p>
        <p style="font-size: 0.8rem; color: #999;">For official information, always verify with the university administration.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
