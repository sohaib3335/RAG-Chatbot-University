"""
RAG Chatbot - Main Application
Command-line interface for the RAG chatbot
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_chain import RAGChain, create_rag_chain
from src.config import Config


def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                    RAG CHATBOT v1.0                      ║
    ║         Retrieval Augmented Generation System            ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def interactive_mode(rag_chain: RAGChain):
    """
    Run the chatbot in interactive mode
    
    Args:
        rag_chain: Initialized RAGChain instance
    """
    print("\n💬 Interactive Mode")
    print("Type your questions. Use 'quit' or 'exit' to stop.")
    print("Commands: /stats, /sources, /help, /reset\n")
    print("-" * 50)
    
    show_sources = True
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('/'):
                handle_command(user_input, rag_chain, show_sources)
                continue
            
            # Query the RAG system
            print("\n🔍 Searching knowledge base...")
            result = rag_chain.query(user_input, return_sources=show_sources)
            
            print(f"\n🤖 Assistant: {result['answer']}")
            
            if show_sources and result.get('sources'):
                print(f"\n📚 Sources ({result['num_retrieved']} documents retrieved):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. {source['source']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def handle_command(command: str, rag_chain: RAGChain, show_sources: bool):
    """Handle special commands"""
    cmd = command.lower().strip()
    
    if cmd == '/help':
        print("""
Available Commands:
  /stats    - Show system statistics
  /sources  - Toggle source display
  /reset    - Reset the knowledge base
  /help     - Show this help message
  quit/exit - Exit the chatbot
        """)
    
    elif cmd == '/stats':
        stats = rag_chain.get_stats()
        print("\n📊 System Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    elif cmd == '/sources':
        show_sources = not show_sources
        print(f"📚 Source display: {'ON' if show_sources else 'OFF'}")
    
    elif cmd == '/reset':
        confirm = input("⚠️ This will delete all indexed documents. Confirm? (y/n): ")
        if confirm.lower() == 'y':
            rag_chain.reset()
            print("✅ Knowledge base reset")
    
    else:
        print(f"Unknown command: {command}. Type /help for available commands.")


def ingest_command(args):
    """Handle document ingestion"""
    print_banner()
    print("📥 Document Ingestion Mode\n")
    
    source_path = Path(args.path) if args.path else Config.KNOWLEDGE_BASE_DIR
    
    if not source_path.exists():
        print(f"❌ Path not found: {source_path}")
        return
    
    # Create RAG chain
    rag_chain = RAGChain(
        use_local_embeddings=args.local,
        use_local_llm=args.local
    )
    
    # Ingest documents
    num_chunks = rag_chain.ingest_documents(source_path=source_path)
    
    if num_chunks > 0:
        print(f"\n✅ Successfully indexed {num_chunks} chunks")
        print(f"📂 Vector store saved to: {Config.CHROMA_PERSIST_DIR}")


def chat_command(args):
    """Handle chat mode"""
    print_banner()
    
    # Create RAG chain
    rag_chain = RAGChain(
        use_local_embeddings=args.local,
        use_local_llm=args.local
    )
    
    # Try to load existing store
    if not rag_chain.load_existing_store():
        print("⚠️ No indexed documents found.")
        print(f"Please run: python main.py ingest --path <documents_folder>")
        print(f"Or place documents in: {Config.KNOWLEDGE_BASE_DIR}")
        return
    
    # Start interactive mode
    interactive_mode(rag_chain)


def query_command(args):
    """Handle single query"""
    # Create RAG chain
    rag_chain = RAGChain(
        use_local_embeddings=args.local,
        use_local_llm=args.local
    )
    
    # Load existing store
    if not rag_chain.load_existing_store():
        print("❌ No indexed documents found. Please run ingestion first.")
        return
    
    # Execute query
    result = rag_chain.query(args.question)
    
    print(f"\nQuestion: {args.question}")
    print(f"\nAnswer: {result['answer']}")
    
    if result.get('sources'):
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source['source']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Retrieval Augmented Generation System"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents into knowledge base')
    ingest_parser.add_argument(
        '--path', '-p',
        type=str,
        help='Path to documents directory or file'
    )
    ingest_parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='Use local models (HuggingFace embeddings, Ollama LLM)'
    )
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat mode')
    chat_parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='Use local models'
    )
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Ask a single question')
    query_parser.add_argument(
        'question',
        type=str,
        help='Question to ask'
    )
    query_parser.add_argument(
        '--local', '-l',
        action='store_true',
        help='Use local models'
    )
    
    args = parser.parse_args()
    
    if args.command == 'ingest':
        ingest_command(args)
    elif args.command == 'chat':
        chat_command(args)
    elif args.command == 'query':
        query_command(args)
    else:
        print_banner()
        parser.print_help()
        print("\nQuick Start:")
        print("  1. Add your documents to: data/knowledge_base/")
        print("  2. Run: python main.py ingest")
        print("  3. Run: python main.py chat")


if __name__ == "__main__":
    main()
