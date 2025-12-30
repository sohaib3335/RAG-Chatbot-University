"""
LLM Module
Handles Language Model interactions for response generation
"""

from typing import Optional, List, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from src.config import Config


# Default RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant. Use the following context to answer the user's question. 
If you cannot find the answer in the context, say "I don't have enough information to answer this question based on the provided documents."

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Be concise but comprehensive
3. If the context doesn't contain relevant information, acknowledge it
4. Cite the source if available in the context

Answer:"""


class LLMManager:
    """
    LLM Manager class for handling language model interactions.
    Supports Claude (Anthropic), OpenAI GPT models, and local models via Ollama.
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        use_local: bool = None,
        use_claude: bool = None
    ):
        """
        Initialize LLMManager
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for response generation
            use_local: Whether to use local LLM (Ollama)
            use_claude: Whether to use Claude (Anthropic)
        """
        self.use_local = use_local if use_local is not None else Config.USE_LOCAL_LLM
        self.use_claude = use_claude if use_claude is not None else Config.USE_CLAUDE
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        
        # Set model name based on provider
        if model_name:
            self.model_name = model_name
        elif self.use_claude:
            self.model_name = Config.CLAUDE_MODEL
        else:
            self.model_name = Config.LLM_MODEL
        
        self.llm = self._initialize_llm()
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE
        )
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration"""
        if self.use_local:
            return self._get_local_llm()
        elif self.use_claude:
            return self._get_claude_llm()
        else:
            return self._get_openai_llm()
    
    def _get_claude_llm(self):
        """Get Claude LLM from Anthropic"""
        from langchain_anthropic import ChatAnthropic
        
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY in .env file "
                "or set USE_CLAUDE=false to use OpenAI models."
            )
        
        print(f"🤖 Using Claude LLM: {self.model_name}")
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=Config.LLM_MAX_TOKENS,
            anthropic_api_key=Config.ANTHROPIC_API_KEY
        )
    
    def _get_openai_llm(self):
        """Get OpenAI LLM"""
        from langchain_openai import ChatOpenAI
        
        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file "
                "or set USE_LOCAL_LLM=true to use local models."
            )
        
        print(f"🤖 Using OpenAI LLM: {self.model_name}")
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def _get_local_llm(self):
        """Get local LLM via Ollama"""
        from langchain_community.llms import Ollama
        
        print(f"🤖 Using local LLM via Ollama: {self.model_name}")
        return Ollama(
            model=self.model_name,
            temperature=self.temperature
        )
    
    def set_prompt_template(self, template: str, input_variables: List[str] = None):
        """
        Set a custom prompt template
        
        Args:
            template: Prompt template string
            input_variables: List of input variable names
        """
        input_variables = input_variables or ["context", "question"]
        self.prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=template
        )
    
    def generate_response(
        self,
        question: str,
        context: str,
        **kwargs
    ) -> str:
        """
        Generate a response given a question and context
        
        Args:
            question: User's question
            context: Retrieved context from documents
            **kwargs: Additional arguments for the LLM
            
        Returns:
            Generated response string
        """
        chain = self.prompt_template | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "question": question,
            "context": context,
            **kwargs
        })
        
        return response
    
    def generate_response_from_documents(
        self,
        question: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Generate a response from a list of documents
        
        Args:
            question: User's question
            documents: List of retrieved Document objects
            
        Returns:
            Dictionary with response and source documents
        """
        # Format context from documents
        context = self._format_documents(documents)
        
        # Generate response
        response = self.generate_response(question, context)
        
        # Extract unique sources (deduplicate by source path)
        seen_sources = set()
        unique_sources = []
        for doc in documents:
            source_path = doc.metadata.get("source", "Unknown")
            if source_path not in seen_sources:
                seen_sources.add(source_path)
                # Extract just the filename for cleaner display
                from pathlib import Path
                filename = Path(source_path).name if source_path != "Unknown" else "Unknown"
                unique_sources.append({
                    "source": source_path,
                    "filename": filename,
                    "content_preview": doc.page_content[:200] + "..."
                })
        
        return {
            "answer": response,
            "sources": unique_sources,
            "num_sources": len(unique_sources)
        }
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format documents into a context string
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Document {i}] (Source: {source})\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_llm(self):
        """Get the underlying LLM object"""
        return self.llm


if __name__ == "__main__":
    # Test the LLM manager
    print("Testing LLM Manager...")
    
    try:
        # This will fail without API key, but shows the structure
        manager = LLMManager()
        
        test_context = """
        Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience without being explicitly 
        programmed. It focuses on developing computer programs that can access 
        data and use it to learn for themselves.
        """
        
        test_question = "What is machine learning?"
        
        response = manager.generate_response(test_question, test_context)
        print(f"Response: {response}")
        
    except ValueError as e:
        print(f"Note: {e}")
        print("This is expected if no API key is configured.")
