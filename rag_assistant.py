"""
Intelligent RAG Assistant for Customer Support
Uses customer_orders_and_policies.txt as the primary document store for retrieval-augmented generation.

This implementation follows the exact template requested:
1. Retrieve relevant chunks from the document store using the retriever
2. Combine them as context for the model
3. Use a chain to process user queries and generate answers grounded in that context
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import groq
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata"""
    text: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Result from document retrieval with relevance score"""
    chunk: DocumentChunk
    score: float

class DocumentRetriever:
    """
    Retrieves relevant chunks from the customer orders and policies document.
    Uses TF-IDF vectorization for semantic similarity search.
    """
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.chunk_vectors: Optional[np.ndarray] = None
        
        # Load and process the document
        self._load_and_chunk_document()
        self._create_vector_index()
    
    def _load_and_chunk_document(self):
        """Load the document and split it into meaningful chunks"""
        try:
            with open(self.document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split document into logical sections
            chunks = self._intelligent_chunking(content)
            
            for i, chunk_text in enumerate(chunks):
                chunk = DocumentChunk(
                    text=chunk_text.strip(),
                    source=os.path.basename(self.document_path),
                    chunk_id=f"chunk_{i:03d}",
                    metadata={
                        "chunk_index": i,
                        "char_count": len(chunk_text),
                        "source_file": self.document_path
                    }
                )
                self.chunks.append(chunk)
            
            logger.info(f"Created {len(self.chunks)} chunks from {self.document_path}")
            
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise
    
    def _intelligent_chunking(self, content: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Intelligently chunk the document preserving order boundaries and policy sections
        """
        chunks = []
        
        # First, split by major sections (marked by ====== lines)
        major_sections = re.split(r'={50,}', content)
        
        for section in major_sections:
            section = section.strip()
            if not section:
                continue
            
            # Check if this section contains order information
            if 'ORDER #' in section and '--------------------------------------------------------------------------------' in section:
                # Split by order separators to keep complete orders together
                order_chunks = self._chunk_orders(section)
                chunks.extend(order_chunks)
            else:
                # For policy sections, chunk by logical breaks
                policy_chunks = self._chunk_policies(section, max_chunk_size)
                chunks.extend(policy_chunks)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _chunk_orders(self, orders_section: str) -> List[str]:
        """Chunk order section keeping complete orders together"""
        # Split by the long dash separator between orders
        order_separator = '-' * 80
        orders = orders_section.split(order_separator)
        
        chunks = []
        for order in orders:
            order = order.strip()
            if order and len(order) > 50:  # Filter out very short fragments
                chunks.append(order)
        
        return chunks
    
    def _chunk_policies(self, policy_section: str, max_size: int) -> List[str]:
        """Chunk policy sections by logical breaks"""
        # Split by double newlines first (paragraphs)
        paragraphs = [p.strip() for p in policy_section.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_vector_index(self):
        """Create TF-IDF vectors for all chunks"""
        try:
            if not self.chunks:
                logger.warning("No chunks available for vectorization")
                return
            
            # Extract text from all chunks
            chunk_texts = [chunk.text for chunk in self.chunks]
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Include bigrams for better context
                stop_words='english',
                lowercase=True
            )
            
            # Fit and transform the chunks
            self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
            
            logger.info(f"Created TF-IDF index with {self.chunk_vectors.shape[0]} chunks and {self.chunk_vectors.shape[1]} features")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for the given query
        """
        if not self.vectorizer or self.chunk_vectors is None:
            logger.warning("Vector index not available")
            return []
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate cosine similarity with all chunks
            similarities = (self.chunk_vectors @ query_vector.T).toarray().flatten()
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Only include chunks with positive similarity
                    result = RetrievalResult(
                        chunk=self.chunks[idx],
                        score=float(similarities[idx])
                    )
                    results.append(result)
            
            logger.info(f"Retrieved {len(results)} relevant chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []

class RAGChain:
    """
    RAG Chain that combines retrieved context with user questions to generate grounded answers
    """
    
    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.groq_client = groq.Groq(api_key=groq_api_key)
        self.model = model
        
        # Test the connection
        self._test_connection()
    
    def _test_connection(self):
        """Test the Groq API connection"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model=self.model,
                max_tokens=10
            )
            logger.info("Successfully connected to Groq API")
        except Exception as e:
            logger.error(f"Failed to connect to Groq API: {e}")
            raise
    
    def format_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Format retrieved chunks into context for the model"""
        if not retrieval_results:
            return "No relevant information found in the document store."
        
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            chunk_text = result.chunk.text
            source = result.chunk.source
            score = result.score
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {score:.3f})]\n{chunk_text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate an answer based on the provided context and question"""
        
        # Create the prompt using the exact template format requested
        prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question:
{question}

Answer:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful customer service assistant. Use only the retrieved information to generate concise, correct, and contextually relevant responses. If the answer cannot be found in the provided context, respond with: 'I'm not sure about that based on the given information.'"
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=512
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm experiencing technical difficulties. Please try again later."

class RAGAssistant:
    """
    Main RAG Assistant that combines document retrieval with answer generation
    """
    
    def __init__(self, document_path: str, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.document_path = document_path
        
        # Initialize components
        self.retriever = DocumentRetriever(document_path)
        self.rag_chain = RAGChain(groq_api_key, model)
        
        logger.info("RAG Assistant initialized successfully")
    
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a user question and return a grounded answer
        
        Example flow:
        - User asks a question
        - Retriever fetches the top k most relevant chunks
        - Chain formats the context and user question into a final prompt
        - Model generates a grounded, factual answer
        """
        
        # Step 1: Retrieve relevant chunks from the document store
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieval_results:
            return {
                "answer": "I'm not sure about that based on the given information.",
                "sources": [],
                "context_used": "",
                "confidence": 0.0
            }
        
        # Step 2: Combine them as context for the model
        context = self.rag_chain.format_context(retrieval_results)
        
        # Step 3: Use chain to process user query and generate answer grounded in context
        answer = self.rag_chain.generate_answer(context, question)
        
        # Prepare response with metadata
        sources = [
            {
                "source": result.chunk.source,
                "chunk_id": result.chunk.chunk_id,
                "relevance_score": result.score,
                "preview": result.chunk.text[:200] + "..." if len(result.chunk.text) > 200 else result.chunk.text
            }
            for result in retrieval_results
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "confidence": max(result.score for result in retrieval_results) if retrieval_results else 0.0,
            "num_sources": len(retrieval_results)
        }

def load_config() -> Dict[str, str]:
    """Load configuration from environment or config file"""
    config = {}
    
    # Try environment variable first
    config['groq_api_key'] = os.getenv('GROQ_API_KEY')
    config['model'] = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
    
    # If not in environment, try config file
    if not config['groq_api_key']:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config['groq_api_key'] = file_config.get('api_key')
                config['model'] = file_config.get('model', 'llama-3.1-8b-instant')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    if not config['groq_api_key'] or config['groq_api_key'] == "your-groq-api-key-here":
        raise ValueError(
            "Groq API key not found. Please either:\n"
            "1. Set GROQ_API_KEY environment variable, or\n"
            "2. Add your API key to config.json"
        )
    
    return config

def main():
    """Example usage of the RAG Assistant"""
    try:
        # Load configuration
        config = load_config()
        
        # Path to the customer orders and policies document
        document_path = os.path.join(os.path.dirname(__file__), 'company', 'customer_orders_and_policies.txt')
        
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"Document not found: {document_path}")
        
        # Initialize the RAG Assistant
        assistant = RAGAssistant(
            document_path=document_path,
            groq_api_key=config['groq_api_key'],
            model=config['model']
        )
        
        # Example questions
        example_questions = [
            "What is the return policy?",
            "How long does standard shipping take?",
            "What are the warranty options available?",
            "Tell me about order TM-2024-101234",
            "What payment methods do you accept?",
            "How much does express shipping cost?"
        ]
        
        print("🤖 RAG Assistant Demo")
        print("=" * 50)
        print("Using document:", os.path.basename(document_path))
        print("Model:", config['model'])
        print("=" * 50)
        
        # Interactive mode
        while True:
            print("\nExample questions:")
            for i, q in enumerate(example_questions, 1):
                print(f"  {i}. {q}")
            
            user_input = input("\nEnter your question (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process the question
            print(f"\n🔍 Processing: {user_input}")
            print("-" * 30)
            
            result = assistant.ask(user_input)
            
            # Display results
            print(f"📝 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            print(f"📚 Sources used: {result['num_sources']}")
            
            if result['sources']:
                print("\n📖 Source details:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source']} (Score: {source['relevance_score']:.3f})")
                    print(f"     Preview: {source['preview']}")
            
            print("\n" + "=" * 50)
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
