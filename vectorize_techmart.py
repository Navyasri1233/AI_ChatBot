"""
TechMart Global Document Vectorization Script
==============================================
This script intelligently parses, chunks, and vectorizes the TechMart Global
e-commerce document for optimal RAG (Retrieval Augmented Generation) performance.

It creates semantic chunks with rich metadata for accurate policy and order retrieval.
"""

import os
import sys
import json
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import Groq for embeddings
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not installed. Run: pip install groq")

from sklearn.feature_extraction.text import TfidfVectorizer


class TechMartVectorizer:
    """Intelligent document vectorizer for TechMart Global e-commerce data."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the vectorizer with optional Groq API key."""
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_client = None
        self.embedding_model = "text-embedding-3-small"
        self.use_groq = False
        
        # Initialize Groq client if available
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.groq_client = groq.Groq(api_key=self.groq_api_key)
                self.use_groq = True
                logger.info("Groq embeddings initialized successfully")
            except Exception as e:
                logger.warning(f"Groq initialization failed: {e}. Will use TF-IDF fallback.")
        else:
            logger.info("Using TF-IDF embeddings (Groq not available)")
        
        # Storage
        self.chunks: List[Dict[str, Any]] = []
        self.metadata_version = "1.0"
        self.last_updated = "October 5, 2024"
    
    def parse_techmart_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the TechMart document into semantic chunks with metadata.
        
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        logger.info(f"Reading document: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = []
        
        # 1. Extract and chunk Company Overview
        chunks.extend(self._extract_company_overview(content))
        
        # 2. Extract and chunk each Policy section
        chunks.extend(self._extract_policies(content))
        
        # 3. Extract and chunk each Customer Order
        chunks.extend(self._extract_orders(content))
        
        # 4. Extract and chunk FAQs
        chunks.extend(self._extract_faqs(content))
        
        # 5. Extract and chunk Contact Information
        chunks.extend(self._extract_contact_info(content))
        
        logger.info(f"Parsed {len(chunks)} semantic chunks from document")
        return chunks
    
    def _extract_company_overview(self, content: str) -> List[Dict[str, Any]]:
        """Extract company overview section."""
        chunks = []
        
        # Pattern to match company overview section
        pattern = r'COMPANY OVERVIEW:(.*?)(?=\n={50,}|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            overview_text = match.group(1).strip()
            chunks.append({
                'text': f"TechMart Global Company Overview:\n{overview_text}",
                'metadata': {
                    'section': 'Company Overview',
                    'subsection': 'General Information',
                    'category': 'company_info',
                    'last_updated': self.last_updated,
                    'version': self.metadata_version
                }
            })
        
        return chunks
    
    def _extract_policies(self, content: str) -> List[Dict[str, Any]]:
        """Extract all policy sections with detailed metadata."""
        chunks = []
        
        # Define policy sections to extract
        policies = [
            'SHIPPING POLICY',
            'RETURN POLICY',
            'REPLACEMENT POLICY',
            'WARRANTY INFORMATION',
            'CANCELLATION POLICY',
            'PRICE MATCH GUARANTEE'
        ]
        
        for policy_name in policies:
            # Pattern to extract each policy section
            pattern = rf'{policy_name}:(.*?)(?=\n[A-Z]{{2,}} [A-Z]|\n={50,}|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                policy_text = match.group(1).strip()
                
                # Create comprehensive policy chunk
                full_text = f"{policy_name}:\n{policy_text}"
                
                chunks.append({
                    'text': full_text,
                    'metadata': {
                        'section': 'Company Policies',
                        'subsection': policy_name.title(),
                        'category': 'policy',
                        'policy_type': policy_name.lower().replace(' ', '_'),
                        'last_updated': self.last_updated,
                        'version': self.metadata_version
                    }
                })
                
                logger.info(f"Extracted policy: {policy_name}")
        
        return chunks
    
    def _extract_orders(self, content: str) -> List[Dict[str, Any]]:
        """Extract individual customer orders with rich metadata."""
        chunks = []
        
        # Pattern to match order sections
        order_pattern = r'ORDER #(\d+)\n-+\n(.*?)(?=\nORDER #|\n={50,}|\Z)'
        matches = re.finditer(order_pattern, content, re.DOTALL)
        
        for match in matches:
            order_num = match.group(1)
            order_content = match.group(2).strip()
            
            # Extract key order details using regex
            order_details = self._parse_order_details(order_content)
            
            # Create comprehensive order text
            full_text = f"ORDER #{order_num}\n{order_content}"
            
            # Build metadata
            metadata = {
                'section': 'Order Details',
                'subsection': f'Order #{order_num}',
                'category': 'customer_order',
                'order_number': order_details.get('order_number', f'TM-2024-{order_num}'),
                'customer_name': order_details.get('customer_name', ''),
                'customer_email': order_details.get('customer_email', ''),
                'order_date': order_details.get('order_date', ''),
                'order_status': order_details.get('order_status', ''),
                'order_total': order_details.get('order_total', ''),
                'last_updated': self.last_updated,
                'version': self.metadata_version
            }
            
            chunks.append({
                'text': full_text,
                'metadata': metadata
            })
            
            # Also create a summary chunk for quick order lookup
            summary_text = self._create_order_summary(order_details, order_num)
            chunks.append({
                'text': summary_text,
                'metadata': {
                    **metadata,
                    'subsection': f'Order #{order_num} Summary',
                    'category': 'order_summary'
                }
            })
            
            logger.info(f"Extracted Order #{order_num}: {order_details.get('order_number', '')}")
        
        return chunks
    
    def _parse_order_details(self, order_text: str) -> Dict[str, str]:
        """Parse key details from order text."""
        details = {}
        
        # Extract order number
        match = re.search(r'Order Number:\s*([A-Z0-9-]+)', order_text)
        if match:
            details['order_number'] = match.group(1)
        
        # Extract customer name
        match = re.search(r'Customer Name:\s*([^\n]+)', order_text)
        if match:
            details['customer_name'] = match.group(1).strip()
        
        # Extract customer email
        match = re.search(r'Customer Email:\s*([^\n]+)', order_text)
        if match:
            details['customer_email'] = match.group(1).strip()
        
        # Extract order date
        match = re.search(r'Order Date:\s*([^\n]+)', order_text)
        if match:
            details['order_date'] = match.group(1).strip()
        
        # Extract order status
        match = re.search(r'Order Status:\s*([^\n]+)', order_text)
        if match:
            details['order_status'] = match.group(1).strip()
        
        # Extract order total
        match = re.search(r'ORDER TOTAL:\s*\$([0-9,\.]+)', order_text)
        if match:
            details['order_total'] = f"${match.group(1)}"
        
        # Extract tracking number
        match = re.search(r'Tracking Number:\s*([^\n]+)', order_text)
        if match:
            details['tracking_number'] = match.group(1).strip()
        
        return details
    
    def _create_order_summary(self, details: Dict[str, str], order_num: str) -> str:
        """Create a concise order summary for quick retrieval."""
        order_number = details.get('order_number', f'TM-2024-{order_num}')
        customer = details.get('customer_name', 'Unknown Customer')
        date = details.get('order_date', 'Unknown Date')
        status = details.get('order_status', 'Unknown Status')
        total = details.get('order_total', 'Unknown Total')
        
        summary = f"""Order Summary - {order_number}
Customer: {customer}
Email: {details.get('customer_email', 'N/A')}
Order Date: {date}
Status: {status}
Total: {total}
Tracking: {details.get('tracking_number', 'Not available')}
"""
        return summary
    
    def _extract_faqs(self, content: str) -> List[Dict[str, Any]]:
        """Extract FAQ items individually for precise retrieval."""
        chunks = []
        
        # Pattern to match FAQ section
        faq_section_pattern = r'FREQUENTLY ASKED QUESTIONS\n={50,}(.*?)(?=\n={50,}|\Z)'
        faq_section_match = re.search(faq_section_pattern, content, re.DOTALL)
        
        if faq_section_match:
            faq_content = faq_section_match.group(1)
            
            # Pattern to match individual Q&A pairs
            qa_pattern = r'Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|$)'
            qa_matches = re.finditer(qa_pattern, faq_content, re.DOTALL)
            
            for qa_match in qa_matches:
                question = qa_match.group(1).strip()
                answer = qa_match.group(2).strip()
                
                full_text = f"Q: {question}\nA: {answer}"
                
                chunks.append({
                    'text': full_text,
                    'metadata': {
                        'section': 'FAQ',
                        'subsection': question[:50] + '...' if len(question) > 50 else question,
                        'category': 'faq',
                        'question': question,
                        'last_updated': self.last_updated,
                        'version': self.metadata_version
                    }
                })
                
            logger.info(f"Extracted {len(list(re.finditer(qa_pattern, faq_content)))} FAQ items")
        
        return chunks
    
    def _extract_contact_info(self, content: str) -> List[Dict[str, Any]]:
        """Extract contact information section."""
        chunks = []
        
        # Pattern to match contact information section
        pattern = r'CONTACT INFORMATION\n={50,}(.*?)(?=\n={50,}|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            contact_text = match.group(1).strip()
            chunks.append({
                'text': f"TechMart Global Contact Information:\n{contact_text}",
                'metadata': {
                    'section': 'Contact Information',
                    'subsection': 'Company Contacts',
                    'category': 'contact_info',
                    'last_updated': self.last_updated,
                    'version': self.metadata_version
                }
            })
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            List of chunk dictionaries with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        if self.use_groq and self.groq_client:
            return self._generate_groq_embeddings(chunks)
        else:
            return self._generate_tfidf_embeddings(chunks)
    
    def _generate_groq_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings using Groq API."""
        result_chunks = []
        batch_size = 20  # Process in batches to avoid rate limits
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk['text'] for chunk in batch]
            
            try:
                # Try primary embedding model
                response = self.groq_client.embeddings.create(
                    model=self.embedding_model,
                    input=texts
                )
                
                embeddings = [item.embedding for item in response.data]
                
                for chunk, embedding in zip(batch, embeddings):
                    result_chunks.append({
                        'text': chunk['text'],
                        'embedding': embedding,
                        'metadata': chunk['metadata']
                    })
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Groq embedding failed for batch {i//batch_size + 1}: {e}")
                logger.info("Falling back to TF-IDF for this batch")
                
                # Fallback to TF-IDF for this batch
                for chunk in batch:
                    result_chunks.append({
                        'text': chunk['text'],
                        'embedding': [],  # Empty embedding = use TF-IDF at query time
                        'metadata': chunk['metadata']
                    })
        
        return result_chunks
    
    def _generate_tfidf_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate TF-IDF embeddings (stored as empty for runtime computation)."""
        logger.info("Using TF-IDF backend - embeddings will be computed at query time")
        
        result_chunks = []
        for chunk in chunks:
            result_chunks.append({
                'text': chunk['text'],
                'embedding': [],  # Empty = compute TF-IDF at query time
                'metadata': chunk['metadata']
            })
        
        return result_chunks
    
    def save_to_vector_store(self, chunks: List[Dict[str, Any]], output_path: str):
        """
        Save vectorized chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries with embeddings and metadata
            output_path: Path to save the vector store JSON
        """
        logger.info(f"Saving {len(chunks)} chunks to {output_path}")
        
        # Load existing vector store if it exists
        existing_chunks = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
                logger.info(f"Loaded {len(existing_chunks)} existing chunks")
            except Exception as e:
                logger.warning(f"Could not load existing vector store: {e}")
        
        # Filter out old TechMart chunks (based on metadata)
        filtered_chunks = [
            chunk for chunk in existing_chunks
            if chunk.get('metadata', {}).get('category') not in [
                'company_info', 'policy', 'customer_order', 'order_summary', 'faq', 'contact_info'
            ]
        ]
        
        if len(filtered_chunks) < len(existing_chunks):
            logger.info(f"Removed {len(existing_chunks) - len(filtered_chunks)} old TechMart chunks")
        
        # Add new chunks
        all_chunks = filtered_chunks + chunks
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Successfully saved {len(all_chunks)} total chunks to vector store")
        logger.info(f"   - {len(chunks)} new TechMart chunks")
        logger.info(f"   - {len(filtered_chunks)} existing chunks from other sources")
    
    def vectorize_document(self, input_file: str, output_file: str):
        """
        Complete vectorization pipeline.
        
        Args:
            input_file: Path to TechMart document
            output_file: Path to save vector store
        """
        logger.info("=" * 70)
        logger.info("TechMart Global Document Vectorization Pipeline")
        logger.info("=" * 70)
        
        # Step 1: Parse document into semantic chunks
        chunks = self.parse_techmart_document(input_file)
        
        # Step 2: Generate embeddings
        vectorized_chunks = self.generate_embeddings(chunks)
        
        # Step 3: Save to vector store
        self.save_to_vector_store(vectorized_chunks, output_file)
        
        # Print summary
        self._print_summary(vectorized_chunks)
    
    def _print_summary(self, chunks: List[Dict[str, Any]]):
        """Print vectorization summary statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("VECTORIZATION SUMMARY")
        logger.info("=" * 70)
        
        # Count by category
        category_counts = {}
        for chunk in chunks:
            category = chunk['metadata'].get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"\n📊 Total Chunks: {len(chunks)}")
        logger.info(f"\n📁 Breakdown by Category:")
        for category, count in sorted(category_counts.items()):
            logger.info(f"   - {category}: {count} chunks")
        
        # Sample chunks
        logger.info(f"\n📝 Sample Chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            logger.info(f"\n   Chunk {i}:")
            logger.info(f"   Category: {chunk['metadata'].get('category')}")
            logger.info(f"   Section: {chunk['metadata'].get('section')}")
            logger.info(f"   Text Preview: {chunk['text'][:100]}...")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Vectorization Complete!")
        logger.info("=" * 70)


def main():
    """Main execution function."""
    # File paths
    script_dir = Path(__file__).parent
    input_file = script_dir / "company" / "customer_orders_and_policies.txt"
    output_file = script_dir / "vector_store.json"
    
    # Check if input file exists
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Initialize vectorizer
    vectorizer = TechMartVectorizer()
    
    # Run vectorization pipeline
    vectorizer.vectorize_document(str(input_file), str(output_file))


if __name__ == "__main__":
    main()
