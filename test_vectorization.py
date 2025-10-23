"""
Test script to verify TechMart vectorization and retrieval.
This script demonstrates how to query the vectorized document.
"""

import json
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import math

def load_vector_store(path: str = "vector_store.json") -> List[Dict[str, Any]]:
    """Load the vector store from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tfidf_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Simple TF-IDF based search."""
    corpus = [chunk['text'] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    # Fit and transform corpus
    corpus_vectors = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_vector, corpus_vectors).flatten()
    
    # Get top-k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append({
                'chunk': chunks[idx],
                'score': float(similarities[idx])
            })
    
    return results

def filter_by_metadata(chunks: List[Dict[str, Any]], filters: Dict[str, str]) -> List[Dict[str, Any]]:
    """Filter chunks by metadata fields."""
    filtered = []
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        match = True
        for key, value in filters.items():
            if metadata.get(key) != value:
                match = False
                break
        if match:
            filtered.append(chunk)
    return filtered

def print_results(results: List[Dict[str, Any]], query: str):
    """Pretty print search results."""
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}\n")
    
    if not results:
        print("❌ No results found.\n")
        return
    
    for i, result in enumerate(results, 1):
        chunk = result['chunk']
        score = result['score']
        metadata = chunk.get('metadata', {})
        
        print(f"Result #{i} (Score: {score:.3f})")
        print(f"Section: {metadata.get('section', 'N/A')}")
        print(f"Category: {metadata.get('category', 'N/A')}")
        
        if 'order_number' in metadata:
            print(f"Order: {metadata.get('order_number')}")
            print(f"Customer: {metadata.get('customer_name')}")
        
        if 'question' in metadata:
            print(f"Question: {metadata.get('question')}")
        
        print(f"\nText Preview:")
        text = chunk['text']
        preview = text[:300] + "..." if len(text) > 300 else text
        print(preview)
        print(f"\n{'-'*70}\n")

def run_test_queries():
    """Run a series of test queries to demonstrate vectorization."""
    print("\n" + "="*70)
    print("TechMart Global - RAG Vectorization Test")
    print("="*70)
    
    # Load vector store
    chunks = load_vector_store()
    print(f"\n✅ Loaded {len(chunks)} chunks from vector store")
    
    # Filter to only TechMart chunks for testing
    techmart_chunks = [c for c in chunks if c.get('metadata', {}).get('last_updated') == 'October 5, 2024']
    print(f"✅ Found {len(techmart_chunks)} TechMart chunks\n")
    
    # Test 1: Policy Query
    print("\n" + "="*70)
    print("TEST 1: Policy Retrieval")
    print("="*70)
    query = "What is your return policy?"
    results = tfidf_search(query, techmart_chunks, top_k=2)
    print_results(results, query)
    
    # Test 2: Order Lookup
    print("\n" + "="*70)
    print("TEST 2: Order Number Search")
    print("="*70)
    order_chunks = filter_by_metadata(techmart_chunks, {'order_number': 'TM-2024-100457'})
    if order_chunks:
        print(f"\n✅ Found {len(order_chunks)} chunks for order TM-2024-100457\n")
        for chunk in order_chunks[:1]:
            metadata = chunk['metadata']
            print(f"Order: {metadata.get('order_number')}")
            print(f"Customer: {metadata.get('customer_name')}")
            print(f"Email: {metadata.get('customer_email')}")
            print(f"Status: {metadata.get('order_status')}")
            print(f"Total: {metadata.get('order_total')}")
    
    # Test 3: FAQ Search
    print("\n" + "="*70)
    print("TEST 3: FAQ Retrieval")
    print("="*70)
    query = "How do I track my package?"
    results = tfidf_search(query, techmart_chunks, top_k=2)
    print_results(results, query)
    
    # Test 4: Shipping Policy
    print("\n" + "="*70)
    print("TEST 4: Shipping Information")
    print("="*70)
    query = "How long does express shipping take?"
    results = tfidf_search(query, techmart_chunks, top_k=2)
    print_results(results, query)
    
    # Test 5: Customer Search
    print("\n" + "="*70)
    print("TEST 5: Customer Email Search")
    print("="*70)
    customer_chunks = filter_by_metadata(techmart_chunks, {'customer_email': 'sarah.mitchell@email.com'})
    if customer_chunks:
        print(f"\n✅ Found {len(customer_chunks)} chunks for sarah.mitchell@email.com\n")
        for chunk in customer_chunks[:1]:
            metadata = chunk['metadata']
            print(f"Order: {metadata.get('order_number')}")
            print(f"Status: {metadata.get('order_status')}")
            print(f"Total: {metadata.get('order_total')}")
            print(f"Date: {metadata.get('order_date')}")
    
    # Test 6: Category Filtering
    print("\n" + "="*70)
    print("TEST 6: Category Statistics")
    print("="*70)
    categories = {}
    for chunk in techmart_chunks:
        cat = chunk.get('metadata', {}).get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Chunks by Category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} chunks")
    
    print("\n" + "="*70)
    print("✅ All Tests Completed Successfully!")
    print("="*70)

if __name__ == "__main__":
    run_test_queries()
