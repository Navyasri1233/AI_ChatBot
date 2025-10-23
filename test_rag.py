"""
Test script for the RAG Assistant
Demonstrates the retrieval-augmented generation system using customer_orders_and_policies.txt
"""

import os
import sys
from rag_assistant import RAGAssistant, load_config

def test_rag_system():
    """Test the RAG system with sample questions"""
    
    print("🧪 Testing RAG Assistant")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        
        # Path to the customer orders and policies document
        document_path = os.path.join(os.path.dirname(__file__), 'company', 'customer_orders_and_policies.txt')
        
        if not os.path.exists(document_path):
            print(f"❌ Document not found: {document_path}")
            return False
        
        print(f"📄 Using document: {os.path.basename(document_path)}")
        print(f"🤖 Model: {config['model']}")
        print()
        
        # Initialize the RAG Assistant
        assistant = RAGAssistant(
            document_path=document_path,
            groq_api_key=config['groq_api_key'],
            model=config['model']
        )
        
        # Test questions that should be answerable from the document
        test_questions = [
            {
                "question": "What is the return policy?",
                "expected_keywords": ["30-day", "return", "original condition"]
            },
            {
                "question": "How much does express shipping cost?",
                "expected_keywords": ["express", "shipping", "$15.99", "2-3 business days"]
            },
            {
                "question": "Tell me about order TM-2024-101234",
                "expected_keywords": ["Emily Chen", "Sony", "headphones"]
            },
            {
                "question": "What warranty options are available?",
                "expected_keywords": ["warranty", "manufacturer", "extended"]
            },
            {
                "question": "What are the cancellation policies?",
                "expected_keywords": ["cancel", "policy"]
            }
        ]
        
        print("🔍 Running test questions...")
        print("-" * 30)
        
        for i, test in enumerate(test_questions, 1):
            question = test["question"]
            expected_keywords = test["expected_keywords"]
            
            print(f"\n{i}. Question: {question}")
            
            # Get answer from RAG system
            result = assistant.ask(question, top_k=3)
            
            answer = result["answer"]
            confidence = result["confidence"]
            num_sources = result["num_sources"]
            
            print(f"   Answer: {answer}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Sources: {num_sources}")
            
            # Check if expected keywords are present
            answer_lower = answer.lower()
            found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
            
            if found_keywords:
                print(f"   ✅ Found keywords: {found_keywords}")
            else:
                print(f"   ⚠️  Expected keywords not found: {expected_keywords}")
            
            # Show sources
            if result["sources"]:
                print(f"   📚 Top source: {result['sources'][0]['source']} (Score: {result['sources'][0]['relevance_score']:.3f})")
        
        print("\n" + "=" * 50)
        print("✅ RAG system test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def interactive_demo():
    """Interactive demo of the RAG system"""
    
    try:
        # Load configuration
        config = load_config()
        
        # Path to the customer orders and policies document
        document_path = os.path.join(os.path.dirname(__file__), 'company', 'customer_orders_and_policies.txt')
        
        # Initialize the RAG Assistant
        assistant = RAGAssistant(
            document_path=document_path,
            groq_api_key=config['groq_api_key'],
            model=config['model']
        )
        
        print("\n🤖 Interactive RAG Demo")
        print("=" * 50)
        print("Ask questions about TechMart Global's policies and orders!")
        print("Type 'quit' to exit.\n")
        
        while True:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n🔍 Processing: {question}")
            print("-" * 30)
            
            # Get answer
            result = assistant.ask(question)
            
            # Display results
            print(f"📝 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            
            if result['sources']:
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source']} (Relevance: {source['relevance_score']:.3f})")
            
            print("\n" + "=" * 50)
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    # Run tests first
    success = test_rag_system()
    
    if success:
        # If tests pass, offer interactive demo
        response = input("\n🎮 Would you like to try the interactive demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo()
    else:
        print("❌ Tests failed. Please check your configuration.")
