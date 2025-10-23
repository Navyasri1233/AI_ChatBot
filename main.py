'''1. AI Powered Customer Support Assistant for E-commerce
Problem: US retailers struggle with 24 by 7 multilingual customer support and long response times
Solution: Built a real-time chatbot using large language models integrated with customer history and product catalog. 
Used retrieval augmented generation to ensure responses were accurate and grounded.
Impact: Reduced customer service tickets by 40 percent and improved resolution speed with minimal human escalation.'''

import os
import sys
import json
import logging
import math
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import groq
import sqlite3
from pathlib import Path
import asyncio
import aiohttp
from pathlib import Path
try:
    from pypdf import PdfReader  # lightweight PDF text extraction
except Exception:
    PdfReader = None
try:
    import docx  # python-docx
except Exception:
    docx = None
from sklearn.feature_extraction.text import TfidfVectorizer

# Import RAG components
from rag_assistant import DocumentRetriever, RAGChain, RetrievalResult

# ---------------------- Internationalization (i18n) ----------------------
# Minimal translation layer for CLI strings and assistant messages
LANG_STRINGS = {
    "en": {
        "welcome": "\n🤖 Welcome to the E-commerce Customer Support Assistant!",
        "instructions": "Type your question below. Type 'quit' to exit.\n",
        "prompt_email": "Enter your email: ",
        "prompt_order_id": "Enter your order ID: ",
        "prompt_language": "Select language (en/es): ",
        "you": "\n🧑 You: ",
        "goodbye": "👋 Thank you for chatting with us. Goodbye!",
        "assistant_prefix": "🤖 Assistant: ",
        "escalation": "⚠ Escalation required. Connecting you to a human agent...",
        "sources": "📚 Sources: ",
        "divider": "-" * 50,
        "escalate_text": "I understand this is an important matter that requires personalized attention. Let me connect you with one of our specialist representatives.",
        "tech_difficulties": "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
        "system_preamble": "You are a helpful customer service agent for TechMart Global, an e-commerce company. CRITICAL INSTRUCTIONS:\n1. NEVER calculate totals - always use the exact 'ORDER TOTAL' or 'REVISED ORDER TOTAL' stated in the context\n2. Always cite EXACT prices, numbers, and dates from the provided context - never estimate or approximate\n3. For order totals, search for 'ORDER TOTAL:' or 'REVISED ORDER TOTAL:' and quote that exact amount\n4. If a specific number (price, date, percentage, time) is in the context, quote it verbatim - DO NOT calculate\n5. If information is not in the context, say 'I don't have that information' rather than guessing\n6. Be professional, accurate, and concise\n7. For policy questions, quote the exact policy text when relevant",
        "relevant_policies": "Relevant policies:",
        "customer_info": "Customer info:",
        "order_details": "Order Details:",
        "order_id": "Order ID",
        "status": "Status",
        "order_date": "Order Date",
        "estimated_delivery": "Estimated Delivery",
        "tracking_number": "Tracking Number",
        "product": "Product",
        "recent_orders": "Recent Orders:",
        "order": "Order",
        "date": "Date"
    },
    "es": {
        "welcome": "\n🤖 ¡Bienvenido al Asistente de Atención al Cliente de Comercio Electrónico!",
        "instructions": "Escribe tu pregunta abajo. Escribe 'quit' para salir.\n",
        "prompt_email": "Introduce tu correo electrónico: ",
        "prompt_order_id": "Introduce tu número de pedido: ",
        "prompt_language": "Selecciona idioma (en/es): ",
        "you": "\n🧑 Tú: ",
        "goodbye": "👋 ¡Gracias por chatear con nosotros! ¡Hasta luego!",
        "assistant_prefix": "🤖 Asistente: ",
        "escalation": "⚠ Se requiere escalamiento. Conectándote con un agente humano...",
        "sources": "📚 Fuentes: ",
        "divider": "-" * 50,
        "escalate_text": "Entiendo que este es un asunto importante que requiere atención personalizada. Permíteme conectarte con uno de nuestros especialistas.",
        "tech_difficulties": "Lo siento, estoy experimentando dificultades técnicas. Inténtalo de nuevo en un momento.",
        "system_preamble": "Eres un agente de atención al cliente útil para TechMart Global, una empresa de comercio electrónico. INSTRUCCIONES CRÍTICAS:\n1. NUNCA calcules totales - siempre usa el 'ORDER TOTAL' o 'REVISED ORDER TOTAL' exacto indicado en el contexto\n2. Siempre cita precios, números y fechas EXACTOS del contexto proporcionado - nunca estimes o aproximes\n3. Para totales de pedidos, busca 'ORDER TOTAL:' o 'REVISED ORDER TOTAL:' y cita esa cantidad exacta\n4. Si un número específico (precio, fecha, porcentaje, tiempo) está en el contexto, cítalo textualmente - NO calcules\n5. Si la información no está en el contexto, di 'No tengo esa información' en lugar de adivinar\n6. Sé profesional, preciso y conciso\n7. Para preguntas sobre políticas, cita el texto exacto de la política cuando sea relevante",
        "relevant_policies": "Políticas relevantes:",
        "customer_info": "Información del cliente:",
        "order_details": "Detalles del pedido:",
        "order_id": "ID de pedido",
        "status": "Estado",
        "order_date": "Fecha del pedido",
        "estimated_delivery": "Entrega estimada",
        "tracking_number": "Número de seguimiento",
        "product": "Producto",
        "recent_orders": "Pedidos recientes:",
        "order": "Pedido",
        "date": "Fecha"
    }
}

# Simple on-disk cache for dynamic translations
_I18N_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'i18n_cache.json')
_i18n_cache: Dict[str, Dict[str, str]] = {}

def _load_i18n_cache() -> None:
    global _i18n_cache
    try:
        if os.path.exists(_I18N_CACHE_PATH):
            with open(_I18N_CACHE_PATH, 'r', encoding='utf-8') as f:
                _i18n_cache = json.load(f)
        else:
            _i18n_cache = {}
    except Exception:
        _i18n_cache = {}

def _save_i18n_cache() -> None:
    try:
        with open(_I18N_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(_i18n_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_load_i18n_cache()

class DynamicTranslator:
    def __init__(self, groq_client, model: str):
        self.groq_client = groq_client
        self.model = model

    def translate(self, text: str, target_lang: str) -> str:
        if not text:
            return text
        # cache hit
        if target_lang in _i18n_cache and text in _i18n_cache[target_lang]:
            return _i18n_cache[target_lang][text]
        # use Groq to translate succinctly
        try:
            prompt = (
                "Translate the following UI string into the target language. "
                "Keep emojis and punctuation. Return only the translated text, no quotes.\n" 
                f"Target language code: {target_lang}\n"
                f"Text: {text}"
            )
            resp = self.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": "You are a professional UI translator."},
                         {"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=128
            )
            translated = resp.choices[0].message.content.strip()
            _i18n_cache.setdefault(target_lang, {})[text] = translated
            _save_i18n_cache()
            return translated
        except Exception:
            # fallback to original text
            return text

# Global translator, set after Groq client creation
_dynamic_translator: Optional[DynamicTranslator] = None

def t(key: str, lang: str = "en") -> str:
    """Translate a key to the target language; dynamically translate if missing."""
    if not lang:
        lang = "en"
    base_en = LANG_STRINGS.get("en", {})
    # if key exists as literal in maps, treat value as canonical English for dynamic
    canonical_text = base_en.get(key, key)
    if lang == "en":
        return canonical_text
    # use static map if present
    language_map = LANG_STRINGS.get(lang)
    if language_map and key in language_map:
        return language_map[key]
    # dynamic translation fallback
    if _dynamic_translator is not None:
        return _dynamic_translator.translate(canonical_text, lang)
    # last resort
    return canonical_text

# Groq Configuration
@dataclass
class GroqConfig:
    """Configuration class for Groq API settings."""
    api_key: str
    model: str
    embedding_model: str
    
    @classmethod
    def load_config(cls) -> 'GroqConfig':
        """Load configuration from environment variables or config file."""
        # First try environment variable
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_MODEL")
        embedding_model = os.getenv("GROQ_EMBEDDING_MODEL")
        
        # If not in environment, try config file
        if not api_key or not model or not embedding_model:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = api_key or config.get('api_key')
                    model = model or config.get('model')
                    embedding_model = embedding_model or config.get('embedding_model')
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass
        
        if not api_key or api_key == "your-groq-api-key-here":
            raise ValueError(
                "Groq API key not found. Please either:\n"
                "1. Set GROQ_API_KEY environment variable, or\n"
                "2. Add your API key to config.json"
            )
        
        # Default/fallback models if none provided
        if not model:
            # Choose a widely available Groq model
            model = "llama-3.1-8b-instant"
        if not embedding_model:
            # Groq doesn't support embedding models, use TF-IDF fallback
            embedding_model = "tfidf-fallback"
        
        cfg = cls(api_key=api_key, model=model, embedding_model=embedding_model)
        logging.info(f"Groq config loaded. Model: {cfg.model}, Embeddings: {cfg.embedding_model}")
        return cfg

def create_groq_client(config: GroqConfig):
    """Initialize and configure the Groq API client."""
    try:
        # Initialize the Groq client with the API key
        client = groq.Groq(api_key=config.api_key)
        
        # Test connection with a simple completion, with fallback models on model_not_found
        candidate_models = [config.model, "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        last_error = None
        for candidate in candidate_models:
            try:
                logging.info(f"Creating Groq chat completion with model: {candidate}")
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=candidate,
                    temperature=0.7,
                    max_tokens=100
                )
                if candidate != config.model:
                    logging.warning(f"Configured model '{config.model}' unavailable. Falling back to '{candidate}'.")
                    config.model = candidate
                break
            except Exception as inner_e:
                # Try to detect model-not-found reliably across exceptions
                message = str(inner_e)
                status_code = getattr(getattr(inner_e, 'response', None), 'status_code', None)
                should_fallback = False
                if status_code == 404:
                    should_fallback = True
                elif any(token in message.lower() for token in ["model_not_found", "does not exist", "not exist", "unknown model", "invalid model", "404"]):
                    should_fallback = True
                last_error = inner_e
                if should_fallback:
                    logging.warning(f"Model '{candidate}' not available (detected 404/model-not-found). Trying next fallback, if any.")
                    continue
                # If it's not a model-not-found error, re-raise immediately
                raise
        else:
            # Exhausted candidates
            raise last_error if last_error else Exception("Failed to initialize Groq client.")
        
        logging.info("Successfully connected to Groq API")
        return client
    except Exception as e:
        logging.error(f"Error initializing Groq client: {str(e)}")
        raise

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_groq_api():
    """Test the Groq API with a simple prompt."""
    try:
        # Initialize the client
        config = GroqConfig.load_config()
        client = create_groq_client(config)
        
        # Test with a simple prompt
        logging.info("Attempting to generate content...")
        logging.info(f"Generating with Groq model: {config.model}")
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello and tell me what you can do in one sentence."}],
            model=config.model,
            temperature=0.7,
            max_tokens=100
        )
        
        print("\nAPI Response:")
        print("-" * 50)
        print(response.choices[0].message.content)
        print("-" * 50)
        return True
        
    except Exception as e:
        logging.error(f"Error testing Groq API: {e}")
        return False

# Note: keep test_groq_api available for manual testing, but do not auto-run here

logger = logging.getLogger(__name__)

@dataclass
class CustomerData:
    """Customer information structure"""
    customer_id: str
    email: str
    name: str
    order_history: List[Dict]
    preferences: Dict
    support_history: List[Dict]

@dataclass
class ProductInfo:
    """Product information structure"""
    product_id: str
    name: str
    description: str
    specifications: Dict
    price: float
    category: str
    inventory_status: str
    warranty_info: Dict

@dataclass
class OrderInfo:
    """Order information structure"""
    order_id: str
    customer_id: str
    products: List[Dict]
    status: str
    shipping_address: Dict
    tracking_number: Optional[str]
    order_date: datetime
    estimated_delivery: Optional[datetime]



class DatabaseManager:
    """Manages customer, product, and order data"""
    
    def __init__(self, db_path: str = "ecommerce.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                preferences TEXT,
                created_date TEXT
            )
        ''')
        
        # Create index on email for fast lookup (email is primary search key)
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_customer_email ON customers(email)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                specifications TEXT,
                price REAL,
                category TEXT,
                inventory_status TEXT,
                warranty_info TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT,
                products TEXT,
                status TEXT,
                shipping_address TEXT,
                tracking_number TEXT,
                order_date TEXT,
                estimated_delivery TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        ''')
        
        # Insert sample data - matches knowledge_base/customer_orders_and_policies.txt
        sample_customers = [
            ("CUST001", "sarah.mitchell@email.com", "Sarah Mitchell", 
             json.dumps({"language": "en", "notifications": True}), "2024-09-15"),
            ("CUST002", "j.rodriguez.tx@email.com", "James Rodriguez", 
             json.dumps({"language": "en", "notifications": True}), "2024-09-20"),
            ("CUST003", "emilychen.design@email.com", "Emily Chen", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-02"),
            ("CUST004", "mike.thompson.gamer@email.com", "Michael Thompson", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-02"),
            ("CUST005", "l.anderson.home@email.com", "Lisa Anderson", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-03"),
            ("CUST006", "davidkim.photo@email.com", "David Kim", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-03"),
            ("CUST007", "rachel.w.fitness@email.com", "Rachel Williams", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-04"),
            ("CUST008", "chris.lee.tech@email.com", "Christopher Lee", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-04"),
            ("CUST009", "jenny.martinez.edu@email.com", "Jennifer Martinez", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-05"),
            ("CUST010", "robert.johnson.home@email.com", "Robert Johnson", 
             json.dumps({"language": "en", "notifications": True}), "2024-10-05")
        ]
        
        sample_products = [
            # Order #1 products
            ("SMSG-S24U-256-TG", "Samsung Galaxy S24 Ultra 256GB", 
             "Titanium Gray",
             json.dumps({"storage": "256GB", "color": "Titanium Gray"}),
             1199.99, "Electronics", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Manufacturer warranty"})),
            ("SMSG-CASE-S24-BK", "Samsung Silicone Phone Case", 
             "Black case for Samsung Galaxy S24",
             json.dumps({"color": "Black", "material": "Silicone"}),
             29.99, "Accessories", "In Stock", 
             json.dumps({"duration": "90 days", "coverage": "Manufacturing defects"})),
            ("ANKR-45W-USBC", "Anker 45W Fast Charger with USB-C Cable", 
             "Fast charging with USB-C cable included",
             json.dumps({"power": "45W", "connector": "USB-C"}),
             34.99, "Accessories", "In Stock", 
             json.dumps({"duration": "18 months", "coverage": "Manufacturing defects"})),
            # Order #2 products
            ("APPL-MBP14-M3P-512-SB", "Apple MacBook Pro 14\" M3 Pro", 
             "18GB RAM, 512GB SSD, Space Black",
             json.dumps({"chip": "M3 Pro", "ram": "18GB", "storage": "512GB"}),
             2399.00, "Electronics", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Apple warranty"})),
            ("APPL-MGC-MOUSE-BK", "Apple Magic Mouse", 
             "Black wireless mouse",
             json.dumps({"color": "Black", "connectivity": "Bluetooth"}),
             99.00, "Accessories", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Apple warranty"})),
            ("TECH-HUB-7IN1-GY", "USB-C Hub 7-in-1 Adapter", 
             "Multi-port USB-C hub adapter",
             json.dumps({"ports": "7", "type": "USB-C"}),
             49.99, "Accessories", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Manufacturing defects"})),
            ("ACC-SLEEVE-14-NV", "Laptop Sleeve 14\"", 
             "Navy Blue protective sleeve",
             json.dumps({"size": "14 inch", "color": "Navy Blue"}),
             24.99, "Accessories", "In Stock", 
             json.dumps({"duration": "90 days", "coverage": "Manufacturing defects"})),
            # Order #3 products
            ("SONY-WH1000XM5-SV", "Sony WH-1000XM5 Wireless Noise-Cancelling Headphones", 
             "Silver premium headphones",
             json.dumps({"color": "Silver", "features": "Noise-Cancelling"}),
             399.99, "Electronics", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Manufacturer warranty"})),
            ("BOSE-SLFLEX-BK", "Bose SoundLink Flex Bluetooth Speaker", 
             "Portable Bluetooth speaker in Black",
             json.dumps({"color": "Black", "connectivity": "Bluetooth"}),
             149.99, "Electronics", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Manufacturer warranty"})),
            # Order #4 products (Gaming setup)
            ("ASUS-ROG-17-4080-I9", "ASUS ROG Strix Gaming Laptop 17.3\"", 
             "RTX 4080, i9-13980HX, 32GB RAM, 1TB SSD",
             json.dumps({"processor": "Intel i9-13980HX", "ram": "32GB", "storage": "1TB SSD", "gpu": "RTX 4080"}),
             2799.99, "Electronics", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Hardware defects"})),
            ("RAZR-DAV3-PRO-BK", "Razer DeathAdder V3 Pro Wireless Gaming Mouse", 
             "Wireless gaming mouse with ergonomic design and RGB lighting",
             json.dumps({"connectivity": "Wireless", "dpi": "30000", "battery": "90 hours"}),
             149.99, "Electronics", "In Stock", 
             json.dumps({"duration": "2 years", "coverage": "Manufacturing defects"})),
            ("HPRX-CLD3-BK", "HyperX Cloud III Gaming Headset", 
             "Premium gaming headset with 7.1 surround sound and noise cancellation",
             json.dumps({"drivers": "53mm", "frequency": "10Hz-40kHz", "impedance": "64 Ohm"}),
             129.99, "Electronics", "In Stock", 
             json.dumps({"duration": "2 years", "coverage": "Manufacturing defects"})),
            ("ACC-MPAD-RGB-XXL", "RGB Gaming Mouse Pad XXL", 
             "Extra large gaming mouse pad with RGB lighting",
             json.dumps({"size": "900x400mm", "thickness": "4mm", "lighting": "RGB"}),
             39.99, "Accessories", "In Stock", 
             json.dumps({"duration": "1 year", "coverage": "Manufacturing defects"}))
        ]
        
        sample_orders = [
            # Order #1 - TM-2024-100457
            ("TM-2024-100457", "CUST001", 
             json.dumps([{"product_id": "SMSG-S24U-256-TG", "quantity": 1, "price": 1199.99}, 
                        {"product_id": "SMSG-CASE-S24-BK", "quantity": 1, "price": 29.99},
                        {"product_id": "ANKR-45W-USBC", "quantity": 1, "price": 34.99}]),
             "Delivered", json.dumps({"street": "1523 Oak Street, Apartment 4B", "city": "Seattle", "state": "WA", "zip": "98101"}),
             "1Z999AA10123456784", "2024-10-01", "2024-10-03"),
            # Order #2 - TM-2024-100892
            ("TM-2024-100892", "CUST002", 
             json.dumps([{"product_id": "APPL-MBP14-M3P-512-SB", "quantity": 1, "price": 2399.00},
                        {"product_id": "APPL-MGC-MOUSE-BK", "quantity": 1, "price": 99.00},
                        {"product_id": "TECH-HUB-7IN1-GY", "quantity": 1, "price": 49.99},
                        {"product_id": "ACC-SLEEVE-14-NV", "quantity": 1, "price": 24.99}]),
             "In Transit", json.dumps({"street": "8904 Meadow Lane", "city": "Austin", "state": "TX", "zip": "78704"}),
             "1Z999AA10123456901", "2024-10-01", "2024-10-08"),
            # Order #3 - TM-2024-101234 (Emily Chen - FIXED to match document)
            ("TM-2024-101234", "CUST003", 
             json.dumps([{"product_id": "SONY-WH1000XM5-SV", "quantity": 2, "price": 399.99},
                        {"product_id": "BOSE-SLFLEX-BK", "quantity": 1, "price": 149.99}]),
             "Processing - Will ship by October 4, 2024", json.dumps({"street": "456 Broadway Avenue, Suite 200", "city": "New York", "state": "NY", "zip": "10013"}),
             None, "2024-10-02", ""),
            # Order #4 - TM-2024-101567 (Michael Thompson)
            ("TM-2024-101567", "CUST004", 
             json.dumps([{"product_id": "ASUS-ROG-17-4080-I9", "quantity": 1, "price": 2799.99}, 
                        {"product_id": "RAZR-DAV3-PRO-BK", "quantity": 1, "price": 149.99}, 
                        {"product_id": "HPRX-CLD3-BK", "quantity": 1, "price": 129.99}, 
                        {"product_id": "ACC-MPAD-RGB-XXL", "quantity": 1, "price": 39.99}]),
             "Delivered", json.dumps({"street": "2847 Pine Ridge Road", "city": "Denver", "state": "CO", "zip": "80202"}),
             "1Z999AA10123457012", "2024-10-02", "2024-10-05")
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?)", sample_customers)
        cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_products)
        cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_orders)
        
        conn.commit()
        conn.close()
    
    def get_customer_by_email(self, email: str) -> Optional[CustomerData]:
        """Retrieve customer data by email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM customers WHERE email = ?", (email,))
        customer = cursor.fetchone()
        
        if customer:
            cursor.execute("SELECT * FROM orders WHERE customer_id = ?", (customer[0],))
            orders = cursor.fetchall()
            
            conn.close()
            return CustomerData(
                customer_id=customer[0],
                email=customer[1],
                name=customer[2],
                order_history=[{
                    "order_id": order[0],
                    "products": json.loads(order[2]),
                    "status": order[3],
                    "order_date": order[6]
                } for order in orders],
                preferences=json.loads(customer[3]),
                support_history=[]
            )
        
        conn.close()
        return None
    
    def get_order_by_id(self, order_id: str) -> Optional[OrderInfo]:
        """Retrieve order information by order ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
        order = cursor.fetchone()
        conn.close()
        
        if order:
            # Handle empty estimated_delivery
            est_delivery = None
            if order[7] and order[7].strip():
                try:
                    est_delivery = datetime.fromisoformat(order[7])
                except ValueError:
                    est_delivery = None
            
            return OrderInfo(
                order_id=order[0],
                customer_id=order[1],
                products=json.loads(order[2]),
                status=order[3],
                shipping_address=json.loads(order[4]),
                tracking_number=order[5],
                order_date=datetime.fromisoformat(order[6]),
                estimated_delivery=est_delivery
            )
        
        return None
    
    def get_product_by_id(self, product_id: str) -> Optional[ProductInfo]:
        """Retrieve product information by product ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        product = cursor.fetchone()
        conn.close()
        
        if product:
            return ProductInfo(
                product_id=product[0],
                name=product[1],
                description=product[2],
                specifications=json.loads(product[3]),
                price=product[4],
                category=product[5],
                inventory_status=product[6],
                warranty_info=json.loads(product[7])
            )
        
        return None

class EcommerceSupport:
    """Main customer support assistant class"""
    
    def __init__(self, language: str = "en"):
        # Load Groq config and client
        self.groq_config = GroqConfig.load_config()
        self.groq_client = create_groq_client(self.groq_config)
        
        # set global dynamic translator
        global _dynamic_translator
        _dynamic_translator = DynamicTranslator(self.groq_client, self.groq_config.model)
        
        # Initialize database and conversation history
        self.db_manager = DatabaseManager()
        self.conversation_history = {}
        self.language = language if language in LANG_STRINGS else "en"
        
        # Initialize RAG components
        self.document_retriever = None
        self.rag_chain = None
        self._initialize_rag_components()
        
        # Initialize vector store for document indexing
        self._init_vector_store()
        
        self._initialize_knowledge_base()
        
    def _initialize_rag_components(self):
        """Initialize RAG components with proper error handling and directory creation"""
        self.document_retriever = None
        self.rag_chain = None
        try:
            # Ensure knowledge base directory exists with proper permissions
            knowledge_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knowledge_base')
            
            # Try to create the directory with full permissions
            os.makedirs(knowledge_base_path, mode=0o777, exist_ok=True)
            
            # Double check directory is writable
            test_file = os.path.join(knowledge_base_path, '.permission_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
                
        except (OSError, IOError) as e:
            logger.error(f"Permission error accessing knowledge base directory: {e}")
            logger.info("Please ensure the application has write permissions to the directory or run as administrator")
            logger.info(f"Knowledge base path: {knowledge_base_path}")
            self.document_retriever = None
            self.rag_chain = None
            return
        
        # Check if directory is empty
        try:
            if not os.listdir(knowledge_base_path):
                logger.warning(f"Knowledge base directory is empty: {knowledge_base_path}")
                # Create a sample document if directory is empty
                sample_doc = os.path.join(knowledge_base_path, 'customer_orders_and_policies.txt')
                if not os.path.exists(sample_doc):
                    with open(sample_doc, 'w', encoding='utf-8') as f:
                        f.write("This is a sample document. Please replace with your actual knowledge base files.\n")
                logger.info("A sample document has been created in the knowledge base")
                
            # Initialize document retriever with the actual document file path
            document_file = os.path.join(knowledge_base_path, 'customer_orders_and_policies.txt')
            if not os.path.exists(document_file):
                raise FileNotFoundError(f"Document file not found: {document_file}")
            
            self.document_retriever = DocumentRetriever(document_file)
            
            # Initialize RAG chain with Groq API key and model
            self.rag_chain = RAGChain(
                groq_api_key=self.groq_config.api_key,
                model=self.groq_config.model
            )
            logger.info("RAG components initialized successfully with knowledge base")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")
            logger.info("Falling back to LLM-only mode")
            self.document_retriever = None
            self.rag_chain = None
    
    # ---------------------- Knowledge Base Initialization ----------------------
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with company policies and procedures"""
        self.knowledge_documents = [
            {
                "content": """
                RETURN POLICY: Customers can return items within 30 days of purchase for a full refund. 
                Items must be in original condition with tags attached. Electronics need to be in original packaging.
                Return shipping is free for defective items, $7.99 for other returns.
                """,
                "metadata": {"type": "policy", "category": "returns", "last_updated": "2024-09-01"}
            },
            {
                "content": """
                SHIPPING INFORMATION: Standard shipping (5-7 business days) is free on orders over $50.
                Express shipping (2-3 business days) costs $15.99. Overnight shipping costs $29.99.
                We ship Monday through Friday. Orders placed before 2 PM EST ship same day.
                """,
                "metadata": {"type": "policy", "category": "shipping", "last_updated": "2024-09-01"}
            },
            {
                "content": """
                WARRANTY COVERAGE: Electronics come with manufacturer warranty plus our extended protection plan.
                Headphones and audio equipment: 2 years coverage including accidental damage.
                Fitness trackers and wearables: 1 year manufacturer warranty, optional 2-year extended plan.
                """,
                "metadata": {"type": "policy", "category": "warranty", "last_updated": "2024-09-01"}
            },
            {
                "content": """
                PAYMENT METHODS: We accept Visa, Mastercard, American Express, Discover, PayPal, and Apple Pay.
                For security, we may require verification for large orders or new payment methods.
                Payment is processed when items ship, not when order is placed.
                """,
                "metadata": {"type": "policy", "category": "payment", "last_updated": "2024-09-01"}
            },
            {
                "content": """
                ACCOUNT MANAGEMENT: Customers can update their profile, shipping addresses, and payment methods
                in their account dashboard. Password resets are sent via email. For security, some changes
                may require identity verification.
                """,
                "metadata": {"type": "procedure", "category": "account", "last_updated": "2024-09-01"}
            }
        ]
        logger.info(f"Initialized knowledge base with {len(self.knowledge_documents)} documents")

    # ---------------------- RAG Vector Store ----------------------
    def _init_vector_store(self):
        """Create or load a simple local vector store backed by Groq embeddings"""
        self._vector_store_path = os.path.join(os.path.dirname(__file__), 'vector_store.json')
        # configurable embedding model with default from config
        self._embedding_model = self.groq_config.embedding_model
        self._embedding_backend = 'groq'  # 'groq' or 'tfidf'
        self._embedding_switch_logged = False
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._vectors: List[Dict[str, Any]] = []
        self._current_docs_base = None  # Initialize document base path
        # Try load existing store
        try:
            if os.path.exists(self._vector_store_path):
                with open(self._vector_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._vectors = data
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")

    def _save_vector_store(self):
        try:
            with open(self._vector_store_path, 'w', encoding='utf-8') as f:
                json.dump(self._vectors, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save vector store: {e}")

    def _chunk_text(self, text: str, max_chars: int = 2500, overlap: int = 200) -> List[str]:
        """Chunk text while preserving order boundaries for e-commerce documents"""
        text = (text or '').strip()
        if not text:
            return []
        
        # Detect if this is an order document by looking for order sections
        order_separator = '-' * 60  # The separator line between orders
        
        # If document contains order sections, chunk by order
        if 'ORDER #' in text and order_separator in text:
            # Split by the separator lines to preserve complete orders
            sections = text.split(order_separator)
            chunks: List[str] = []
            current = ''
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                # If adding this section exceeds max_chars, save current and start new
                if current and len(current) + len(section) + 100 > max_chars:
                    chunks.append(current)
                    # Add overlap by including last part of previous chunk
                    current = current[-overlap:] + '\n\n' + order_separator + '\n\n' + section
                else:
                    if current:
                        current = current + '\n\n' + order_separator + '\n\n' + section
                    else:
                        current = section
            
            if current:
                chunks.append(current)
            return chunks if chunks else [text]
        
        # Default chunking for non-order documents
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks: List[str] = []
        current = ''
        for p in paragraphs:
            if len(current) + len(p) + 2 <= max_chars:
                current = f"{current}\n\n{p}".strip()
            else:
                if current:
                    chunks.append(current)
                # if single paragraph too long, hard wrap
                while len(p) > max_chars:
                    chunks.append(p[:max_chars])
                    p = p[max_chars-overlap:]
                current = p
        if current:
            chunks.append(current)
        return chunks

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # If we've switched to TF-IDF backend or it's the default, do not attempt Groq embeddings
        if getattr(self, '_embedding_backend', 'groq') != 'groq':
            return []
        
        # Check if using TF-IDF fallback from config
        if self._embedding_model == "tfidf-fallback":
            self._embedding_backend = 'tfidf'
            if not self._embedding_switch_logged:
                logger.info(f"Using TF-IDF backend for RAG (Groq doesn't support embedding models).")
                self._embedding_switch_logged = True
            return []
        
        # Try Groq embedding models (if user specified a custom one)
        candidate_models = [self._embedding_model]
        last_error: Optional[Exception] = None
        for candidate in candidate_models:
            try:
                resp = self.groq_client.embeddings.create(model=candidate, input=texts)
                vectors = [d.embedding for d in resp.data]
                return vectors
            except Exception as e:
                msg = str(e).lower()
                last_error = e
                if any(tok in msg for tok in ["model_not_found", "does not exist", "unknown model", "404"]):
                    logger.warning(f"Embedding model '{candidate}' not available on Groq.")
                    continue
                # Not a model-not-found error; log it
                logger.warning(f"Embedding request failed: {e}")
                break
        # Switch to TF-IDF backend as a fallback (only log once)
        self._embedding_backend = 'tfidf'
        if not self._embedding_switch_logged:
            logger.info(f"Groq doesn't support embedding models. Using TF-IDF backend for RAG.")
            self._embedding_switch_logged = True
        return []

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def index_documents_directory(self, directory_path: str) -> int:
        """Index text-like files from a directory into the vector store."""
        try:
            base = Path(directory_path)
            if not base.exists() or not base.is_dir():
                logger.warning(f"Docs directory not found: {directory_path}")
                return 0
            # Scope RAG to this directory for this session and reset prior vectors
            try:
                self._current_docs_base = str(base.resolve())
            except Exception:
                self._current_docs_base = str(base)
            # Clear existing vectors so we don't mix previous corpora
            self._vectors = []
            self._save_vector_store()
            files = [p for p in base.rglob('*') if p.suffix.lower() in {'.txt', '.md', '.markdown', '.pdf', '.docx'}]
            total_chunks = 0
            new_records: List[Dict[str, Any]] = []
            for fp in files:
                try:
                    content = self._read_file_text(fp)
                except Exception:
                    continue
                chunks = self._chunk_text(content)
                embeddings = self._embed_texts(chunks)
                if embeddings:
                    for chunk_text, emb in zip(chunks, embeddings):
                        new_records.append({
                            'embedding': emb,
                            'text': chunk_text,
                            'metadata': {
                                'source': str(fp),
                                'created': datetime.now().isoformat()
                            }
                        })
                    total_chunks += len(embeddings)
                else:
                    # TF-IDF backend: store text only; vectors computed at query time
                    for chunk_text in chunks:
                        new_records.append({
                            'embedding': [],
                            'text': chunk_text,
                            'metadata': {
                                'source': str(fp),
                                'created': datetime.now().isoformat()
                            }
                        })
                    total_chunks += len(chunks)
            if new_records:
                self._vectors.extend(new_records)
                self._save_vector_store()
            logger.info(f"Indexed {total_chunks} chunks from {len(files)} files")
            return total_chunks
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return 0

    def _read_file_text(self, fp: Path) -> str:
        suffix = fp.suffix.lower()
        if suffix in {'.txt', '.md', '.markdown'}:
            return fp.read_text(encoding='utf-8', errors='ignore')
        if suffix == '.pdf':
            if PdfReader is None:
                logger.warning("PDF support not installed. Run: pip install pypdf. Skipping %s", fp)
                return ""
            try:
                reader = PdfReader(str(fp))
                texts: List[str] = []
                for page in reader.pages:
                    try:
                        page_text = page.extract_text() or ""
                        texts.append(page_text)
                    except Exception:
                        continue
                return "\n".join(texts)
            except Exception as e:
                logger.warning(f"Failed to read PDF {fp}: {e}")
                return ""
        if suffix == '.docx':
            if docx is None:
                logger.warning("DOCX support not installed. Run: pip install python-docx. Skipping %s", fp)
                return ""
            try:
                doc = docx.Document(str(fp))
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception as e:
                logger.warning(f"Failed to read DOCX {fp}: {e}")
                return ""
        return ""

    def _rag_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []
        
        # Extract order numbers from query for exact matching boost
        query_order_numbers = self._extract_order_numbers(query)
        
        # Restrict to current docs base if set
        records: List[Dict[str, Any]] = self._vectors
        try:
            if self._current_docs_base:
                base_norm = self._current_docs_base.lower()
                filtered = []
                for rec in self._vectors:
                    src = (rec.get('metadata') or {}).get('source', '')
                    if isinstance(src, str) and src:
                        try:
                            src_norm = str(Path(src).resolve()).lower()
                        except Exception:
                            src_norm = src.lower()
                        if src_norm.startswith(base_norm):
                            filtered.append(rec)
                if filtered:
                    records = filtered
        except Exception:
            pass
        if not records:
            return []
        
        # If using TF-IDF backend, compute similarities over text corpus directly
        if getattr(self, '_embedding_backend', 'groq') == 'tfidf':
            try:
                corpus = [rec.get('text', '') for rec in records]
                if not any(corpus):
                    return []
                # Fit/refresh vectorizer each call to include latest corpus (small corpora expected)
                self._tfidf_vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
                doc_matrix = self._tfidf_vectorizer.fit_transform(corpus)
                q_vec = self._tfidf_vectorizer.transform([query])
                # Compute cosine similarity
                doc_norms = (doc_matrix.power(2)).sum(axis=1)
                doc_norms = doc_norms.A.ravel() ** 0.5
                q_norm = (q_vec.power(2)).sum() ** 0.5
                sims: List[Tuple[Dict[str, Any], float]] = []
                if q_norm == 0:
                    return []
                # Efficient dot product
                dots = (doc_matrix @ q_vec.T).toarray().ravel()
                for rec, dot, dnorm in zip(records, dots, doc_norms):
                    denom = (dnorm * q_norm)
                    base_score = (dot / denom) if denom else 0.0
                    
                    # Boost score significantly if chunk contains exact order number match
                    boost = 0.0
                    if query_order_numbers:
                        rec_text = rec.get('text', '').upper()
                        for order_num in query_order_numbers:
                            if order_num.upper() in rec_text:
                                boost = 0.5  # Large boost for exact order number match
                                break
                    
                    sims.append((rec, base_score + boost))
                    
                sims.sort(key=lambda x: x[1], reverse=True)
                # Lower threshold to capture short facts; ensure at least one hit if any positive sim
                threshold = 0.02  # Very low threshold to capture all relevant content
                top_hits = [rec for rec, score in sims[:k] if score > threshold]
                if top_hits:
                    return top_hits
                if sims and sims[0][1] > 0:
                    return [sims[0][0]]
                return []
            except Exception as e:
                logger.warning(f"TF-IDF RAG search failed: {e}")
                return []
        
        # Default: use stored embeddings
        q_embeds = self._embed_texts([query])
        if not q_embeds:
            return []
        q = q_embeds[0]
        scored = []
        for rec in records:
            base_score = self._cosine_sim(q, rec.get('embedding') or [])
            
            # Boost score if chunk contains exact order number match
            boost = 0.0
            if query_order_numbers:
                rec_text = rec.get('text', '').upper()
                for order_num in query_order_numbers:
                    if order_num.upper() in rec_text:
                        boost = 0.3  # Boost for exact order number match
                        break
            
            scored.append((rec, base_score + boost))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        threshold = 0.2
        top_hits = [rec for rec, score in scored[:k] if score > threshold]
        if top_hits:
            return top_hits
        if scored and scored[0][1] > 0:
            return [scored[0][0]]
        return []
    
    # ---------------------- Embeddings ----------------------
    def _validate_api_response(self, response: Any, expected_type: str) -> Dict:
        """Validate and parse API response"""
        try:
            # Convert response to dictionary
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response for {expected_type}")
                    raise ValueError(f"Invalid JSON response for {expected_type}")
            elif isinstance(response, dict):
                response_dict = response
            else:
                try:
                    # Try to get dictionary representation
                    response_dict = getattr(response, 'model_dump', lambda: response)()
                except Exception:
                    logger.error(f"Unexpected {expected_type} response type: {type(response)}")
                    raise ValueError(f"Unexpected {expected_type} response type: {type(response)}")

            # Check for error in response
            if isinstance(response_dict, dict) and 'error' in response_dict:
                error = response_dict['error']
                error_msg = error.get('message', str(error))
                if isinstance(error, dict) and error.get('code') == 401:
                    raise Exception(f"Authentication failed: {error_msg}")
                raise Exception(f"API Error: {error_msg}")

            return response_dict

        except Exception as e:
            logger.error(f"Error validating {expected_type} response: {str(e)}")
            raise

    def _get_relevant_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """Find relevant documents using Groq model to score relevance"""
        try:
            # Create a prompt to compare query with each document
            results = []
            for doc in documents:
                prompt = f"""On a scale of 0 to 100, how relevant is the following content to the query: "{query}"?
                Content: {doc['content']}
                Just return a number between 0 and 100, nothing else."""
                
                groq_response = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Return only a number between 0 and 100."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.groq_config.model,
                    temperature=0.0,
                    max_tokens=10
                )
                try:
                    relevance_score = float(groq_response.choices[0].message.content.strip())
                    if relevance_score > 50:  # Only keep relevant documents
                        results.append((doc, relevance_score))
                except ValueError:
                    continue
                    
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in results[:3]]  # Return top 3 most relevant documents
            
        except Exception as e:
            logger.error(f"Error finding relevant documents: {e}")
            return []
    
    # ---------------------- Entity Validation ----------------------
    def _extract_order_numbers(self, query: str) -> List[str]:
        """Extract potential order numbers from query"""
        # Match patterns like TM-2024-XXXXXX, TMG-XXXX, ORD-XXX, etc.
        # Try longest patterns first to avoid partial matches
        query_upper = query.upper()
        order_numbers = []
        
        # Pattern 1: TM-2024-100457 (longer pattern - try first)
        long_pattern = r'\b[A-Z]{2,4}-\d{4,}-\d{4,}\b'
        long_matches = re.findall(long_pattern, query_upper)
        order_numbers.extend(long_matches)
        
        # Only try shorter pattern if no long matches found
        if not long_matches:
            short_pattern = r'\b[A-Z]{2,4}-\d{4,}\b'
            short_matches = re.findall(short_pattern, query_upper)
            order_numbers.extend(short_matches)
        
        return list(set(order_numbers))  # Remove duplicates
    
    def _extract_email(self, query: str) -> Optional[str]:
        """Extract email address from query"""
        # Simple email pattern
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, query)
        return matches[0] if matches else None
    
    def _requires_order_specification(self, query: str, order_id: str = None) -> bool:
        """Check if query asks about order-specific info without providing order number"""
        # If order_id is already provided in context, we're good
        if order_id:
            return False
        
        # If query contains an explicit order number, we're good
        if self._extract_order_numbers(query):
            return False
        
        # Keywords that indicate order-specific information is being requested
        order_specific_keywords = [
            r'\b(the|my)\s+order\b',
            r'\border\s+(total|status|tracking|number|date|details)\b',
            r'\b(total|payment|shipping)\s+(for|of|from)\s+(the|my)?\s*order\b',
            r'\bwhen\s+(will|did)\s+(my|the)?\s*order\b',
            r'\bwhere\s+is\s+(my|the)?\s*order\b',
            r'\btrack\s+(my|the)?\s*order\b',
            r'\bpayment\s+method\s+(for|used)\b',
            r'\bshipping\s+address\s+for\b',
        ]
        
        query_lower = query.lower()
        for pattern in order_specific_keywords:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def _validate_order_exists(self, order_number: str, context_text: str = None) -> bool:
        """Check if order number exists in the database"""
        if not order_number:
            return False
        # Check database directly for the order
        order = self.db_manager.get_order_by_id(order_number)
        return order is not None
    
    # ---------------------- Retrieve Context ----------------------
    def _retrieve_relevant_context(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Retrieve relevant context using RAG components with LLM fallback"""
        context_details = {
            'rag_documents': [],
            'rag_chunks_text': [],
            'used_llm_fallback': False
        }
        
        # If RAG components aren't available, use LLM directly
        if not self.document_retriever or not self.rag_chain:
            logger.warning("RAG components not initialized, using LLM directly")
            context = self._generate_fallback_response(query)
            context_details['used_llm_fallback'] = True
            return context, context_details
            
        try:
            # Try to get relevant documents
            retrieval_results = self.document_retriever.retrieve(query, top_k=5)
            
            # If no relevant documents found, use LLM directly
            if not retrieval_results:
                logger.info("No relevant documents found, using LLM directly")
                context = self._generate_fallback_response(query)
                context_details['used_llm_fallback'] = True
                return context, context_details
            
            # If we have documents, format the context using RAG
            context = self.rag_chain.format_context(retrieval_results)
            
            # Prepare context details for response
            for result in retrieval_results:
                if not isinstance(result, RetrievalResult):
                    continue
                    
                source = getattr(result.chunk, 'source', 'unknown')
                text = getattr(result.chunk, 'text', '').strip()
                
                if not text:
                    continue
                    
                context_details['rag_documents'].append(os.path.basename(source))
                context_details['rag_chunks_text'].append({
                    'source': os.path.basename(source),
                    'text_preview': text[:300] + '...' if len(text) > 300 else text,
                    'score': float(getattr(result, 'score', 0.0))
                })
            
            return context, context_details
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")
            # Fallback to LLM if there's an error
            context = self._generate_fallback_response(query)
            context_details['used_llm_fallback'] = True
            return context, context_details
            
    def _generate_fallback_response(self, query: str) -> str:
        """Generate a response using just the LLM when RAG isn't available"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate and helpful information. If you don't know something, say so."},
                    {"role": "user", "content": query}
                ],
                model=self.groq_config.model,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in fallback LLM response: {str(e)}")
            return "I'm having trouble generating a response at the moment. Please try again later."
    
    def _get_customer_context(self, customer_email: str = None, order_id: str = None) -> Tuple[str, Dict[str, Any]]:
        context_parts = []
        context_details = {
            'customer_data': None,
            'order_data': None
        }
        
        # If we have an order_id but no customer_email, get the customer from the order
        if order_id and not customer_email:
            order = self.db_manager.get_order_by_id(order_id)
            if order:
                # Get customer by customer_id from the order
                conn = sqlite3.connect(self.db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT email FROM customers WHERE customer_id = ?", (order.customer_id,))
                result = cursor.fetchone()
                conn.close()
                if result:
                    customer_email = result[0]
        
        if customer_email:
            customer = self.db_manager.get_customer_by_email(customer_email)
            if customer:
                context_parts.append(f"Customer: {customer.name} ({customer.email})")
                context_parts.append(f"Customer ID: {customer.customer_id}")
                
                # Add total order count
                total_orders = len(customer.order_history)
                context_parts.append(f"TOTAL ORDERS PLACED: {total_orders}")
                
                context_details['customer_data'] = {
                    'name': customer.name,
                    'email': customer.email,
                    'customer_id': customer.customer_id,
                    'order_count': total_orders
                }
                
                if customer.order_history:
                    # List ALL order IDs for easy reference
                    all_order_ids = [o['order_id'] for o in customer.order_history]
                    context_parts.append(f"ALL ORDER IDs: {', '.join(all_order_ids)}")
                    
                    # Show COMPLETE details for ALL orders with products
                    context_parts.append(f"\n=== COMPLETE ORDER HISTORY ({total_orders} orders) ===")
                    
                    all_orders_details = []
                    total_items_count = 0
                    
                    for idx, order_hist in enumerate(customer.order_history, 1):
                        order_detail = self.db_manager.get_order_by_id(order_hist['order_id'])
                        if order_detail:
                            order_summary = f"\nOrder #{idx}: {order_detail.order_id}"
                            order_summary += f"\n  Status: {order_detail.status}"
                            order_summary += f"\n  Date: {order_detail.order_date}"
                            if order_detail.estimated_delivery:
                                order_summary += f"\n  Delivery: {order_detail.estimated_delivery}"
                            if order_detail.tracking_number:
                                order_summary += f"\n  Tracking: {order_detail.tracking_number}"
                            
                            # Get all products in this order
                            order_summary += f"\n  Items ({len(order_detail.products)}):"
                            for prod_item in order_detail.products:
                                product = self.db_manager.get_product_by_id(prod_item['product_id'])
                                if product:
                                    qty = prod_item.get('quantity', 1)
                                    price = prod_item.get('price', product.price)
                                    order_summary += f"\n    - {product.name} (Qty: {qty}, Price: ${price})"
                                    total_items_count += qty
                            
                            all_orders_details.append(order_summary)
                    
                    context_parts.extend(all_orders_details)
                    context_parts.append(f"\nTOTAL ITEMS ACROSS ALL ORDERS: {total_items_count}")
                    
                    # Add order IDs list and complete details to context
                    context_details['customer_data']['all_order_ids'] = all_order_ids
                    context_details['customer_data']['total_items'] = total_items_count
                    context_details['customer_data']['complete_orders'] = all_orders_details
        
        if order_id:
            order = self.db_manager.get_order_by_id(order_id)
            if order:
                order_info = (
                    f"{t('order_details', self.language)}\n"
                    f"{t('order_id', self.language)}: {order.order_id}\n"
                    f"{t('status', self.language)}: {order.status}\n"
                    f"{t('order_date', self.language)}: {order.order_date}"
                )
                if order.estimated_delivery:
                    order_info += f"\n{t('estimated_delivery', self.language)}: {order.estimated_delivery}"
                context_parts.append(order_info)
                
                if order.tracking_number:
                    context_parts.append(f"{t('tracking_number', self.language)}: {order.tracking_number}")
                
                products_info = []
                for p_item in order.products:
                    product = self.db_manager.get_product_by_id(p_item['product_id'])
                    if product:
                        context_parts.append(f"{t('product', self.language)}: {product.name} - {product.inventory_status}")
                        products_info.append({
                            'name': product.name,
                            'status': product.inventory_status
                        })
                
                context_details['order_data'] = {
                    'order_id': order.order_id,
                    'status': order.status,
                    'tracking_number': order.tracking_number,
                    'products': products_info
                }
        
        return "\n".join(context_parts) if context_parts else "", context_details
    
    # ---------------------- Escalation ----------------------
    async def _generate_general_response(self, query: str, sources: List[str] = None) -> Dict[str, Any]:
        """Generate a response using the model's general knowledge when no specific context is available."""
        try:
            system_prompt = (
                "You are a helpful customer support agent for an e-commerce store. "
                "Provide a helpful and accurate response to the user's question. "
                "If you don't know the specific answer, provide the best information you can "
                "based on general e-commerce knowledge and best practices."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            groq_response = self.groq_client.chat.completions.create(
                messages=messages,
                model=self.groq_config.model,
                temperature=0.5,  # Slightly higher temperature for more creative responses
                max_tokens=512
            )
            
            response_text = groq_response.choices[0].message.content.strip()
            
            # Add a note that this is a general response
            if not sources:
                sources = ["General Knowledge"]
            else:
                sources.append("General Knowledge")
                
            return {
                "response": response_text,
                "escalate": False,
                "confidence": 0.7,  # Lower confidence for general knowledge responses
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating general response: {e}")
            return {
                "response": "I'm having trouble accessing the information right now. Please try again later.",
                "escalate": True,
                "confidence": 0.0,
                "sources": ["System: Error generating response"],
                "timestamp": datetime.now().isoformat()
            }
    
    def _should_escalate(self, query: str, context: str) -> bool:
        escalation_keywords = [
            "legal", "lawsuit", "attorney", "lawyer", "sue", "court",
            "fraud", "stolen", "hacked", "unauthorized", "identity theft",
            "emergency", "urgent", "complaint", "unsatisfied", "angry",
            "manager", "supervisor", "speak to human", "representative",
            "cancel account", "close account", "delete data", "gdpr"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in escalation_keywords)
    
    # ---------------------- Process Query ----------------------
    async def process_query(self, query: str, customer_email: str = None, order_id: str = None, conversation_id: str = None) -> Dict[str, Any]:
        try:
            if self._should_escalate(query, ""):
                return {
                    "response": t("escalate_text", self.language),
                    "escalate": True,
                    "confidence": 1.0,
                    "sources": ["System: Escalation required"]
                }
            
            try:
                # Check if query requires order specification but doesn't have one
                if self._requires_order_specification(query, order_id):
                    return {
                        "response": "I need your order number to help with that. Please provide the order number from your confirmation email or account dashboard.",
                        "escalate": False,
                        "confidence": 1.0,
                        "sources": ["System: Order number required"],
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Get context and track sources
                knowledge_context, knowledge_details = self._retrieve_relevant_context(query)
                customer_context, customer_details = self._get_customer_context(customer_email, order_id)
                
                # Track sources
                sources = []
                
                # Add knowledge base sources
                if knowledge_details.get('knowledge_base_docs'):
                    sources.extend([
                        f"Knowledge Base: {doc['category']}" 
                        for doc in knowledge_details['knowledge_base_docs']
                    ])
                
                # Add RAG document sources
                if knowledge_details.get('rag_documents'):
                    sources.extend([
                        f"Document: {os.path.basename(doc)}" 
                        for doc in knowledge_details['rag_documents']
                    ])
                
                # Add customer/order source if applicable
                if customer_email:
                    sources.append(f"Customer: {customer_email}")
                if order_id:
                    sources.append(f"Order: {order_id}")
                
                # Make sources unique
                sources = list(dict.fromkeys(sources))
                
                # Entity extraction: Extract email and order numbers from query
                # Query data takes priority over startup parameters
                
                # Try to extract email from query
                extracted_email = self._extract_email(query)
                if extracted_email:
                    # Override startup email with extracted one
                    customer_email = extracted_email
                    sources = [s for s in sources if not s.startswith("Customer:")]
                    sources.append(f"Customer: {customer_email}")
                
                # Try to extract order numbers from query
                order_numbers = self._extract_order_numbers(query)
                if order_numbers:
                    # Validate the first extracted order number against database
                    extracted_order_id = order_numbers[0]
                    if self._validate_order_exists(extracted_order_id):
                        # Found valid order! Override any startup order_id with this one
                        order_id = extracted_order_id
                        sources = [s for s in sources if not s.startswith("Order:")]
                        sources.append(f"Order: {order_id}")
                    else:
                        # Order not found in database
                        missing_list = ", ".join(order_numbers)
                        return {
                            "response": f"I cannot find order number(s): {missing_list}. Please verify the order number.",
                            "escalate": False,
                            "confidence": 1.0,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Re-fetch customer context with the correct email and order_id
                if extracted_email or order_numbers:
                    customer_context, customer_details = self._get_customer_context(customer_email, order_id)
                
                # Make sources unique
                sources = list(dict.fromkeys(sources))
                
                # Build the context for the LLM
                context_parts = []
                if knowledge_context:
                    context_parts.append(f"Relevant Information:\n{knowledge_context}")
                if customer_context:
                    context_parts.append(f"Customer Information:\n{customer_context}")
                
                # First try with context
                try:
                    if context_parts:  # If we have some context
                        system_prompt = (
                            "You are a helpful customer support agent. "
                            "CRITICAL INSTRUCTIONS:\n"
                            "1. Only use actual customer/order data from the 'Customer Information' section\n"
                            "2. NEVER reference specific order numbers, addresses, or personal data from documents\n"
                            "3. Use general policy information from knowledge base, but NOT example orders\n"
                            "4. If you don't have the specific customer's data, say 'I don't have that information'\n"
                            "5. Be concise and accurate in your response.\n\n"
                            f"{'\n\n'.join(context_parts)}"
                        )
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query}
                        ]
                        
                        groq_response = self.groq_client.chat.completions.create(
                            messages=messages,
                            model=self.groq_config.model,
                            temperature=0.3,  # Slightly higher temperature for more natural responses
                            max_tokens=512
                        )
                        
                        response_text = groq_response.choices[0].message.content.strip()
                        
                        # If the model indicates it doesn't have the information, fall back to general knowledge
                        if "i don't have that information" in response_text.lower() or not response_text:
                            return await self._generate_general_response(query, sources)
                            
                        # Add sources to the response if we have any
                        if sources:
                            response_text += "\n\nSources:\n" + "\n".join([f"- {src}" for src in sources])
                        
                        return {
                            "response": response_text,
                            "escalate": False,
                            "confidence": 1.0,
                            "sources": sources,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # If no context is available, use general knowledge
                        return await self._generate_general_response(query, sources)
                        
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    # Fall back to general knowledge if there's an error with the context-based response
                    return await self._generate_general_response(query, sources)
                    
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return {
                    "response": t("tech_difficulties", self.language),
                    "escalate": True,
                    "confidence": 0.0,
                    "sources": ["System: Processing error"],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "response": t("tech_difficulties", self.language),
                "escalate": True,
                "confidence": 0.0,
                "sources": ["System: Unexpected error"],
                "timestamp": datetime.now().isoformat()
            }

# ---------------------- Interactive Chat ----------------------
async def interactive_chat():
    """Interactive CLI chat with the customer support assistant"""
    # ... (rest of the code remains the same)
    # accept any language code; default to 'en'
    selected_lang = (input(t("prompt_language", "en")).strip() or "en").lower()

    # Initialize support system with language
    support_system = EcommerceSupport(language=selected_lang)

    print(t("welcome", selected_lang))
    print(t("instructions", selected_lang))

    # Optionally ask for customer email/order once
    customer_email = input(t("prompt_email", selected_lang)).strip() or None
    order_id = input(t("prompt_order_id", selected_lang)).strip() or None
    # Optional: docs directory for indexing
    docs_dir = input("Docs directory to index for RAG (optional): ").strip()
    if docs_dir:
        try:
            chunks = support_system.index_documents_directory(docs_dir)
            logger.info(f"RAG indexed chunks: {chunks}")
        except Exception as e:
            logger.warning(f"Skipping RAG indexing: {e}")
    conversation_id = "conv_live_001"

    while True:
        query = input(t("you", selected_lang))
        if query.lower() in ["quit", "exit"]:
            print(t("goodbye", selected_lang))
            break

        result = await support_system.process_query(
            query=query,
            customer_email=customer_email,
            order_id=order_id,
            conversation_id=conversation_id
        )

        # Print detailed context information
        print("\n" + "="*70)
        print("📊 CONTEXT INFORMATION")
        print("="*70)
        
        context_details = result.get('context_details', {})
        
        # Display Knowledge Base Context
        kb_details = context_details.get('knowledge_base', {})
        if kb_details.get('knowledge_base_docs') or kb_details.get('rag_documents'):
            print("\n📚 KNOWLEDGE BASE SOURCES:")
            if kb_details.get('knowledge_base_docs'):
                for idx, doc in enumerate(kb_details['knowledge_base_docs'], 1):
                    print(f"  {idx}. Category: {doc['category']}")
                    print(f"     Content: {doc['content_preview']}")
            
            if kb_details.get('rag_chunks_text'):
                print("\n📄 RAG DOCUMENT CHUNKS RETRIEVED:")
                for idx, chunk in enumerate(kb_details['rag_chunks_text'], 1):
                    print(f"  {idx}. Source: {chunk['source']}")
                    print(f"     Content: {chunk['text_preview']}")
        
        # Display Customer Database Context
        customer_details_info = context_details.get('customer_data', {})
        if customer_details_info.get('customer_data'):
            print("\n" + "="*70)
            print("👤 CUSTOMER DATABASE INFORMATION")
            print("="*70)
            cust_data = customer_details_info['customer_data']
            print(f"  Name: {cust_data['name']}")
            print(f"  Email: {cust_data['email']} ⭐ (PRIMARY KEY)")
            print(f"  Customer ID: {cust_data['customer_id']}")
            print(f"  Total Orders: {cust_data['order_count']}")
            
            # Display all order IDs if available
            if cust_data.get('all_order_ids'):
                print(f"  All Order IDs: {', '.join(cust_data['all_order_ids'])}")
            
            # Display total items count
            if cust_data.get('total_items'):
                print(f"  Total Items Purchased: {cust_data['total_items']}")
            
            # Display complete order details
            if cust_data.get('complete_orders'):
                print("\n" + "="*70)
                print("📦 COMPLETE ORDER HISTORY")
                print("="*70)
                for order_detail in cust_data['complete_orders']:
                    print(order_detail)
        
        if customer_details_info.get('order_data'):
            print("\n📦 ORDER DATABASE INFORMATION:")
            order_data = customer_details_info['order_data']
            print(f"  Order ID: {order_data['order_id']}")
            print(f"  Status: {order_data['status']}")
            if order_data.get('tracking_number'):
                print(f"  Tracking: {order_data['tracking_number']}")
            if order_data.get('products'):
                print(f"  Products ({len(order_data['products'])}):")
                for prod in order_data['products']:
                    print(f"    - {prod['name']} [{prod['status']}]")
        
        # Display sources summary
        if result['sources']:
            print(f"\n🔍 DATA SOURCES USED: {', '.join(result['sources'])}")
        
        print("\n" + "="*70)
        print("💬 ASSISTANT RESPONSE")
        print("="*70)
        print(f"{t('assistant_prefix', selected_lang)}{result['response']}")
        
        if result.get("escalate"):
            print("\n" + t("escalation", selected_lang))
        
        print(t("divider", selected_lang))


if __name__ == "__main__":
    asyncio.run(interactive_chat())

