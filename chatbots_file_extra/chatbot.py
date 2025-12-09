import streamlit as st
from openai import OpenAI
import re
import logging
import time

# ⚡ NEW: Import FastKGQuerier
from src.query.fast_querier import FastKGQuerier

# ========== CONFIG ==========
OPENROUTER_API_KEY = ""

# ⚡ UPDATED: Correct port and credentials
NEO4J_URI = "bolt://localhost:7692"  # ← NEW PORT
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"  # ← NEW PASSWORD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== FAST NEO4J CONNECTION ==========
@st.cache_resource
def init_fast_querier():
    """⚡ Initialize FastKGQuerier with caching for reuse across requests"""
    try:
        querier = FastKGQuerier(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USERNAME,
            neo4j_password=NEO4J_PASSWORD
        )
        # Test connection
        querier.semantic_query("test", top_k=1)
        logger.info("✅ FastKGQuerier initialized successfully")
        return querier
    except Exception as e:
        logger.error(f"❌ Failed to initialize FastKGQuerier: {e}")
        st.error(f"Failed to connect to Neo4j: {e}")
        return None

querier = init_fast_querier()

def query_knowledge_graph_fast(question: str, top_k: int = 15):
    """
    ⚡ SUPER FAST: Query using FastKGQuerier with fulltext indexes
    Expected time: <1 second (vs 20+ seconds with old method)
    """
    if not querier:
        return []
    
    start_time = time.time()
    
    try:
        # Use semantic query with fulltext index (lightning fast!)
        results = querier.semantic_query(
            query_text=question,
            top_k=top_k,
            include_neighbors=True,
            use_cache=True
        )
        
        # Format results for display
        formatted_facts = []
        
        for item in results:
            entity = item.get('entity', '')
            label = item.get('label', '')
            score = item.get('score', 0)
            neighbors = item.get('neighbors', [])
            
            # Add main entity
            formatted_facts.append(
                f"[Entity] {entity} ({label}) - relevance: {score:.2f}"
            )
            
            # Add relationships
            for neighbor in neighbors[:3]:  # Top 3 neighbors per entity
                rel_type = neighbor.get('type', 'RELATED')
                neighbor_text = neighbor.get('text', '')
                confidence = neighbor.get('confidence', 0)
                
                formatted_facts.append(
                    f"  → {entity} --[{rel_type}]--> {neighbor_text} (confidence: {confidence:.2f})"
                )
        
        elapsed = time.time() - start_time
        logger.info(f"⚡ Query completed in {elapsed:.2f}s - Found {len(results)} entities")
        
        return formatted_facts
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return []

def get_entity_statistics():
    """Get basic statistics about the knowledge graph"""
    if not querier:
        return {}
    
    try:
        with querier.driver.session() as session:
            # Get basic counts
            result = session.run("""
                MATCH (n:Entity)
                OPTIONAL MATCH (n)-[r]-()
                RETURN count(DISTINCT n) as total_entities,
                       count(r) as total_relationships,
                       count(DISTINCT n.label) as entity_types
            """)
            stats = result.single()
            
            # Get top entity types
            result = session.run("""
                MATCH (n:Entity)
                RETURN n.label as type, count(*) as count
                ORDER BY count DESC
                LIMIT 5
            """)
            top_types = [(record["type"], record["count"]) for record in result]
            
            return {
                "total_entities": stats["total_entities"],
                "total_relationships": stats["total_relationships"],
                "entity_types": stats["entity_types"],
                "top_entity_types": top_types
            }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {}

# ========== OPENROUTER CLIENT ==========
@st.cache_resource
def init_openai_client():
    """Initialize OpenAI client with caching"""
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

client = init_openai_client()

def extract_keywords_simple(question: str):
    """Simple keyword extraction with domain-specific terms"""
    stopwords = {
        "what", "is", "a", "an", "the", "who", "where", "when", "why", "how",
        "in", "of", "and", "to", "for", "with", "by", "from", "up", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "do", "does", "are", "was", "were", "been", "have", "has", "had",
        "will", "would", "could", "should", "may", "might", "can", "?", "!",
        "tell", "me", "show", "find", "get", "give"
    }
    
    # Extract words and clean
    words = re.findall(r'\w+', question.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Join with space for broad matching
    return " ".join(keywords)

def extract_keywords_llm(question: str):
    """LLM-assisted keyword extraction with domain focus"""
    prompt = f"""
    Extract the most important search keywords from this question for querying a knowledge graph about business, suppliers, risk management, and organizational data.
    
    Focus on:
    - Key business concepts and entities
    - Names of organizations, people, or systems
    - Process names and technical terms
    - Risk-related terminology
    
    Question: {question}
    
    Return only the essential keywords separated by spaces (no explanations):
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM keyword extraction failed: {e}")
        return extract_keywords_simple(question)  # Fallback

def generate_answer(question: str, context: list):
    """Generate answer using Llama3 model with enhanced prompting"""
    if not context:
        context_text = "No relevant facts found in the knowledge graph."
    else:
        context_text = "\n".join(context[:25])  # Increased from 20 to 25
    
    prompt = f"""
You are an AI assistant with access to a comprehensive knowledge graph containing information about business processes, supplier relationships, risk management, organizations, and related business concepts.

User Question: {question}

Knowledge Graph Facts:
{context_text}

Instructions:
1. If the graph facts directly answer the question, provide a clear, comprehensive response based on those facts.
2. If the facts are partially relevant, use them to provide the best possible answer and explain what information is available.
3. If the facts contain relationships or connections, explain how the entities are related.
4. If no relevant facts are found, suggest alternative ways to phrase the question or related topics that might be in the knowledge graph.
5. Always cite specific facts when possible (e.g., "According to the knowledge graph, X is connected to Y through Z relationship").

Keep your response informative but concise.
"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"I encountered an error while generating the response: {e}"

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="⚡ Fast KG RAG Chatbot", 
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Fast Knowledge Graph RAG Chatbot")
st.caption("⚡ Powered by indexed Neo4j queries - Sub-second response time!")

# Sidebar with statistics
if querier:
    with st.sidebar:
        st.header("📊 Graph Statistics")
        with st.spinner("Loading statistics..."):
            stats = get_entity_statistics()
        
        if stats:
            st.metric("Total Entities", f"{stats.get('total_entities', 0):,}")
            st.metric("Total Relationships", f"{stats.get('total_relationships', 0):,}")
            st.metric("Entity Types", stats.get('entity_types', 0))
            
            if stats.get('top_entity_types'):
                st.subheader("Top Entity Types")
                for entity_type, count in stats['top_entity_types']:
                    st.text(f"• {entity_type}: {count:,}")
        
        # Cache statistics
        cache_stats = querier.get_statistics()
        st.subheader("⚡ Query Performance")
        st.text(f"Cache hit rate: {cache_stats.get('hit_rate', '0%')}")
        st.text(f"Cached queries: {cache_stats.get('cache_size', 0)}")
        
        if st.button("Clear Query Cache"):
            querier.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
        
        st.subheader("💡 Try asking about:")
        st.text("• Supplier risk management")
        st.text("• Risk assessment processes")
        st.text("• Organizational relationships")
        st.text("• Business processes")
        st.text("• Compliance procedures")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.chat_input("Ask me about the knowledge graph...")

if user_input and querier:
    # Save user input
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ⚡ FAST QUERY
    query_start = time.time()
    
    with st.spinner("⚡ Searching knowledge graph..."):
        # Direct query - no need for complex keyword extraction anymore!
        context = query_knowledge_graph_fast(user_input, top_k=15)
        
        # If very few results, try keyword extraction
        if len(context) < 3:
            st.info("Limited results. Trying keyword extraction...")
            keywords = extract_keywords_simple(user_input)
            context = query_knowledge_graph_fast(keywords, top_k=15)
    
    query_elapsed = time.time() - query_start
    st.success(f"⚡ Search completed in {query_elapsed:.2f} seconds!")
    
    with st.spinner("Generating response..."):
        answer_start = time.time()
        answer = generate_answer(user_input, context)
        answer_elapsed = time.time() - answer_start
    
    # Save bot answer
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    # Performance metrics
    total_time = query_elapsed + answer_elapsed
    st.info(f"⏱️ Total time: {total_time:.2f}s (Query: {query_elapsed:.2f}s, Answer: {answer_elapsed:.2f}s)")
    
    # Show found context in expander
    if context:
        with st.expander(f"📋 Found {len(context)} relevant facts from knowledge graph"):
            for fact in context[:15]:  # Show first 15 facts
                st.text(fact)
    else:
        st.warning("No relevant facts found. Try rephrasing your question.")

elif user_input and not querier:
    st.error("Cannot process questions - Neo4j connection failed. Please check your configuration.")

# Display the chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Connection status
if not querier:
    st.error("⚠️ Neo4j connection failed. Please check your Neo4j configuration and ensure the database is running.")
    st.text("Troubleshooting:")
    st.text("1. Make sure your Neo4j Docker container is running")
    st.text(f"2. Verify connection: {NEO4J_URI}")
    st.text("3. Check credentials: neo4j / 12345678")
    st.text("4. Ensure port 7690 is accessible")
else:
    st.success("✅ Connected to Neo4j knowledge graph with FastKGQuerier")
