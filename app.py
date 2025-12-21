import os, json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----- LangChain RAG -----
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --------- Environment checks ----------
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. In Hugging Face Spaces, add it under Settings → Repository secrets.")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
API_TOKEN = os.getenv("API_TOKEN", "")

# def print_env_vars() -> None:
#     groq_api_key = os.getenv("GROQ_API_KEY", "")
#     print("Environment variables:")
#     print(f"API_TOKEN={API_TOKEN if API_TOKEN else '<empty>'}")
#     print(f"GROQ_API_KEY={groq_api_key if groq_api_key else '<empty>'}")

# print_env_vars()

app = FastAPI(title="CV Ask API (HF Spaces)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def guard(token: str | None):
    if API_TOKEN and token != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- RAG setup (will build from cv.json if present) ----------
# Improved chunking strategy with semantic awareness
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks for better precision
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Better semantic splitting
)

# Better embedding model for technical content
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Better for technical content
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)  # Lower temp for consistency

def load_cv_docs() -> List[Document]:
    """
    Loads CV documents from a JSON file formatted as an array of objects:
    [
        {"page_content": "...", "metadata": {...}},
        {"page_content": "...", "metadata": {...}},
        ...
    ]
    Returns a list of LangChain Document objects compatible with all-mpnet-base-v2 embeddings.
    """
    try:
        with open("cv.json", "r", encoding="utf-8") as f:
            cv_data = json.load(f)
    except FileNotFoundError:
        print("⚠️ cv.json not found — returning empty list.")
        return []
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON decoding error: {e}")
        return []

    docs = []
    for i, item in enumerate(cv_data):
        if not isinstance(item, dict):
            print(f"⚠️ Skipping non-dict item at index {i}: {item}")
            continue

        text = item.get("page_content", "")
        meta = item.get("metadata", {})

        if text and isinstance(meta, dict):
            docs.append(Document(page_content=text.strip(), metadata=meta))
        else:
            print(f"⚠️ Skipping invalid entry at index {i}: {item}")

    print(f"✅ Loaded {len(docs)} documents from cv.json.")
    return docs

def build_retriever():
    docs = load_cv_docs()
    if not docs:
        return None
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    
    # Enhanced retriever with better search parameters
    return vs.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={
            "k": 10,  # Retrieve more candidates
            "lambda_mult": 0.7,  # Balance relevance vs diversity
            "fetch_k": 20  # Fetch more for MMR selection
        }
    )

retriever = build_retriever()

SYSTEM = """You are an expert career assistant and technical interviewer with deep knowledge of software engineering, cloud architecture, and DevOps practices.

CORE CAPABILITIES:
- Answer questions about professional experience, skills, and achievements
- Generate role-targeted summaries and cover letters
- Create STAR format behavioral interview examples
- Provide technical deep-dives on projects and implementations
- Offer career advice and interview preparation guidance

RESPONSE MODES (auto-detect based on query):
1. GENERAL Q&A: Direct answers about experience, skills, projects
2. ROLE TARGETING: When query mentions specific roles/companies, provide targeted summaries
3. COVER LETTER: When asked for cover letters, create compelling narratives
4. STAR FORMAT: When asked for examples/stories, use STAR methodology
5. TECHNICAL DEEP-DIVE: For technical questions, provide implementation details
6. INTERVIEW PREP: For interview questions, provide comprehensive preparation

INSTRUCTIONS:
- Answer ONLY using facts from the provided context
- Auto-detect query intent and respond in the most appropriate format
- Be precise with technical details and quantify achievements
- If information is missing, state "I don't have that information in the context"
- Maintain professional tone while being conversational
- Prioritize recent and relevant experience"""

def format_docs(_docs: List[Document]) -> str:
    if not _docs:
        return "No relevant information found."
    
    lines = []
    seen_content = set()  # Avoid duplicate content
    
    # Group documents by section for better organization
    sections = {}
    for d in _docs:
        section = d.metadata.get('section', 'other')
        if section not in sections:
            sections[section] = []
        sections[section].append(d)
    
    # Priority order for sections
    section_order = ['summary', 'experience', 'skills', 'project', 'education', 'certification', 'mentorship', 'languages', 'basics']
    
    for section in section_order:
        if section in sections:
            for d in sections[section]:
                content = d.page_content.strip()
                if content not in seen_content:
                    seen_content.add(content)
                    m = d.metadata
                    
                    # Enhanced formatting with better context
                    tag_parts = [section.upper()]
                    if m.get("company"): tag_parts.append(m['company'])
                    if m.get("name"): tag_parts.append(m['name'])
                    if m.get("dates"): tag_parts.append(m['dates'])
                    if m.get("type"): tag_parts.append(f"({m['type']})")
                    
                    tag = f"[{' | '.join(tag_parts)}]"
                    lines.append(f"• {tag} {content}")
    
    # Add any remaining sections not in priority order
    for section, docs in sections.items():
        if section not in section_order:
            for d in docs:
                content = d.page_content.strip()
                if content not in seen_content:
                    seen_content.add(content)
                    lines.append(f"• [{section.upper()}] {content}")
    
    return "\n".join(lines[:15])  # Limit to prevent context overflow

# Intent-specific prompts for enhanced responses
PROMPTS = {
    'general_qa': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide a direct, informative answer using only the context provided."),
        ("user", "Question: {question}")
    ]),
    
    'role_targeting': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nCreate a targeted response highlighting the most relevant qualifications for the specific role mentioned. Focus on alignment between experience and role requirements."),
        ("user", "Query: {question}\n\nProvide a role-targeted summary (120-150 words) with bullet points emphasizing relevant experience, skills, and quantified achievements.")
    ]),
    
    'cover_letter': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nWrite a compelling cover letter using specific examples and quantifiable achievements from the context."),
        ("user", "Request: {question}\n\nCreate a professional cover letter (~180 words) that:\n1. Opens with strong alignment\n2. Highlights 2-3 relevant achievements with metrics\n3. Shows enthusiasm and understanding\n4. Closes with next steps")
    ]),
    
    'star_examples': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide STAR format examples (Situation, Task, Action, Result) with specific details and quantified outcomes."),
        ("user", "Request: {question}\n\nGenerate compelling STAR examples that demonstrate:\n- Technical leadership and problem-solving\n- Specific actions taken\n- Quantified business impact\n\nFormat: **Situation:** [context] **Task:** [challenge] **Action:** [steps] **Result:** [outcome]")
    ]),
    
    'technical_deepdive': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide detailed technical explanations focusing on implementation details, architecture decisions, and technologies used."),
        ("user", "Technical Question: {question}\n\nProvide a comprehensive technical explanation including:\n- Specific technologies and tools used\n- Architecture and implementation approach\n- Challenges faced and solutions\n- Technical outcomes and metrics")
    ]),
    
    'interview_prep': ChatPromptTemplate.from_messages([
        ("system", SYSTEM + "\n\nRELEVANT CONTEXT:\n{context}\n\nProvide comprehensive interview preparation focusing on relevant experience, specific examples, and potential follow-up questions."),
        ("user", "Interview Prep: {question}\n\nProvide interview-ready responses including:\n- Key talking points with specific examples\n- Quantified achievements and impact\n- Technical details where relevant\n- Potential follow-up questions to prepare for")
    ])
}
# Enhanced query processing with intent detection
def detect_query_intent(query: str) -> str:
    """Detect the intent of the user query"""
    query_lower = query.lower()
    
    # Role targeting patterns
    if any(phrase in query_lower for phrase in ['for the role', 'position at', 'applying to', 'job at', 'role of']):
        return 'role_targeting'
    
    # Cover letter patterns
    if any(phrase in query_lower for phrase in ['cover letter', 'application letter', 'letter for']):
        return 'cover_letter'
    
    # STAR/example patterns
    if any(phrase in query_lower for phrase in ['give me an example', 'tell me about a time', 'describe a situation', 'star format', 'behavioral']):
        return 'star_examples'
    
    # Technical deep-dive patterns
    if any(phrase in query_lower for phrase in ['how did you implement', 'technical details', 'architecture', 'deep dive', 'explain the technical']):
        return 'technical_deepdive'
    
    # Interview prep patterns
    if any(phrase in query_lower for phrase in ['interview', 'prepare for', 'questions about', 'what would you say']):
        return 'interview_prep'
    
    return 'general_qa'

def enhance_query(query: str, intent: str = None) -> str:
    """Enhance user queries with context for better retrieval"""
    if not intent:
        intent = detect_query_intent(query)
    
    query_lower = query.strip().lower()
    
    # Intent-based query enhancement
    if intent == 'role_targeting':
        return f"relevant experience skills achievements for role: {query}"
    elif intent == 'cover_letter':
        return f"achievements leadership impact quantified results: {query}"
    elif intent == 'star_examples':
        return f"specific examples achievements leadership problem solving: {query}"
    elif intent == 'technical_deepdive':
        return f"technical implementation architecture details: {query}"
    elif intent == 'interview_prep':
        return f"experience skills achievements examples: {query}"
    
    # General query enhancement patterns
    if any(word in query_lower for word in ['experience', 'worked', 'job', 'role']):
        return f"professional experience work history: {query}"
    elif any(word in query_lower for word in ['skill', 'technology', 'tech', 'programming']):
        return f"technical skills technologies: {query}"
    elif any(word in query_lower for word in ['project', 'built', 'created', 'developed']):
        return f"projects development work: {query}"
    elif any(word in query_lower for word in ['education', 'degree', 'university', 'college']):
        return f"education academic background: {query}"
    elif any(word in query_lower for word in ['aws', 'cloud', 'kubernetes', 'docker']):
        return f"cloud infrastructure devops: {query}"
    
    return query

def get_enhanced_context(query: str) -> tuple[str, str]:
    """Get enhanced context based on query intent with multiple retrieval strategies"""
    if not retriever:
        return "No context available", "general_qa"
    
    intent = detect_query_intent(query)
    
    try:
        # Primary retrieval with enhanced query
        enhanced_q = enhance_query(query, intent)
        primary_docs = retriever.invoke(enhanced_q)
        
        # Secondary retrieval for comprehensive context
        secondary_docs = []
        if intent in ['role_targeting', 'cover_letter', 'interview_prep']:
            # Get additional achievement-focused content
            achievement_query = "achievements leadership impact results metrics"
            secondary_docs = retriever.invoke(achievement_query)
        elif intent == 'technical_deepdive':
            # Get additional technical content
            tech_query = "technical implementation architecture technologies stack"
            secondary_docs = retriever.invoke(tech_query)
        elif intent == 'star_examples':
            # Get additional example-rich content
            example_query = "led architected implemented improved reduced increased"
            secondary_docs = retriever.invoke(example_query)
        
        # Combine and deduplicate
        all_docs = primary_docs + secondary_docs
        seen_content = set()
        unique_docs = []
        
        for doc in all_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)
        
        return format_docs(unique_docs[:15]), intent
        
    except Exception as e:
        print(f"Retriever error: {e}")
        return "Error retrieving context", "general_qa"

def create_enhanced_response(query: str) -> dict:
    """Create enhanced response based on query intent"""
    context, intent = get_enhanced_context(query)
    
    # Select appropriate prompt based on intent
    prompt = PROMPTS.get(intent, PROMPTS['general_qa'])
    
    # Create chain with selected prompt
    chain = (
        {"context": lambda x: context, "question": lambda x: query}
        | prompt | llm | StrOutputParser()
    )
    
    try:
        response = chain.invoke({})
        return {
            "answer": response,
            "intent": intent,
            "context_used": len(context.split('\n')) if context else 0,
            "query_enhanced": enhance_query(query, intent) != query
        }
    except Exception as e:
        return {
            "answer": f"I apologize, but I encountered an error processing your request: {str(e)}",
            "intent": intent,
            "context_used": 0,
            "query_enhanced": False,
            "error": True
        }

# All functionality now handled by the enhanced ask endpoint

# ------------ API models ------------
class Q(BaseModel): 
    query: str

class CVPayload(BaseModel):
    cv: Dict

# ------------ endpoints ------------
@app.get("/")
def root(): 
    return {
        "ok": True, 
        "service": "Enhanced CV RAG API - Unified Ask Endpoint",
        "version": "3.0",
        "primary_endpoint": "/ask",
        "capabilities": [
            "General Q&A about experience and skills",
            "Role-targeted summaries (mention specific roles/companies)",
            "Cover letter generation (ask for cover letters)",
            "STAR format examples (ask for examples/stories)",
            "Technical deep-dives (ask for technical details)",
            "Interview preparation (ask interview-related questions)"
        ],
        "features": [
            "Intent auto-detection from natural language queries",
            "Advanced embedding model (all-mpnet-base-v2)",
            "MMR retrieval for diverse, relevant results",
            "Context-aware response formatting",
            "Multi-strategy document retrieval"
        ],
        "examples": [
            "What experience do I have with Kubernetes?",
            "Create a summary for a Senior DevOps Engineer role at AWS",
            "Write a cover letter for a Full Stack Developer position",
            "Give me STAR examples of leadership",
            "Explain the technical details of the migration project",
            "How should I prepare for questions about cloud architecture?"
        ]
    }

@app.post("/set-cv")
def set_cv(payload: CVPayload, authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = payload.cv
    # persist and rebuild
    with open("cv.json", "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    global retriever
    retriever = build_retriever()
    return {
        "ok": True, 
        "message": "CV data saved and search index rebuilt", 
        "stats": {
            "skills": len(cv.get("skills", [])), 
            "experience_roles": len(cv.get("experience", [])),
            "projects": len(cv.get("projects", [])),
            "total_documents": len(load_cv_docs())
        }
    }

@app.post("/ask")
def ask(q: Q, authorization: str | None = Header(default=None)):
    """
    Enhanced ask endpoint with automatic intent detection and specialized responses.
    
    Supports:
    - General Q&A: "What experience do I have with Python?"
    - Role targeting: "Create a summary for Software Engineer at Google"
    - Cover letters: "Write a cover letter for DevOps role"
    - STAR examples: "Give me examples of leadership"
    - Technical details: "Explain the Kubernetes migration architecture"
    - Interview prep: "How to answer questions about cloud experience?"
    """
    guard(authorization)
    
    if not q.query.strip():
        return {"error": "Query cannot be empty"}
    
    response = create_enhanced_response(q.query.strip())
    return response
