import os
import re
import asyncio
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun

from services.embedder import embedder_instance
from services.vector_store import vector_store_instance

load_dotenv()


# RAG Context Retrieval 

def retrieve_context(query: str) -> str:
  
    context_parts = []

    #  Step 1: Vector / semantic similarity search
    try:
        print(f"[RAG] Running vector_search for: '{query}'")
        embedding = embedder_instance.embed_text(query)
        chunks = vector_store_instance.query_similar_chunks(embedding, n_results=5)
        if chunks:
            context_parts.append("=== Semantically Relevant Passages ===")
            context_parts.extend(chunks)
            print(f"[RAG] Vector search returned {len(chunks)} chunks.")
        else:
            print("[RAG] Vector search returned no results.")
    except Exception as e:
        print(f"[RAG] vector_search error: {e}")

    # --- Step 2: Keyword search for each significant word in the query ---
    try:
        # Extract meaningful keywords (skip stopwords)
        stopwords = {"what", "is", "the", "a", "an", "in", "of", "for", "are", "was",
                     "were", "their", "my", "your", "his", "her", "its", "how", "many",
                     "which", "who", "when", "where", "tell", "me", "about", "give", "list"}
        words = [w for w in re.findall(r'\b\w{3,}\b', query.lower()) if w not in stopwords]

        all_data = vector_store_instance.collection.get()
        all_chunks = all_data.get("documents", [])

        keyword_hits = []
        seen = set()
        for keyword in words[:4]:  # check top 4 keywords
            for chunk in all_chunks:
                lines = [l.strip() for l in re.split(r'\n+', chunk) if l.strip()]
                for line in lines:
                    if re.search(rf'\b{re.escape(keyword)}\b', line, re.IGNORECASE):
                        if line not in seen:
                            seen.add(line)
                            keyword_hits.append(line)

        if keyword_hits:
            context_parts.append("=== Keyword Matches ===")
            context_parts.extend(keyword_hits[:15])
            print(f"[RAG] Keyword search returned {len(keyword_hits)} lines.")
    except Exception as e:
        print(f"[RAG] keyword_search error: {e}")

    if not context_parts:
        return ""  # No document content — agent will rely on web search only

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# LangGraph Multi-Agent Pipeline
# ─────────────────────────────────────────────

# Define the State schema for our graph
class AgentState(TypedDict):
    query: str
    context: str
    web_context: str
    extracted_facts: str
    analysis: str
    final_answer: str

def init_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in backend/.env")
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0.1)

# Node 1: Retrieval Specialist
def retrieval_specialist_node(state: AgentState):
    print("[Agent 1] Retrieval Specialist processing...")
    
    # If no document was uploaded or no relevant chunks found, skip LLM call
    if not state["context"] or not state["context"].strip():
        print("[Agent 1] No document context available — skipping retrieval.")
        return {"extracted_facts": "No document has been uploaded, or no relevant content was found in the document for this query."}

    llm = init_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at reading document excerpts of any type — resumes, financial reports, "
                   "contracts, product specs, research papers, legal documents, and more. "
                   "Your job is to extract exactly the pieces of information that are relevant to the user's question. "
                   "Filter out noise, remove duplication, and organize the key facts clearly and logically."),
        ("user", "User Question: {query}\n\n"
                 "--- Raw Document Context ---\n{context}\n--- End of Context ---\n\n"
                 "Extract ONLY the information from the context above that is relevant to the user's question. "
                 "Organize the extracted facts clearly. Do not add anything not present in the context.")
    ])
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "context": state["context"]})
    return {"extracted_facts": response.content}

# Node: Web Search Specialist
def web_search_node(state: AgentState):
    print("[Agent] Web Search processing...")
    search = DuckDuckGoSearchRun()
    try:
        web_results = search.invoke(state["query"])
        print(f"[Agent] Web search retrieved context: {len(web_results)} chars")
    except Exception as e:
        print(f"[Agent] Web search failed: {e}")
        web_results = "Web search failed or returned no results."
        
    return {"web_context": web_results}

# Node 2: Document Analyst
def analyst_node(state: AgentState):
    print("[Agent 2] Document Analyst processing...")
    llm = init_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert analyst who synthesizes information from two sources:\n"
         "  - LOCAL DOCUMENT FACTS: Extracted content from an uploaded document (could be a resume, "
         "financial report, contract, research paper, product spec, legal doc, or any other document type).\n"
         "  - INTERNET WEB CONTEXT: Live search results from the web.\n\n"
         "Apply the correct reasoning mode based on the user's query:\n"
         "1. DOCUMENT-ONLY: The user is asking about content within the uploaded document "
         "   (e.g., 'what does this report say about revenue?', 'list the clauses in this contract', 'what are my skills'). "
         "   → Use Local Document Facts only.\n"
         "2. WEB-ONLY: The user is asking about external, real-world, or current information "
         "   (e.g., 'what are industry trends?', 'what skills are in demand?', 'what is the market rate?'). "
         "   → Use Internet Web Context only.\n"
         "3. CROSS-SOURCE: The user wants a comparison, gap analysis, or recommendation that requires "
         "   combining document content with external knowledge "
         "   (e.g., 'compare my financials with industry benchmarks', 'does this contract meet standard terms?', "
         "   'am I qualified for this role?', 'how does this product compare to the market?'). "
         "   → Synthesize BOTH sources. Extract the relevant facts from the document, the relevant benchmarks/standards "
         "   from the web, then reason across both to give a structured comparison or recommendation.\n"
         "4. If NEITHER source contains relevant data, respond: 'I don't have enough information to answer that.'\n\n"
         "DO NOT hallucinate. Only use facts explicitly present in the provided sources."),
        ("user", "User Question: {query}\n\n"
                 "--- Local Document Facts (from uploaded document) ---\n{extracted_facts}\n"
                 "----------------------------------------------------\n\n"
                 "--- Internet Web Context (from live search) ---\n{web_context}\n"
                 "-----------------------------------------------\n\n"
                 "Identify which reasoning mode applies and answer accordingly.")
    ])
    chain = prompt | llm
    response = chain.invoke({
        "query": state["query"],
        "extracted_facts": state["extracted_facts"],
        "web_context": state.get("web_context", "No web context retrieved.")
    })
    return {"analysis": response.content}

# Node 3: Response Writer
def writer_node(state: AgentState):
    print("[Agent 3] Response Writer processing...")
    llm = init_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a direct, conversational AI assistant. "
                   "You take raw analytical output and provide the final answer to the user. "
                   "You must be completely invisible to the user — NEVER mention the 'Analyst', 'Local Document Facts', "
                   "or 'Internet Web Context'. Just answer the question directly and naturally."),
        ("user", "Original User Question: {query}\n\n"
                 "You will receive the analyst's full answer. Your job is to format it into the final response:\n"
                 "- Give just the answer. \n"
                 "- Do NOT include any filler text like 'Key points:', 'Here is the answer', or 'Based on the context'.\n"
                 "- Do NOT explain where the data came from (e.g., do not say 'obtained from local document').\n"
                 "- If the question is simple (like a name), just give the name. Only use bullet points if the user asked for a list or summary.\n\n"
                 "Analyst's Output:\n{analysis}")
    ])
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "analysis": state["analysis"]})
    return {"final_answer": response.content}

def create_graph():
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes representing the agents
    workflow.add_node("retriever", retrieval_specialist_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    # Define edges (sequential workflow)
    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "web_search")
    workflow.add_edge("web_search", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", END)
    
    return workflow.compile()


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────
async def run_agent(query: str) -> dict:
    """
    Full pipeline migrated to LangGraph:
    1. Manually retrieve context via RAG
    2. Pass state to the LangGraph workflow
    3. Return final condensed answer
    """
    print(f"[AGENT] Query received: {query}")

    # Step 1: RAG retrieval
    context = retrieve_context(query)
    print(f"[AGENT] Context length: {len(context)} chars")

    # Step 2: Initialize graph and state
    graph = create_graph()
    initial_state = {
        "query": query,
        "context": context,
        "web_context": ""
    }
    
    # Run the graph synchronously in a separate thread
    def invoke_graph():
        return graph.invoke(initial_state)
        
    result = await asyncio.to_thread(invoke_graph)
    
    # Extract final answer
    answer = result.get("final_answer", "")
    print(f"[AGENT] Final answer: {answer[:200]}")

    steps = [
        {"tool": "vector_search", "input": f"Semantic search for: '{query[:60]}'"},
        {"tool": "keyword_search", "input": f"Keyword scan for key terms in query"},
        {"tool": "duckduckgo_search", "input": f"Live web search for: '{query[:60]}'"},
        {"tool": "LangGraph Pipeline (Groq)", "input": f"Multi-node analysis over local & web context"},
    ]

    return {
        "answer": answer.strip(),
        "steps": steps
    }
