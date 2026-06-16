import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our custom services
from services.parser import parse_document
from services.chunker import chunk_text
from services.embedder import embedder_instance
from services.vector_store import vector_store_instance
from services.agent import run_agent

app = FastAPI(title="RAG Knowledge Assistant API")

# Allow the Angular frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = None
    try:
        # 1. Save uploaded file temporarily to disk
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Parse text from document
        print(f"[UPLOAD] Parsing: {file.filename}")
        extracted_text = parse_document(temp_path)
        if not extracted_text:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")

        print(f"[UPLOAD] Parsed {len(extracted_text)} characters. Clearing old vector store...")

        # 3. Clear previous document data so new file is the only source
        vector_store_instance.clear_all()

        # 4. Chunk the extracted text
        chunks = chunk_text(extracted_text)
        print(f"[UPLOAD] Created {len(chunks)} chunks.")

        # 5. Generate embeddings for all chunks
        embeddings = embedder_instance.embed_batch(chunks)

        # 6. Save chunks and embeddings to Vector Database
        vector_store_instance.add_chunks(texts=chunks, embeddings=embeddings, source_name=file.filename)
        print(f"[UPLOAD] Stored {len(chunks)} chunks in vector DB.")

        return {"message": f"Successfully processed {file.filename}", "chunks_added": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[UPLOAD ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Always clean up temp files
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/chat")
async def chat(request: ChatRequest):
    import re
    import numpy as np

    # Common stop words to ignore when extracting keywords
    STOP_WORDS = {
        "what", "is", "are", "was", "were", "my", "your", "his", "her", "their",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "and",
        "or", "but", "not", "do", "does", "did", "has", "have", "had", "i",
        "me", "we", "he", "she", "they", "it", "this", "that", "how", "when",
        "where", "who", "which", "tell", "about", "give", "list", "show", "find",
        "please", "can", "could", "would", "should", "be", "been", "being"
    }

    try:
        query = request.query.strip()

        # Step 1: Extract keywords from the query (remove stop words)
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        print(f"[CHAT] Query: '{query}' | Keywords: {keywords}")

        # Step 2: Embed query and retrieve top 5 chunks via vector similarity
        query_embedding = embedder_instance.embed_text(query)
        relevant_chunks = vector_store_instance.query_similar_chunks(query_embedding, n_results=3)

        if not relevant_chunks:
            return {
                "answer": "No relevant content found in the document for your query. Try different keywords.",
                "sources_used": 0
            }

        # Step 3: Keyword search — find lines that contain the keywords (whole-word match)
        matched_lines = []
        if keywords:
            for chunk in relevant_chunks:
                lines = [l.strip() for l in re.split(r'\n+', chunk) if l.strip()]
                for line in lines:
                    # Use word-boundary matching so "name" won't hit "nandannaik909"
                    hit_count = sum(
                        1 for kw in keywords
                        if re.search(rf'\b{re.escape(kw)}\b', line, re.IGNORECASE)
                    )
                    if hit_count > 0:
                        matched_lines.append((hit_count, line))

        # Step 4a: If keyword matches found, return the best matching lines
        if matched_lines:
            matched_lines.sort(key=lambda x: x[0], reverse=True)
            # Deduplicate and take top 3 unique lines
            seen = set()
            top_lines = []
            for _, line in matched_lines:
                if line not in seen:
                    seen.add(line)
                    top_lines.append(line)
                if len(top_lines) == 3:
                    break
            answer = "\n".join(top_lines)
            print(f"[CHAT] Keyword match: {top_lines}")
            return {"answer": answer, "sources_used": len(relevant_chunks)}

        # Step 4b: Fallback — cosine similarity re-ranking at sentence level
        sentences = []
        for chunk in relevant_chunks:
            parts = re.split(r'(?<=[.!?])\s+|\n+', chunk)
            sentences.extend([s.strip() for s in parts if len(s.strip()) > 20])

        if not sentences:
            return {"answer": relevant_chunks[0], "sources_used": 1}

        sentence_embeddings = embedder_instance.embed_batch(sentences)
        query_vec = np.array(query_embedding)
        scores = []
        for i, sent_emb in enumerate(sentence_embeddings):
            sent_vec = np.array(sent_emb)
            sim = np.dot(query_vec, sent_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(sent_vec) + 1e-9)
            scores.append((sim, sentences[i]))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scores[:3]]
        answer = "\n".join(top_sentences)
        return {"answer": answer, "sources_used": len(relevant_chunks)}

    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")


@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    """Agent-powered chat endpoint using CrewAI and Groq."""
    try:
        print(f"[AGENT] Query: {request.query}")
        result = await run_agent(request.query)
        print(f"[AGENT] Steps: {result['steps']}")
        return {
            "answer": result["answer"],
            "steps": result["steps"],
            "sources_used": len(result["steps"])
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[AGENT ERROR] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
