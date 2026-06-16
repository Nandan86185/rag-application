import os
from typing import List
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

api_key = os.environ.get("GROQ_API_KEY")
if api_key:
    client = Groq(api_key=api_key)
    print("Groq API configured successfully.")
else:
    client = None
    print("WARNING: GROQ_API_KEY not set. Add it to backend/.env")


def generate_answer(query: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "I couldn't find any relevant context in the uploaded documents to answer your question."

    if not client:
        return "⚠️ GROQ_API_KEY is not configured. Please add it to backend/.env"

    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Use ONLY the following context from the uploaded document to answer the question.
If the answer is not in the context, say you don't know based on the provided document.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        print(f"LLM ERROR: {err}")
        if "429" in err or "rate_limit" in err.lower():
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        if "401" in err or "invalid_api_key" in err.lower():
            return "⚠️ Invalid Groq API key. Please check your backend/.env file."
        return f"⚠️ Error: {err}"
