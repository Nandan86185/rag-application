import chromadb
import uuid
from typing import List

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="rag_documents")

    def clear_all(self):
        """Remove all chunks from the collection without deleting the collection itself."""
        existing = self.collection.get()
        if existing and existing['ids']:
            self.collection.delete(ids=existing['ids'])
            print(f"Vector store cleared ({len(existing['ids'])} old chunks removed).")
        else:
            print("Vector store already empty.")

    def add_chunks(self, texts: List[str], embeddings: List[List[float]], source_name: str):
        if not texts or not embeddings:
            return
            
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas = [{"source": source_name} for _ in texts]
        
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def query_similar_chunks(self, query_embedding: List[float], n_results: int = 3) -> List[str]:
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if results['documents'] and len(results['documents']) > 0:
            return results['documents'][0]
            
        return []

vector_store_instance = VectorStore()
