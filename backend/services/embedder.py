from fastembed import TextEmbedding
from typing import List
import numpy as np

class Embedder:

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        print(f"Loading FastEmbed model: {model_name}...")
        try:
            self.model = TextEmbedding(model_name=model_name)
            print("FastEmbed model loaded successfully!")
        except Exception as e:
            print(f"FastEmbed load failed: {e}")
            raise e

    def embed_text(self, text: str) -> List[float]:
        # FastEmbed returns a generator of numpy arrays, we want a list of floats
        embedding_gen = self.model.embed([text])
        embedding_array = next(embedding_gen)
        return embedding_array.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # FastEmbed returns a generator of numpy arrays
        embedding_gen = self.model.embed(texts)
        return [emb.tolist() for emb in embedding_gen]

# Initialize the singleton instance
embedder_instance = Embedder()
