import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingsModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get(self, inp: str) -> np.ndarray:
        return self._text_to_vec([inp])[0]
    
    def get_batch(self, inp: list) -> np.ndarray:
        return self._text_to_vec(inp)
   
    def _text_to_vec(self, texts: list) -> np.ndarray:
        embeddings = self.model.encode(texts)
        return embeddings
  
