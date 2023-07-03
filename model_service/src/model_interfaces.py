import logging
logger = logging.getLogger(__name__)

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, pipeline

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
  
class SummarisationModel:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="bart-large-cnn")
        self.tokens_per_word = 1.3

    def get(self, context, max_length=250, min_length=50):
        return self._get([context], max_length=max_length, min_length=min_length)[0]

    def _get(self, list_of_contexts: list, max_length=250, min_length=50) -> list:
        m = min([int(len(s.split(" "))*self.tokens_per_word) for s in list_of_contexts])
        max_length=min(m, max_length)
        min_length = min(min_length, max_length-1)
        logger.info(max_length)
        logger.info(min_length)
        # Get summary
        summ = self.summarizer(list_of_contexts, max_length=max_length, min_length=min_length, do_sample=False)
        logger.info(summ)
        logger.info(type(summ))
        extracted_text = [s["summary_text"] for s in summ]
        logger.info(extracted_text)
        return extracted_text