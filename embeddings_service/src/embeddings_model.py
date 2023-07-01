import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM


class EmbeddingsModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-small")
        self.model = AutoModelForMaskedLM.from_pretrained("bert-small")

    def get(self, inp: str) -> np.ndarray:
        return self._text_to_vec(inp)
   
    def _text_to_vec(self, text: str) -> np.ndarray:
        encoded_input = self.tokenizer([text], return_tensors='pt')
        model_output = self.model(**encoded_input)
        tensor = self._mean_pooling(model_output, encoded_input['attention_mask'])
        numpy_array = tensor.detach().numpy()
        return numpy_array[0]

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
  
