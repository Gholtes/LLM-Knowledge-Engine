from transformers import AutoTokenizer, AutoModelForMaskedLM

class EmbeddingsModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

    def get(self, inp):
        encoded_input = self.tokenizer("hello", return_tensors='pt')
        model_output = self.model(**encoded_input)
        return self._mean_pooling(model_output, encoded_input['attention_mask'])

    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
  
