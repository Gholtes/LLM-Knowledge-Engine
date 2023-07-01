

class Interface:
    def __init__(self):
        pass

    def search_and_summerise(self, query):
        return "qwertyuiop", [1,2,3,4,5]
    
    def get_embeddings(self, text):
        return self._get_embeddings(text)
    
    def _get_embeddings(self, text):
        return [1,2,3,4,5,6,7,8,9]