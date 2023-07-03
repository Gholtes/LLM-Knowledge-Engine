from typing import Optional
from pydantic import BaseModel

class ExampleRequest(BaseModel):
    name: str
    age: int

class ExampleResponse(BaseModel):
    name: str
    age: int
    id: str

class EmbeddingsGetRequest(BaseModel):
    text: str
    encode: bool

class EmbeddingsGetResponse(BaseModel):
    embeddings_list: list
    embeddings_string: str

class EmbeddingsGetBatchRequest(BaseModel):
    texts: list
    encode: bool

class EmbeddingsGetBatchResponse(BaseModel):
    embeddings: list

class SummarisationRequest(BaseModel):
    context: str
    min_length: int
    max_length: int

class SummarisationResponse(BaseModel):
    summary: str
