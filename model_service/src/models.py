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
    embeddings: str

class EmbeddingsGetBatchRequest(BaseModel):
    text: list
    encode: bool

class EmbeddingsGetBatchResponse(BaseModel):
    embeddings: list