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

class EmbeddingsGetResponse(BaseModel):
    embeddings: str