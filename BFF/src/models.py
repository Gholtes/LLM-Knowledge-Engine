from typing import Optional, List
from pydantic import BaseModel

class ExampleRequest(BaseModel):
    name: str
    age: int

class ExampleResponse(BaseModel):
    name: str
    age: int
    id: str

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    summary: str
    document_ids: List(str)