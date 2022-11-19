from typing import List,Any
from pydantic import BaseModel

class Taxonomies(BaseModel):
    taxonomy: str

class Question(BaseModel):
    content: str