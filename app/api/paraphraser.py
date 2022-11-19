from typing import List,Any
from fastapi import Header, APIRouter
from app.api.models import Taxonomies
from app.api.models import Question
from app.api.reconstruction_paraqd_pipeline import paraphrase

paraphraser = APIRouter()

@paraphraser.post('/paraphrase',response_model=List[Any])
async def get_paraphrase(payload: Question):
    return paraphrase(payload.content)
