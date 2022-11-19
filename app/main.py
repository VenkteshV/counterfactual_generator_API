from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.paraphraser import paraphraser

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def index():
    return {"health": "hello this is the paraphrasing API"}

app.include_router(paraphraser)