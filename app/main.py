from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(title="SPS GenAI API")

# Sample corpus for the bigram model (from Module 3 activity)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_md")


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


# ---------- Assignment 1: word embedding ----------
@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="word must not be empty")
    return embedding_model.embed(word)


@app.get("/embedding/{word}")
def get_embedding_by_path(word: str):
    return embedding_model.embed(word)


@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    return {"word1": word1, "word2": word2,
            "similarity": embedding_model.similarity(word1, word2)}