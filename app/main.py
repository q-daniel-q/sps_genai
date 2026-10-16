from fastapi import FastAPI, File, HTTPException, UploadFile  # NEW: File, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel
from app.image_classifier import ImageClassifier  # NEW

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
image_classifier = ImageClassifier()  # NEW: loads app/models/cnn_cifar10.pth


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


# ---------- Assignment 2: CNN image classifier (NEW) ----------
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Upload an image and get its predicted CIFAR10 class."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="please upload an image file")
    image_bytes = await file.read()
    try:
        result = image_classifier.predict(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="could not read the image")
    return {"filename": file.filename, **result}