# sps_genai

APAN PS5560 Applied Generative AI, Assignment 1 (Yutao Fan)

A FastAPI service with a bigram text generator (Module 3 class activity) and a **word embedding endpoint** using spaCy `en_core_web_md` (Module 2).

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/generate` | POST | Bigram text generation, body: `{"start_word": "the", "length": 10}` |
| `/embedding` | POST | **Word embedding of a query word**, body: `{"word": "apple"}` |
| `/embedding/{word}` | GET | Same as above, browser friendly |
| `/similarity?word1=apple&word2=car` | GET | Cosine similarity between two words |

The `/embedding` response contains the word, model name, vector dimension (300), `has_vector` (false if out of vocabulary) and the embedding vector.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
```

## Project structure

```
app/
  main.py              # FastAPI app and endpoints
  bigram_model.py      # BigramModel: bigram probabilities and text generation
  embedding_model.py   # EmbeddingModel: spaCy word embeddings and similarity
Dockerfile
pyproject.toml         # dependencies, including the spaCy en_core_web_md model
uv.lock
```