# sps_genai

APAN PS5560 Applied Generative AI (Yutao Fan)

A FastAPI service that grows with each assignment:

1. **Assignment 1:** bigram text generation and spaCy word embeddings
2. **Assignment 2:** a CNN image classifier trained on CIFAR10

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/generate` | POST | Bigram text generation, body: `{"start_word": "the", "length": 10}` |
| `/embedding` | POST | Word embedding of a query word, body: `{"word": "apple"}` |
| `/embedding/{word}` | GET | Same as above, browser friendly |
| `/similarity?word1=apple&word2=car` | GET | Cosine similarity between two words |
| `/classify` | POST | **Upload an image, get its CIFAR10 class** (form field `file`) |

Example `/classify` response:

```json
{
  "filename": "test_0_cat.png",
  "predicted_class": "cat",
  "confidence": 0.8946,
  "probabilities": {"airplane": 0.0008, "cat": 0.8946, "dog": 0.0662, "...": "..."}
}
```

## Assignment 2: CNN architecture

Implemented in `app/cnn_model.py`, adapted from `SimpleCNN` in Module 4 Practical 3.

| Layer | Output shape |
|---|---|
| Input (RGB image) | 64 × 64 × 3 |
| Conv2D 16 filters, 3×3, stride 1, padding 1 + ReLU | 64 × 64 × 16 |
| MaxPool 2×2, stride 2 | 32 × 32 × 16 |
| Conv2D 32 filters, 3×3, stride 1, padding 1 + ReLU | 32 × 32 × 32 |
| MaxPool 2×2, stride 2 | 16 × 16 × 32 |
| Flatten | 8192 |
| Fully connected + ReLU | 100 |
| Fully connected (output) | 10 |

CIFAR10 images are 32 × 32, so they are resized to 64 × 64 to match the required input size. The same transform is used in training and in the API.

**Training:** 10 epochs, Adam (lr 0.0005), batch size 32, cross entropy loss. **Test accuracy: 66.23%** on the 10,000 CIFAR10 test images. The trained weights are saved in `app/models/cnn_cifar10.pth`, so the API works without retraining.

To retrain:

```bash
uv run python -m app.train_cnn
```

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
  bigram_model.py      # Assignment 1: bigram text generation
  embedding_model.py   # Assignment 1: spaCy word embeddings
  cnn_model.py         # Assignment 2: CNN architecture and image transform
  train_cnn.py         # Assignment 2: training script for CIFAR10
  image_classifier.py  # Assignment 2: loads the trained model and predicts
  models/
    cnn_cifar10.pth    # Assignment 2: trained weights
Dockerfile
pyproject.toml         # dependencies (CPU build of PyTorch on Linux)
uv.lock
```