"""Load the trained CNN and classify an uploaded image."""

import io
import os

import torch
import torch.nn.functional as F
from PIL import Image

from app.cnn_model import CLASSES, AssignmentCNN, transform

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_cifar10.pth")


class ImageClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model = AssignmentCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()  # inference mode, never training mode here

    def predict(self, image_bytes: bytes) -> dict:
        # Convert any input (PNG, JPEG, grayscale, RGBA) to a 64x64 RGB tensor
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)  # add a batch dimension: 1x3x64x64

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        top = int(torch.argmax(probs))
        return {
            "predicted_class": CLASSES[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        }