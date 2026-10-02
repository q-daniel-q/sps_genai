import spacy


class EmbeddingModel:
    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)       # load once at startup

    def embed(self, word: str) -> dict:
        doc = self.nlp(word)
        return {
            "word": word,
            "model": self.model_name,
            "dimension": len(doc.vector),
            "has_vector": bool(doc.vector_norm > 0),   # False = out of vocabulary
            "embedding": doc.vector.tolist(),          # numpy -> list for JSON
        }

    def similarity(self, word1: str, word2: str) -> float:
        return float(self.nlp(word1).similarity(self.nlp(word2)))