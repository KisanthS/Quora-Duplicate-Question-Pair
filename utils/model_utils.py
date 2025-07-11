from sentence_transformers import SentenceTransformer, util
import torch

def load_model_and_similarity(model_path="model/sbert_quora_model"):
    """
    Loads a trained SentenceTransformer model and returns a function to compute cosine similarity.
    """
    model = SentenceTransformer(model_path)

    def get_similarity(q1, q2):
        emb1 = model.encode(q1, convert_to_tensor=True)
        emb2 = model.encode(q2, convert_to_tensor=True)
        score = util.cos_sim(emb1, emb2)
        return score.item()

    return model, get_similarity
