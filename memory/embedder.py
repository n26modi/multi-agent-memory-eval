from graphiti_core.embedder.client import EmbedderClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class LocalEmbedder(EmbedderClient):
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    async def create(self, input_data) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        else:
            text = list(input_data)[0]
        return self.model.encode(text).tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.encode(input_data_list)]
