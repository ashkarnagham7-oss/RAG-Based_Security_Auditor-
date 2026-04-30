from unixcoder import UniXcoder
from langchain_core.embeddings import Embeddings
from typing import List
import torch


class UniXcoderEmbeddings(Embeddings):
    def __init__(self, model_name="microsoft/unixcoder-base", batch_size=8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading UniXcoder model: {model_name}")
        self.model = UniXcoder(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] UniXcoder loaded on: {self.device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            tokens_ids = self.model.tokenize(
                batch,
                max_length=512,
                mode="<encoder-only>"
            )

            source_ids = torch.tensor(tokens_ids).to(self.device)

            with torch.no_grad():
                _, embeddings = self.model(source_ids)

            embeddings = embeddings.cpu().numpy().tolist()
            all_embeddings.extend(embeddings)

        return all_embeddings
