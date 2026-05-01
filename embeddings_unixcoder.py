from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
import torch.nn.functional as F


class UniXcoderEmbeddings(Embeddings):
    def __init__(self, model_name: str = "microsoft/unixcoder-base", batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading UniXcoder model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] UniXcoder loaded on: {self.device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

            batch_embeddings = cls_embeddings.cpu().numpy().tolist()
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
