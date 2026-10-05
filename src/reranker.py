from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class SemanticReranker:
    """Applies a cross-encoder model to rescore and rerank candidate chunks for precision[cite: 1]."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        if not docs:
            return []

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(pairs)

        # Sort documents by cross-encoder score descending
        doc_score_pairs: List[Tuple[Document, float]] = sorted(
            zip(docs, scores), key=lambda x: x[1], reverse=True
        )

        return [doc for doc, _ in doc_score_pairs[:top_n]]
