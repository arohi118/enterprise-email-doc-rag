from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class EnterpriseVectorStore:
    """Manages vector embeddings and retrieval indices."""

    def __init__(self, embedding_model: str):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.index = None

    def build_index(self, documents: List[Document]):
        """Builds in-memory FAISS index from document chunks[cite: 1]."""
        self.index = FAISS.from_documents(documents, self.embeddings)

    def search_candidates(self, query: str, top_k: int = 10) -> List[Document]:
        """Retrieves semantic nearest-neighbor candidates prior to reranking[cite: 1]."""
        if not self.index:
            raise ValueError("Vector index is not initialized.")
        return self.index.similarity_search(query, k=top_k)
