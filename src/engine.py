import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import EMBEDDING_MODEL, LLM_MODEL
from src.preprocessor import EnterpriseDocumentPreprocessor
from src.vector_store import EnterpriseVectorStore
from src.reranker import SemanticReranker

class EnterpriseRAGEngine:
    """
    End-to-end RAG orchestrator:
    Filtering by role -> Semantic Search -> Cross-Encoder Reranking -> Grounded Synthesis[cite: 1].
    """

    def __init__(self):
        self.preprocessor = EnterpriseDocumentPreprocessor()
        self.vector_store = EnterpriseVectorStore(embedding_model=EMBEDDING_MODEL)
        self.reranker = SemanticReranker()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

    def index_data(self, json_filepath: str):
        with open(json_filepath, "r") as f:
            raw_data = json.load(f)
        docs = self.preprocessor.process(raw_data)
        self.vector_store.build_index(docs)

    def query(self, user_role: str, user_query: str) -> Dict[str, Any]:
        # Step 1: Candidate retrieval
        candidates = self.vector_store.search_candidates(user_query, top_k=8)

        # Step 2: Role-based access control (RBAC) filtering
        authorized_docs = [
            doc for doc in candidates 
            if user_role in doc.metadata.get("allowed_roles", [])
        ]

        if not authorized_docs:
            return {
                "answer": "No documents matched your query within your authorization scope.",
                "sources": []
            }

        # Step 3: Semantic reranking[cite: 1]
        top_chunks = self.reranker.rerank(user_query, authorized_docs, top_n=2)

        # Step 4: Grounded context assembly & synthesis[cite: 1]
        context = "\n\n".join(
            [f"[Source {d.metadata['doc_id']}]: {d.page_content}" for d in top_chunks]
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an enterprise banking compliance assistant. "
                       "Answer strictly using the provided sources. If uncertain, state that the context lacks the answer."),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])

        chain = prompt | self.llm
        response = chain.invoke({"context": context, "query": user_query})

        return {
            "answer": response.content,
            "sources": [
                {
                    "doc_id": d.metadata["doc_id"],
                    "classification": d.metadata["classification"],
                    "chunk": d.page_content
                }
                for d in top_chunks
            ]
        }

if __name__ == "__main__":
    engine = EnterpriseRAGEngine()
    engine.index_data("data/sample_enterprise_data.json")

    print("--- Test 1: Employee Role Query (Allowed: KYC) ---")
    res1 = engine.query(user_role="Employee", user_query="What is the deadline for client KYC refresh?")
    print(res1["answer"])
    print(f"Sources: {[s['doc_id'] for s in res1['sources']]}\n")

    print("--- Test 2: Employee Role Query (Blocked: Margin Restructuring) ---")
    res2 = engine.query(user_role="Employee", user_query="What is the authorized margin limit for portfolio 8892-A?")
    print(res2["answer"])
    print(f"Sources: {[s['doc_id'] for s in res2['sources']]}\n")

    print("--- Test 3: Managing Director Query (Authorized for Margin Restructuring) ---")
    res3 = engine.query(user_role="Managing_Director", user_query="What is the authorized margin limit for portfolio 8892-A?")
    print(res3["answer"])
    print(f"Sources: {[s['doc_id'] for s in res3['sources']]}")
