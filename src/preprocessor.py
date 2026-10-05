from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class EnterpriseDocumentPreprocessor:
    """Preprocesses raw enterprise documents and emails into metadata-enriched chunks."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def process(self, raw_records: List[Dict[str, Any]]) -> List[Document]:
        documents = []
        for record in raw_records:
            full_text = f"Subject: {record.get('subject', '')}\nContent: {record['content']}"
            chunks = self.splitter.split_text(full_text)
            
            for idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": record["id"],
                        "chunk_id": f"{record['id']}_{idx}",
                        "type": record.get("type", "document"),
                        "department": record.get("department", "General"),
                        "classification": record.get("classification", "Public_Internal"),
                        "allowed_roles": record.get("allowed_roles", ["Employee"])
                    }
                )
                documents.append(doc)
        return documents
