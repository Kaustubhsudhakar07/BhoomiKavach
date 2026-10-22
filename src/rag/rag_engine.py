"""
BHOOmiKAVACH AI — Local RAG Engine with FAISS (Phase 21)
========================================================
Implements local semantic retrieval over:
  - Coal Mines Regulations, 2017 & Metalliferous Mines Regulations, 1961
  - DGMS Technical Circulars (SCAMP, Roof Bolting, Testing)
  - DGMS Safety Alerts on Ground Movement / 0111 Fall of Roof
  - Historical DGMS Inquiry Findings and Preventive Recommendations (SANKET)

Vector Store:
  FAISS IndexFlatIP (Cosine Similarity normalized)
  HuggingFace sentence-transformers (`all-MiniLM-L6-v2`)
  Serialized to `vector_store/faiss/`

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Optional

VECTOR_DIR = "vector_store/faiss"
os.makedirs(VECTOR_DIR, exist_ok=True)

KB_FILES = [
    "knowledge_base/regulations/cmr_and_mmr_regulations.json",
    "knowledge_base/dgms/technical_circulars.json",
    "knowledge_base/safety_alerts/roof_fall_safety_alerts.json"
]


class LocalRagEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.embedding_dim = 384
        self.is_fitted = False

    def _load_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer for RAG: {e}. Falling back to TF-IDF.")
                self.encoder = None

    def build_and_index_knowledge_base(self, csv_accidents: str = "data/processed/roof_fall_accidents.csv"):
        """Collects all regulatory documents and historical accident inquiry lessons, then indexes into FAISS."""
        docs = []

        # 1. Load structured JSON knowledge base documents
        for kbf in KB_FILES:
            if os.path.exists(kbf):
                with open(kbf, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                    for item in entries:
                        docs.append({
                            "doc_id": item["doc_id"],
                            "title": item["title"],
                            "category": item["category"],
                            "content": item["text"],
                            "source": item["source"],
                            "page": item["page"],
                            "topics": ", ".join(item.get("topics", []))
                        })

        # 2. Ingest DGMS inquiry findings from historical dataset A
        if os.path.exists(csv_accidents):
            df_acc = pd.read_csv(csv_accidents)
            for _, row in df_acc.iterrows():
                doc_text = (
                    f"Official DGMS Roof-Fall Inquiry finding at {row.get('mine', '')} ({row.get('year', '')}, {row.get('state', '')}). "
                    f"Activity: {row.get('activity', '')}. "
                    f"Root Cause: {row.get('root_cause', '')}. "
                    f"Violations: {row.get('regulation_reference', 'General Safety Failure')}. "
                    f"Mandatory Statutory Remedial Action: {row.get('preventive_action', '')}"
                )
                docs.append({
                    "doc_id": f"DGMS-CASE-{row.get('accident_id', '')}",
                    "title": f"Inquiry Remedial Finding: {row.get('mine', '')} ({row.get('year', '')})",
                    "category": "Historical DGMS Case Precedent",
                    "content": doc_text,
                    "source": str(row.get("source_document", "DGMS SANKET")),
                    "page": str(row.get("source_page", "Official Finding")),
                    "topics": f"{row.get('mineral', '')}, {row.get('coal_noncoal', '')}, 0111 Fall of Roof"
                })

        self.documents = docs
        print(f"Total documents prepared for RAG index: {len(docs)}")

        # 3. Embed text
        texts_to_embed = [f"{d['title']}\n{d['content']}\nTopics: {d['topics']}" for d in docs]
        self._load_encoder()

        if self.encoder is not None:
            embeddings = self.encoder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            faiss.normalize_L2(embeddings)
            self.embedding_dim = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings)

            # Save FAISS and metadata
            faiss.write_index(self.index, os.path.join(VECTOR_DIR, "rag_index.bin"))
            with open(os.path.join(VECTOR_DIR, "rag_metadata.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Saved FAISS RAG index to {VECTOR_DIR} (vectors={len(docs)}, dim={self.embedding_dim})")
            self.is_fitted = True
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(max_features=600, stop_words="english")
            tfidf_mat = self.tfidf.fit_transform(texts_to_embed).toarray().astype("float32")
            faiss.normalize_L2(tfidf_mat)
            self.index = faiss.IndexFlatIP(tfidf_mat.shape[1])
            self.index.add(tfidf_mat)
            with open(os.path.join(VECTOR_DIR, "rag_metadata.pkl"), "wb") as f:
                pickle.dump({"docs": self.documents, "tfidf": self.tfidf}, f)
            self.is_fitted = True

        return len(docs)

    def load_index(self):
        idx_path = os.path.join(VECTOR_DIR, "rag_index.bin")
        meta_path = os.path.join(VECTOR_DIR, "rag_metadata.pkl")
        if os.path.exists(idx_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(idx_path)
            with open(meta_path, "rb") as f:
                self.documents = pickle.load(f)
            self._load_encoder()
            self.is_fitted = True
            return True
        return False

    def query_evidence(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top-k statutory regulations, technical circulars, or historical findings
        supporting a specific safety question or situation.
        """
        if not self.is_fitted:
            if not self.load_index():
                self.build_and_index_knowledge_base()

        if self.encoder is not None:
            q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
        else:
            q_emb = self.tfidf.transform([query]).toarray().astype("float32")
            faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc["category"],
                "content": doc["content"],
                "source": doc["source"],
                "page": doc["page"],
                "similarity_score": round(float(score), 4),
                "confidence": "HIGH" if score > 0.60 else ("MEDIUM" if score > 0.35 else "LOW")
            })

        return results


    def add_documents(self, new_docs: List[Dict[str, Any]]) -> int:
        """
        Dynamically adds new documents to the existing FAISS RAG index at runtime.
        Each doc should have: title, content, source, category, page (optional), topics (optional).
        Returns the number of documents successfully added.
        """
        if not self.is_fitted:
            if not self.load_index():
                self.build_and_index_knowledge_base()

        if not new_docs:
            return 0

        self._load_encoder()

        # Prepare text for embedding
        texts_to_embed = []
        for d in new_docs:
            d.setdefault("doc_id", f"USER-UPLOAD-{len(self.documents) + len(texts_to_embed) + 1}")
            d.setdefault("category", "User Uploaded Document")
            d.setdefault("page", "Uploaded")
            d.setdefault("topics", "")
            texts_to_embed.append(f"{d['title']}\n{d['content']}\nTopics: {d.get('topics', '')}")

        if self.encoder is not None:
            embeddings = self.encoder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False).astype("float32")
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # TF-IDF fallback — need to re-fit, which is costly but functional
            all_texts = [f"{d['title']}\n{d['content']}\nTopics: {d.get('topics', '')}" for d in self.documents] + texts_to_embed
            tfidf_mat = self.tfidf.fit_transform(all_texts).toarray().astype("float32")
            faiss.normalize_L2(tfidf_mat)
            self.index = faiss.IndexFlatIP(tfidf_mat.shape[1])
            self.index.add(tfidf_mat)

        self.documents.extend(new_docs)

        # Persist updated index and metadata
        faiss.write_index(self.index, os.path.join(VECTOR_DIR, "rag_index.bin"))
        with open(os.path.join(VECTOR_DIR, "rag_metadata.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        print(f"[RAG] Added {len(new_docs)} new documents. Total index size: {len(self.documents)}")
        return len(new_docs)


if __name__ == "__main__":
    rag = LocalRagEngine()
    rag.build_and_index_knowledge_base()

    test_q = "What statutory regulations mandate support setting at the goaf edge in depillaring panels?"
    matches = rag.query_evidence(test_q, top_k=2)
    print(f"\n--- RAG RETRIEVAL TEST ---")
    print(f"Query: '{test_q}'\n")
    for m in matches:
        print(f"[{m['category']} | Match: {m['similarity_score']*100:.1f}%] {m['title']}")
        print(f"  Source: {m['source']}, {m['page']}")
        print(f"  Excerpt: {m['content'][:200]}...\n")
