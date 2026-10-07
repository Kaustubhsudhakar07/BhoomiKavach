"""
BHOOmiKAVACH AI — Historical Accident Similarity Matcher (FAISS)
==================================================================
Builds a semantic & attribute-based vector search engine across the 41
DGMS primary fatal accidents (2016-2022) and 144 historical major falls (1901-2022).

Uses:
  - Local HuggingFace sentence transformer (`all-MiniLM-L6-v2`) with zero-cost embedding
  - FAISS Flat Inner Product (Cosine Similarity)
  - Metadata filtering by Coal/Non-Coal, State, and Mineral
  - Fallback TF-IDF cosine similarity if offline / torch memory constraint

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Optional

INDEX_DIR = "models/similarity_index"
os.makedirs(INDEX_DIR, exist_ok=True)


class AccidentSimilarityMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = None
        self.faiss_index = None
        self.records_df = None
        self.embedding_dim = 384
        self.is_fitted = False

    def _load_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[WARN] Failed to load SentenceTransformer: {e}. Falling back to TF-IDF.")
                self.encoder = None

    def fit_from_database(self, csv_path: str = "data/processed/roof_fall_accidents.csv"):
        """Indexes all accident records into FAISS."""
        print(f"Loading accidents from {csv_path}...")
        self.records_df = pd.read_csv(csv_path)
        
        # Build composite document string for each accident
        docs = []
        for idx, row in self.records_df.iterrows():
            doc = (
                f"Accident Code: 0111 Fall of Roof. "
                f"Mine: {row.get('mine', '')}, Company: {row.get('owner', '')}, State: {row.get('state', '')}. "
                f"Mineral: {row.get('mineral', '')}, Coal/Non-Coal: {row.get('coal_noncoal', '')}. "
                f"Place: {row.get('place', '')}. "
                f"Description: {row.get('description', '')}. "
                f"Contributing Factors: {row.get('contributing_factors', '')}. "
                f"Root Cause: {row.get('root_cause', '')}. "
                f"Violations: {row.get('regulation_reference', '')}."
            )
            docs.append(doc)

        self._load_encoder()
        if self.encoder is not None:
            print("Encoding accident narratives via SentenceTransformer...")
            embeddings = self.encoder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype("float32")
            faiss.normalize_L2(embeddings)
            self.embedding_dim = embeddings.shape[1]

            # Build FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(embeddings)
            print(f"Indexed {len(docs)} accidents into FAISS (dim={self.embedding_dim}).")
            self.is_fitted = True

            # Save artifacts
            faiss.write_index(self.faiss_index, os.path.join(INDEX_DIR, "faiss_accidents.bin"))
            self.records_df.to_pickle(os.path.join(INDEX_DIR, "records_metadata.pkl"))
        else:
            # Fallback to TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(max_features=500, stop_words="english")
            tfidf_mat = self.tfidf.fit_transform(docs).toarray().astype("float32")
            faiss.normalize_L2(tfidf_mat)
            self.faiss_index = faiss.IndexFlatIP(tfidf_mat.shape[1])
            self.faiss_index.add(tfidf_mat)
            self.is_fitted = True

        return len(docs)

    def load_index(self):
        """Loads pre-built index and metadata from disk."""
        index_path = os.path.join(INDEX_DIR, "faiss_accidents.bin")
        meta_path = os.path.join(INDEX_DIR, "records_metadata.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.faiss_index = faiss.read_index(index_path)
            self.records_df = pd.read_pickle(meta_path)
            self._load_encoder()
            self.is_fitted = True
            return True
        return False

    def query_similar_cases(self, query_text: str, top_k: int = 3, filter_coal_noncoal: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar historical roof-fall accidents matching the scenario.
        """
        if not self.is_fitted:
            if not self.load_index():
                self.fit_from_database()

        if self.encoder is not None:
            q_emb = self.encoder.encode([query_text], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q_emb)
        else:
            q_emb = self.tfidf.transform([query_text]).toarray().astype("float32")
            faiss.normalize_L2(q_emb)

        # Retrieve a wider pool to allow metadata filtering if requested
        search_k = min(len(self.records_df), top_k * 4)
        scores, indices = self.faiss_index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.records_df.iloc[idx]

            # Filter if specified
            if filter_coal_noncoal and str(row.get("coal_noncoal", "")).lower() != filter_coal_noncoal.lower():
                continue

            results.append({
                "accident_id": row.get("accident_id"),
                "similarity_score": round(float(score), 4),
                "year": int(row.get("year", 0)),
                "date": str(row.get("date", "")),
                "mine": str(row.get("mine", "")),
                "owner": str(row.get("owner", "")),
                "state": str(row.get("state", "")),
                "coal_noncoal": str(row.get("coal_noncoal", "")),
                "fatalities": int(row.get("fatalities", 1)),
                "place": str(row.get("place", "")),
                "description": str(row.get("description", ""))[:300] + "...",
                "contributing_factors": str(row.get("contributing_factors", "")),
                "root_cause": str(row.get("root_cause", "")),
                "regulations": str(row.get("regulation_reference", ""))
            })

            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    matcher = AccidentSimilarityMatcher()
    matcher.fit_from_database()
    
    test_query = "Depillaring stook extraction near active goaf edge with sandstone roof parting and missing props"
    matches = matcher.query_similar_cases(test_query, top_k=2)
    print("\n--- TEST QUERY SIMILARITY MATCHES ---")
    print(f"Query: '{test_query}'\n")
    for m in matches:
        print(f"[{m['similarity_score']*100:.1f}% Match] {m['mine']} ({m['year']}, {m['state']})")
        print(f"  Fatalities: {m['fatalities']} | Mineral: {m['coal_noncoal']}")
        print(f"  Root Cause: {m['root_cause']}")
        print(f"  Regulations: {m['regulations']}\n")
