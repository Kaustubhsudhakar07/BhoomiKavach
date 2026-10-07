# BHOOmiKAVACH AI — Local RAG & Vector Search Architecture

## 1. Vector Database Architecture
- **Vector Database:** **FAISS (Facebook AI Similarity Search)** Flat Inner Product (`IndexFlatIP`).
- **Embedding Model:** Local HuggingFace SentenceTransformer `sentence-transformers/all-MiniLM-L6-v2` (Zero-cost, offline execution, 384-dimensional dense vectors).
- **Metric:** Cosine Similarity via L2 unit vector normalization prior to indexing (`faiss.normalize_L2`).
- **Storage Location:** `vector_store/faiss/rag_index.bin` and `vector_store/faiss/rag_metadata.pkl`.

---

## 2. Ingested Knowledge Repositories

### A. Statutory Mining Regulations
1. **Coal Mines Regulations, 2017 (CMR 2017):**
   - Regulation 111 (Gallery Dimensions & Statutory Maximum Span: 4.8m width, 3.0m height)
   - Regulation 112 (Depillaring & Secondary Pillar Extraction Permissions)
   - Regulation 123 (Securing of Workings & Systematic Support Rules)
   - Regulation 125 (Setting & Pull Testing of Roof Bolts)
   - Regulation 129 (Duties of Mining Sirdars & Overmen; Pre-shift Sounding Inspection)
2. **Metalliferous Mines Regulations, 1961 (MMR 1961):**
   - Regulation 112 (Securing of Underground Stopes, Drives, and Crosscuts)

### B. DGMS Technical Circulars
1. **Circular No. 03 of 2010:** Formulation of Strata Control and Monitoring Plan (SCAMP), dual-height tell-tale installation, RMR classification, and Trigger Action Response Plans (TARP).
2. **Circular No. 06 of 2013:** Standard specifications and routine pull testing of resin-grouted roof bolts (minimum 8-10 tonnes).
3. **Circular No. 02 of 2018:** Mandatory elimination of timber props in mechanized workings and setting of temporary supports at freshly blasted faces.
4. **Circular No. 04 of 2022:** Scientific strata management, microseismic monitoring, and shotcrete wire-meshing in metalliferous stopes.

### C. DGMS Safety Alerts
- Alert 12/2019: Hazards of goaf edge support setting during depillaring operations.
- Alert 04/2021: Hidden slickensides and slip planes in laminated roof strata.
- Alert 08/2022: Dynamic vibration shock waves and secondary delamination post solid blasting.

### D. Historical Inquiry Findings (SANKET 2016–2022)
- 41 verbatim inquiry summaries detailing actual failure modes, dimensions of detached rock slabs, root causes, and statutory remedial findings.

---

## 3. Query & Retrieval Pipeline
When a query is received:
1. The text is encoded into a 384-dimensional unit vector.
2. An exact inner product scan over the FAISS index identifies the top-K nearest document chunks.
3. Chunks are returned with metadata: `doc_id`, `title`, `category`, `source`, `page`, and `similarity_score`.
4. High-confidence citations (>0.60 match) are automatically bound into the decision-support response.
