"""
BHOOmiKAVACH AI — Regulatory & Case Precedent RAG Query Route (Phase 23)
========================================================================
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.rag.rag_engine import LocalRagEngine
import os, tempfile

router = APIRouter()
rag_engine = LocalRagEngine()


def _extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text content from PDF or TXT files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".txt" or ext == ".md" or ext == ".csv":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif ext == ".pdf":
        text_parts = []
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="No PDF reader library available. Install PyPDF2 or pdfplumber: pip install PyPDF2"
                )
        return "\n".join(text_parts)

    elif ext == ".json":
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)

    else:
        # Attempt to read as plain text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks for better RAG retrieval."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


@router.post("/upload")
async def upload_document_to_rag(file: UploadFile = File(...)):
    """Upload a PDF, TXT, or JSON file to add its content to the RAG knowledge base."""
    try:
        allowed_extensions = {".pdf", ".txt", ".md", ".csv", ".json"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(allowed_extensions)}"
            )

        # Save uploaded file temporarily
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Extract text
        try:
            raw_text = _extract_text_from_file(tmp_path, file.filename)
        finally:
            os.unlink(tmp_path)

        if not raw_text or not raw_text.strip():
            raise HTTPException(status_code=400, detail="No readable text could be extracted from the file.")

        # Chunk into documents
        chunks = _chunk_text(raw_text, chunk_size=800, overlap=150)
        new_docs = []
        for i, chunk in enumerate(chunks):
            new_docs.append({
                "doc_id": f"UPLOAD-{file.filename}-CHUNK-{i+1}",
                "title": f"{file.filename} (Chunk {i+1}/{len(chunks)})",
                "category": "User Uploaded Document",
                "content": chunk,
                "source": file.filename,
                "page": f"Chunk {i+1}",
                "topics": "User Upload"
            })

        # Add to FAISS index
        added_count = rag_engine.add_documents(new_docs)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": added_count,
            "total_characters": len(raw_text),
            "total_index_size": len(rag_engine.documents),
            "message": f"Successfully indexed {added_count} chunks from '{file.filename}' into the RAG knowledge base."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload and indexing failed: {str(e)}")


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 4


class RagAskRequest(BaseModel):
    query: str
    user_role: Optional[str] = "Safety Officer"
    top_k: int = 4
    gemini_api_key: Optional[str] = None


@router.post("/query")
def query_regulatory_rag(req: RagQueryRequest):
    try:
        evidence = rag_engine.query_evidence(req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "evidence_count": len(evidence),
            "evidence": evidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/ask")
def ask_regulatory_safety_advisor(req: RagAskRequest):
    try:
        evidence = rag_engine.query_evidence(req.query, top_k=req.top_k)

        # Try Gemini reasoning
        from backend.app.services.gemini_service import generate_gemini_safety_advisory, is_gemini_available
        gemini_answer = None
        gemini_powered = False

        if is_gemini_available(req.gemini_api_key):
            gemini_answer = generate_gemini_safety_advisory(
                query=req.query,
                user_role=req.user_role or "Safety Officer",
                rag_evidence=evidence,
                api_key=req.gemini_api_key
            )
            if gemini_answer:
                gemini_powered = True

        if not gemini_answer:
            # Deterministic fallback synthesis
            blocks = [f"STATUTORY REGULATORY ADVISORY (ROLE: {req.user_role})", f"QUERY: {req.query}\n"]
            if evidence:
                blocks.append(f"Top {len(evidence)} Relevant Provisions Identified from DGMS Knowledge Base:")
                for i, ev in enumerate(evidence, 1):
                    blocks.append(f"{i}. {ev.get('title')} [{ev.get('category')}]")
                    blocks.append(f"   Source: {ev.get('source')}, {ev.get('page')}")
                    blocks.append(f"   Provision Excerpt: {ev.get('content')}\n")
            else:
                blocks.append("No specific regulatory clauses directly matched this phrasing.")
            blocks.append("\n[NOTE: Generative AI is in template fallback mode. Set GEMINI_API_KEY to activate Google Gemini reasoning.]")
            gemini_answer = "\n".join(blocks)

        citations = [
            {"title": e.get("title", ""), "source": e.get("source", ""), "page": e.get("page", "")}
            for e in evidence if "source" in e
        ]

        return {
            "query": req.query,
            "user_role": req.user_role,
            "gemini_powered": gemini_powered,
            "answer": gemini_answer,
            "citations": citations,
            "evidence": evidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regulatory safety advisory failed: {str(e)}")
