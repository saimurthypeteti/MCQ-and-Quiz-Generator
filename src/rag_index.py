"""Persistent per-document FAISS indexes with an explicit keyword-only fallback."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STOP = set("the a an and or but is are was were be been being to of in on at by for with from as into that this these those it its their they them he she we you can may will should must has have had not than then which who what when where how also such".split())
_encoder = None
_encoder_name = None


def _model(name=None):
    global _encoder, _encoder_name
    name = name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if _encoder is None or _encoder_name != name:
        from sentence_transformers import SentenceTransformer
        local_only = os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}
        _encoder = SentenceTransformer(name, local_files_only=local_only)
        _encoder_name = name
    return _encoder


def split_chunks(sections, max_chars=850, overlap_sentences=1):
    """Split section text into bounded overlapping chunks with source metadata."""
    chunks=[]
    for section_index, section in enumerate(sections):
        raw=section.get("text","")
        sentences=[s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",raw) if len(s.split()) >= 4]
        group=[]; size=0; chunk_no=0
        for sentence in sentences:
            if group and size+len(sentence)>max_chars:
                chunks.append({"chunk_id":str(uuid.uuid4()),"text":" ".join(group),"section":section.get("label",f"Section {section_index+1}"),"page":section.get("page"),"chunk_index":len(chunks)})
                group=group[-overlap_sentences:] if overlap_sentences else []; size=sum(map(len,group))+max(0,len(group)-1); chunk_no+=1
            group.append(sentence); size+=len(sentence)+int(size>0)
        if group:chunks.append({"chunk_id":str(uuid.uuid4()),"text":" ".join(group),"section":section.get("label",f"Section {section_index+1}"),"page":section.get("page"),"chunk_index":len(chunks)})
    return chunks


def ensure_schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS document_chunks(
        chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL, section TEXT, page INTEGER, created_at TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS rag_indexes(
        document_id TEXT PRIMARY KEY, vector_path TEXT, model_name TEXT, dimension INTEGER,
        status TEXT NOT NULL, chunk_count INTEGER NOT NULL, updated_at TEXT NOT NULL)""")


def _tokenize(text):
    return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if w.lower() not in STOP}


def index_document(document_id, sections, db_path, index_dir):
    """Persist chunks and, when dependencies/checkpoint are available, a FAISS inner-product index."""
    chunks=split_chunks(sections)
    if not chunks: raise ValueError("No indexable text chunks were found in the document.")
    db_path=Path(db_path); index_dir=Path(index_dir); index_dir.mkdir(parents=True,exist_ok=True)
    stamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    vector_path=index_dir/f"{document_id}.faiss"; temp_path=index_dir/f"{document_id}.tmp.faiss"
    status="keyword_only"; model_name=os.getenv("EMBEDDING_MODEL",DEFAULT_EMBEDDING_MODEL); dimension=None
    try:
        import faiss
        import numpy as np
        vectors=_model(model_name).encode([c["text"] for c in chunks],normalize_embeddings=True,convert_to_numpy=True)
        vectors=np.asarray(vectors,dtype="float32"); dimension=int(vectors.shape[1])
        index=faiss.IndexFlatIP(dimension); index.add(vectors); faiss.write_index(index,str(temp_path)); os.replace(temp_path,vector_path); status="faiss_semantic"
    except Exception:
        if temp_path.exists():temp_path.unlink(missing_ok=True)
        vector_path.unlink(missing_ok=True); model_name=None
    conn=sqlite3.connect(db_path)
    try:
        ensure_schema(conn); conn.execute("DELETE FROM document_chunks WHERE document_id=?",(document_id,))
        conn.executemany("INSERT INTO document_chunks VALUES(?,?,?,?,?,?,?)",[(c["chunk_id"],document_id,c["chunk_index"],c["text"],c["section"],c["page"],stamp) for c in chunks])
        conn.execute("INSERT INTO rag_indexes VALUES(?,?,?,?,?,?,?) ON CONFLICT(document_id) DO UPDATE SET vector_path=excluded.vector_path,model_name=excluded.model_name,dimension=excluded.dimension,status=excluded.status,chunk_count=excluded.chunk_count,updated_at=excluded.updated_at",(document_id,str(vector_path) if status=="faiss_semantic" else None,model_name,dimension,status,len(chunks),stamp));conn.commit()
    finally:conn.close()
    return {"status":status,"chunks":len(chunks),"model":model_name,"vector_path":str(vector_path) if status=="faiss_semantic" else None}


def retrieve(document_id, query, db_path, top_k=5):
    db_path=Path(db_path)
    conn=sqlite3.connect(db_path)
    try:
        conn.row_factory=sqlite3.Row; ensure_schema(conn)
        rows=conn.execute("SELECT c.*,r.status,r.vector_path,r.model_name FROM document_chunks c JOIN rag_indexes r ON r.document_id=c.document_id WHERE c.document_id=? ORDER BY c.chunk_index",(document_id,)).fetchall()
    finally:conn.close()
    if not rows:return [],{"method":"none","query":query,"chunks_found":0}
    meta=rows[0]; scored=[]; method="lexical"
    if meta["status"]=="faiss_semantic" and meta["vector_path"] and Path(meta["vector_path"]).exists():
        try:
            import faiss
            import numpy as np
            vector=_model(meta["model_name"]).encode([query],normalize_embeddings=True,convert_to_numpy=True).astype("float32")
            index=faiss.read_index(meta["vector_path"]); scores,ids=index.search(vector,min(top_k,len(rows)))
            scored=[(float(score),rows[int(i)]) for score,i in zip(scores[0],ids[0]) if 0<=int(i)<len(rows)]
            method="faiss_semantic:"+meta["model_name"]
        except Exception:
            scored=[]
    if not scored:
        q=_tokenize(query)
        for row in rows:
            terms=_tokenize(row["text"]); score=len(q&terms)/len(q|terms) if q and terms else 0.0
            scored.append((score,row))
        scored.sort(key=lambda item:item[0],reverse=True)
        method="keyword_overlap"
    threshold=float(os.getenv("RAG_MIN_SCORE","0.05"))
    selected=[(score,row) for score,row in scored[:max(1,top_k)] if score>=threshold]
    output=[{"chunk_id":r["chunk_id"],"text":r["text"],"section":r["section"],"page":r["page"],"score":score} for score,r in selected]
    return output,{"method":method,"query":query,"chunks_found":len(output),"minimum_score":threshold,"scores":[round(score,4) for score,_ in selected]}


def index_statuses(db_path):
    conn=sqlite3.connect(db_path)
    try:
        conn.row_factory=sqlite3.Row; ensure_schema(conn)
        return [dict(row) for row in conn.execute("SELECT document_id,model_name,dimension,status,chunk_count,updated_at FROM rag_indexes ORDER BY updated_at DESC")]
    finally:conn.close()
