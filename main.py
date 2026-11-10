import os
import io
import re
import json
import requests
import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

app = FastAPI()

# اجازه دادن به Next.js برای ارتباط
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN")
CF_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
CF_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"

# متغیرهای گلوبال برای نگه‌داشتن ایندکس در حافظه سرور
global_index = None
global_chunks = None
global_bm25 = None
global_embedder = None

def cf_generate(prompt, system="You are a precise assistant.", max_tokens=1500, temperature=0.3):
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    messages = []
    if system: messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": CF_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    response = requests.post(CF_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def detect_language(text):
    persian_chars = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
    return "Persian" if any(c in persian_chars for c in text) else "English"

# توابع جستجو و پردازش (عینا از کد خودت)
def split_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk: chunks.append(chunk)
    return chunks

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global global_index, global_chunks, global_bm25, global_embedder
    content = await file.read()
    name = file.filename.lower()
    
    pages = []
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text.strip()})
    elif name.endswith(".txt"):
        text = content.decode("utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        pages = [{"page": i + 1, "text": p} for i, p in enumerate(paragraphs)]
    else:
        raise HTTPException(status_code=400, detail="Only PDF and TXT are supported.")

    chunks = []
    for p in pages:
        for chunk_text in split_text(p["text"]):
            chunks.append({"text": chunk_text, "source": file.filename, "page": p["page"]})

    global_embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = global_embedder.encode(chunk_texts, convert_to_numpy=True)

    global_index = faiss.IndexFlatL2(embeddings.shape[1])
    global_index.add(embeddings)
    global_chunks = chunks
    global_bm25 = BM25Okapi([c["text"].split() for c in chunks])

    return {"message": f"Index built with {len(chunks)} chunks.", "filename": file.filename}

@app.post("/ask")
async def ask_question(question: str = Form(...), history: str = Form("[]")):
    if global_index is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet.")
    
    hist = json.loads(history)
    
    # ۱. گسترش کوئری (Multi-Query)
    variants = [question]
    try:
        rewrite_prompt = f"Rewrite the following question in 3 different ways...\nOriginal: {question}"
        raw_variants = cf_generate(rewrite_prompt, max_tokens=200, temperature=0.2)
        for line in raw_variants.split("\n"):
            line = re.sub(r"^\d+[\.\)\-]\s*", "", line.strip())
            if line and len(line) > 5: variants.append(line)
    except: pass

    # ۲. جستجوی ترکیبی (Hybrid)
    all_results = {}
    for q in variants:
        # Faiss
        q_emb = global_embedder.encode([q], convert_to_numpy=True)
        distances, indices = global_index.search(q_emb, 6)
        for rank, idx in enumerate(indices[0]):
            all_results[idx] = all_results.get(idx, 0) + 1.0 / (rank + 1)
        
        # BM25
        bm25_scores = global_bm25.get_scores(q.split())
        top_bm25 = bm25_scores.argsort()[::-1][:6]
        for rank, idx in enumerate(top_bm25):
            all_results[idx] = all_results.get(idx, 0) + 1.0 / (rank + 1)

    # ۳. انتخاب بهترین‌ها
    top_indices = sorted(all_results, key=all_results.get, reverse=True)[:3]
    retrieved = [global_chunks[i] for i in top_indices]
    context = "\n\n".join([f"[Source: {c['source']}, page {c['page']}]\n{c['text']}" for c in retrieved])

    # ۴. تولید جواب
    lang = detect_language(question)
    messages = [
        {"role": "system", "content": f"You are a precise assistant. Answer ONLY from the text. If not found, say 'Not in text' ({lang}). Answer in {lang}."},
        *hist[-6:],
        {"role": "user", "content": f"Text:\n{context}\n\nQuestion: {question}"}
    ]
    
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": CF_MODEL, "messages": messages}
    response = requests.post(CF_URL, headers=headers, json=payload, timeout=60)
    answer = response.json()["choices"][0]["message"]["content"]

    # ۵. محاسبه اطمینان
    confidence = 0
    if "Not in text" not in answer and "در متن نیست" not in answer and retrieved:
        # محاسبه ساده شباهت
        q_emb = global_embedder.encode([question], convert_to_numpy=True)[0]
        chunk_embs = global_embedder.encode([c["text"] for c in retrieved], convert_to_numpy=True)
        sims = np.dot(chunk_embs, q_emb) / (np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8)
        top_sim = float(np.max(sims))
        confidence = round(min(95, max(10, top_sim * 140)))

    return {"answer": answer, "confidence": confidence, "sources": [f"{c['source']} p.{c['page']}" for c in retrieved]}
