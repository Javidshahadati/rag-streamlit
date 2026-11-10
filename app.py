import os
import io
import re
import json
import requests
import numpy as np
import faiss
import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ============================================================
# تنظیمات کلادفلر (دقیقاً مثل قبل)
# ============================================================
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
CF_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
CF_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"

MAX_HISTORY_TURNS = 3

def cf_generate(prompt, system="You are a precise assistant.", max_tokens=1500, temperature=0.3):
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    messages = []
    if system: messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": CF_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    response = requests.post(CF_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ============================================================
# توابع پردازش سند (کپی عین کدهای خودت)
# ============================================================
def detect_language(text):
    persian_chars = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
    return "Persian" if any(c in persian_chars for c in text) else "English"

def extract_text_from_upload(file):
    content = file.read()
    name = file.name.lower()
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
    return {"name": file.name, "pages": pages}

def split_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk: chunks.append(chunk)
    return chunks

def build_index(doc):
    chunks = []
    for p in doc["pages"]:
        for chunk_text in split_text(p["text"]):
            chunks.append({"text": chunk_text, "source": doc["name"], "page": p["page"]})

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    bm25 = BM25Okapi([c["text"].split() for c in chunks])
    return embedder, index, chunks, bm25

# ============================================================
# توابع جستجو (کپی عین کدهای خودت)
# ============================================================
def generate_query_variants(question, n=3):
    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    prompt = f"Rewrite the following question in {n} different ways...\nOriginal: {question}"
    payload = {"model": CF_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.2}
    try:
        response = requests.post(CF_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            text = response.json()["choices"][0]["message"]["content"]
            variants = []
            for line in text.split("\n"):
                line = re.sub(r"^\d+[\.\)\-]\s*", "", line.strip())
                if line and len(line) > 5 and line.lower() != question.lower():
                    variants.append(line)
            return variants[:n]
    except: pass
    return []

def hybrid_search(question, embedder, index, chunks, bm25, top_k=6, alpha=0.5):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    vector_scores = {int(idx): 1 / (rank + 1) for rank, idx in enumerate(indices[0])}

    tokenized_query = question.split()
    bm25_scores_arr = bm25.get_scores(tokenized_query)
    bm25_top = bm25_scores_arr.argsort()[::-1][:top_k]
    keyword_scores = {int(idx): 1 / (rank + 1) for rank, idx in enumerate(bm25_top)}

    all_indices = set(vector_scores) | set(keyword_scores)
    combined = {idx: alpha * vector_scores.get(idx, 0) + (1 - alpha) * keyword_scores.get(idx, 0) for idx in all_indices}
    sorted_idx = sorted(combined, key=combined.get, reverse=True)[:top_k]
    return [chunks[i] for i in sorted_idx]

def multi_query_search(question, embedder, index, chunks, bm25, top_k=6, use_expansion=True):
    queries = [question]
    if use_expansion:
        variants = generate_query_variants(question, n=3)
        queries.extend(variants)

    seen = {}
    for q in queries:
        results = hybrid_search(q, embedder, index, chunks, bm25, top_k=top_k)
        for rank, c in enumerate(results):
            key = (c["source"], c["page"], c["text"][:60])
            if key not in seen: seen[key] = {"chunk": c, "score": 0.0}
            seen[key]["score"] += 1.0 / (rank + 1)

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return [r["chunk"] for r in ranked[:top_k]]

def rerank_maxsim_scored(question, candidates, embedder, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-8
    scored = []
    for c in candidates:
        sentences = [s.strip() for s in re.split(r"[.?!؟!]", c["text"]) if len(s.strip()) > 10]
        if not sentences: sentences = [c["text"]]
        sent_embs = embedder.encode(sentences, convert_to_numpy=True)
        sent_norms = np.linalg.norm(sent_embs, axis=1) + 1e-8
        sims = np.dot(sent_embs, q_emb) / (sent_norms * q_norm)
        scored.append((c, float(np.max(sims))))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

def build_search_query(question, history):
    if not history: return question
    follow_up_markers = ["آن", "این", "او", "بیشتر", "چرا", "چطور", "چه تفاوتی", "ادامه", "مثال", "it", "this", "that", "more", "why", "how", "difference", "example"]
    is_follow_up = (len(question.split()) <= 5 or any(m in question.lower() for m in follow_up_markers))
    if not is_follow_up: return question
    prev = [m["content"] for m in reversed(history) if m["role"] == "user"][:2]
    if prev: return " — ".join(reversed(prev)) + " — " + question
    return question

def rag_query(question, embedder, index, chunks, bm25, history=None, use_expansion=True):
    history = history or []
    search_query = build_search_query(question, history)
    candidates = multi_query_search(search_query, embedder, index, chunks, bm25, top_k=6, use_expansion=use_expansion)
    scored_chunks = rerank_maxsim_scored(search_query, candidates, embedder, top_k=3)
    retrieved = [c for c, _ in scored_chunks]

    context = "\n\n".join([f"[Source: {c['source']}, page/paragraph {c['page']}]\n{c['text']}" for c in retrieved])
    lang = detect_language(question)
    lang_instruction = "The user's question is in English. You MUST answer in English, even if the retrieved text is in Persian." if lang == "English" else "The user's question is in Persian. Answer in Persian."

    messages = [
        {"role": "system", "content": f"You are an assistant that answers ONLY from the provided text. If the answer is not in the text, say exactly 'Not in text' (in {lang}). {lang_instruction} Use the conversation history to understand follow-up questions, but answer ONLY from the provided text. At the end, cite the source."}
    ]
    recent = history[-(MAX_HISTORY_TURNS * 2):] if history else []
    for msg in recent: messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Text:\n{context}\n\nQuestion: {question}\n\nAnswer ({lang}):"})

    headers = {"Authorization": f"Bearer {CF_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"model": CF_MODEL, "messages": messages}

    response = requests.post(CF_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return f"API error (code {response.status_code}): {response.text}", []

    answer = response.json()["choices"][0]["message"]["content"]
    sources = list({(c["source"], c["page"]) for c in retrieved})

    is_not_in_text = ("Not in text" in answer) or ("در متن نیست" in answer)
    confidence = 0 if is_not_in_text or not scored_chunks else round(min(95, max(10, scored_chunks[0][1] * 140)))

    return answer.strip(), sources

def summarize_document(full_text, lang):
    if not full_text or len(full_text.strip()) < 100:
        return "سند خیلی کوتاه است."
    MAX_CHARS = 6000
    lang_name = "Persian" if lang == "Persian" else "English"
    text = full_text[:MAX_CHARS]
    prompt = f"Summarize the document below in ONE sentence (max 25 words) in {lang_name}. Only use information from the text.\n\nDocument:\n\"\"\"\n{text}\n\"\"\""
    try:
        return cf_generate(prompt, max_tokens=100, temperature=0.2).strip()
    except Exception as e:
        return f"خطا در خلاصه‌سازی: {e}"

def generate_suggested_questions(full_text, lang, n=5):
    if not full_text or len(full_text.strip()) < 100: return []
    lang_name = "Persian" if lang == "Persian" else "English"
    prompt = f"Read the following document and generate exactly {n} questions a curious reader would naturally ask. Write in {lang_name}. Return ONLY the questions, one per line, no numbering.\n\nDocument:\n\"\"\"\n{full_text[:6000]}\n\"\"\""
    try:
        raw = cf_generate(prompt, max_tokens=400, temperature=0.6)
        questions = [re.sub(r'^\s*[\d\-\*•\.\)]+\s*', '', line.strip()) for line in raw.split("\n") if line.strip().endswith(("?", "؟"))]
        return questions[:n]
    except: return []


# ============================================================
# گرادیو اینترفیس (UI)
# ============================================================
def process_upload(file):
    if file is None: return "فایلی آپلود نشد.", None, None, None, gr.update()
    doc = extract_text_from_upload(file)
    if not doc["pages"]:
        return "خطا در خواندن فایل.", None, None, None, gr.update()
    
    embedder, index, chunks, bm25 = build_index(doc)
    full_text = "\n\n".join([p["text"] for p in doc["pages"]])
    lang = detect_language(full_text)
    
    summary = summarize_document(full_text, lang)
    qs = generate_suggested_questions(full_text, lang)
    
    state_data = {"embedder": embedder, "index": index, "chunks": chunks, "bm25": bm25, "history": []}
    
    # فرمت کردن سوالات پیشنهادی
    qs_text = "\n".join([f"- {q}" for q in qs]) if qs else "سوالی پیشنهاد نشد."
    
    return f"✅ سند «{doc['name']}» با {len(chunks)} تکه آماده شد.", summary, qs_text, state_data, gr.update(interactive=True)

def chat_fn(message, history, state_data):
    if not state_data or "index" not in state_data:
        return history + [[message, "لطفاً ابتدا یک سند آپلود کنید."]], state_data
    
    answer, sources = rag_query(
        message, state_data["embedder"], state_data["index"], state_data["chunks"], state_data["bm25"],
        history=[{"role": "user" if i%2==0 else "assistant", "content": h[0] if i%2==0 else h[1]} for i, h in enumerate(history)],
        use_expansion=True
    )
    
    if "Not in text" in answer or "در متن نیست" in answer:
        full_answer = f"❌ {answer}"
    else:
        src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
        full_answer = f"{answer}\n\n---\n**Sources / منابع:**\n{src_text}"
        
    history.append([message, full_answer])
    return history, state_data

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=".gradio-container { font-family: 'Vazirmatn', sans-serif; }") as demo:
    gr.Markdown("# 🧠 Javid RAG")
    gr.Markdown("Ask your documents. Get answers. Zero hallucinations. (Powered by Cloudflare & Faiss)")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="📂 آپلود سند (PDF/TXT)", file_types=[".pdf", ".txt"])
            upload_btn = gr.Button("🚀 پردازش سند", variant="primary")
            status_text = gr.Textbox(label="وضعیت", interactive=False)
            summary_text = gr.Textbox(label="📝 خلاصه سند", interactive=False, lines=3)
            questions_text = gr.Textbox(label="💡 سوالات پیشنهادی", interactive=False, lines=5)
            state = gr.State()
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="💬 گفتگو", height=600, rtl=True)
            msg = gr.Textbox(label="سوال خود را بنویسید...", placeholder="مثلاً: هدف اصلی این قرارداد چیست؟", interactive=False)
            clear = gr.Button("🗑️ پاک کردن گفتگو")
            
    upload_btn.click(process_upload, inputs=[file_input], outputs=[status_text, summary_text, questions_text, state, msg])
    msg.submit(chat_fn, inputs=[msg, chatbot, state], outputs=[chatbot, state])
    msg.submit(lambda: "", outputs=msg)
    clear.click(lambda: (None, []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()
