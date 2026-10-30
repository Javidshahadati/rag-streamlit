import os
import io
import json
import re
import streamlit as st
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖")
st.title("🤖 RAG PDF Assistant — دستیار RAG")
st.caption("Ask in English or Persian — the assistant answers in the same language.")

CF_ACCOUNT_ID = st.secrets["CF_ACCOUNT_ID"]
CF_API_TOKEN = st.secrets["CF_API_TOKEN"]
CF_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
CF_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"


def detect_language(text):
    persian_chars = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
    return "Persian" if any(c in persian_chars for c in text) else "English"


def extract_text_from_upload(uploaded_file):
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text.strip()})
        return {"name": uploaded_file.name, "pages": pages}
    elif name.endswith(".txt"):
        text = content.decode("utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        pages = [{"page": i + 1, "text": p} for i, p in enumerate(paragraphs)]
        return {"name": uploaded_file.name, "pages": pages}
    return None


def extract_text_from_default():
    FILE_PATH = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.lower().endswith((".pdf", ".txt")) and f != "requirements.txt":
                FILE_PATH = os.path.join(root, f)
                break
        if FILE_PATH:
            break
    if not FILE_PATH:
        return None
    if FILE_PATH.lower().endswith(".pdf"):
        reader = PdfReader(FILE_PATH)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i, "text": text.strip()})
    else:
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            full_text = f.read()
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        pages = [{"page": i + 1, "text": p} for i, p in enumerate(paragraphs)]
    return {"name": os.path.basename(FILE_PATH), "pages": pages}


@st.cache_resource(show_spinner=False)
def build_index(doc_key, doc_name, pages_json):
    pages = json.loads(pages_json)

    def split_text(text, chunk_size=400, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    chunks = []
    for p in pages:
        for chunk_text in split_text(p["text"]):
            chunks.append({"text": chunk_text, "source": doc_name, "page": p["page"]})

    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    tokenized_chunks = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return embedder, index, chunks, bm25


def hybrid_search(question, embedder, index, chunks, bm25, top_k=6, alpha=0.5):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    vector_scores = {int(idx): 1 / (rank + 1) for rank, idx in enumerate(indices[0])}

    tokenized_query = question.split()
    bm25_scores_arr = bm25.get_scores(tokenized_query)
    bm25_top = bm25_scores_arr.argsort()[::-1][:top_k]
    keyword_scores = {int(idx): 1 / (rank + 1) for rank, idx in enumerate(bm25_top)}

    all_indices = set(vector_scores) | set(keyword_scores)
    combined = {
        idx: alpha * vector_scores.get(idx, 0) + (1 - alpha) * keyword_scores.get(idx, 0)
        for idx in all_indices
    }

    sorted_idx = sorted(combined, key=combined.get, reverse=True)[:top_k]
    return [chunks[i] for i in sorted_idx]


def rerank_maxsim(question, candidates, embedder, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-8

    scored = []
    for c in candidates:
        sentences = [s.strip() for s in re.split(r"[.?!؟!]", c["text"]) if len(s.strip()) > 10]
        if not sentences:
            sentences = [c["text"]]

        sent_embs = embedder.encode(sentences, convert_to_numpy=True)
        sent_norms = np.linalg.norm(sent_embs, axis=1) + 1e-8
        sims = np.dot(sent_embs, q_emb) / (sent_norms * q_norm)
        max_sim = float(np.max(sims))
        scored.append((c, max_sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


def rag_query(question, embedder, index, chunks, bm25):
    candidates = hybrid_search(question, embedder, index, chunks, bm25, top_k=6)
    retrieved = rerank_maxsim(question, candidates, embedder, top_k=3)

    context = "\n\n".join(
        [f"[Source: {c['source']}, page/paragraph {c['page']}]\n{c['text']}" for c in retrieved]
    )

    lang = detect_language(question)
    lang_instruction = (
        "The user's question is in English. You MUST answer in English, "
        "even if the retrieved text is in Persian. Translate the answer to English."
        if lang == "English"
        else "The user's question is in Persian. Answer in Persian."
    )

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": f"You are an assistant that answers ONLY from the provided text. "
                           f"If the answer is not in the text, say exactly 'Not in text' (in {lang}). "
                           f"{lang_instruction} "
                           f"At the end, cite the source (file name and page/paragraph).",
            },
            {
                "role": "user",
                "content": f"Text:\n{context}\n\nQuestion: {question}\n\nAnswer ({lang}):",
            },
        ],
    }
    response = requests.post(CF_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return f"API error (code {response.status_code}): {response.text}", []

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    sources = list({(c["source"], c["page"]) for c in retrieved})
    return answer.strip(), sources


st.sidebar.header("📂 Select your document / سند خودت را انتخاب کن")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT (optional)", type=["pdf", "txt"])

if uploaded_file is not None:
    doc = extract_text_from_upload(uploaded_file)
    doc_key = f"upload_{uploaded_file.name}_{uploaded_file.size}"
else:
    doc = extract_text_from_default()
    doc_key = "default"

if doc is None or not doc["pages"]:
    st.error("No valid document found. / هیچ سند معتبری یافت نشد.")
    st.stop()

pages_json = json.dumps(doc["pages"], ensure_ascii=False)

with st.spinner(f"Building index from «{doc['name']}»..."):
    embedder, index, chunks, bm25 = build_index(doc_key, doc["name"], pages_json)

st.success(
    f"✅ Ready — {len(chunks)} chunks from «{doc['name']}» "
    f"(Hybrid Search + Re-ranking enabled)"
)

# --- بخش ارزیابی ---
with st.expander("📊 Run Evaluation (in-doc & out-of-doc tests)"):
    if st.button("Run Evaluation"):
        test_cases = [
            {"question": "آیا در متن به Transformer اشاره شده؟", "expected": ["Transformer", "2017"], "in_doc": True},
            {"question": "BERT چه کاری انجام می‌دهد؟", "expected": ["درک متن", "دوطرفه"], "in_doc": True},
            {"question": "تفاوت CNN و RNN چیست؟", "expected": ["تصویر", "ترتیبی"], "in_doc": True},
            {"question": "پایتخت فرانسه کجاست؟", "expected": [], "in_doc": False},
            {"question": "What is the capital of France?", "expected": [], "in_doc": False},
            {"question": "What is Transformer?", "expected": ["Transformer", "2017"], "in_doc": True},
        ]
        results = []
        correct_in = 0
        total_in = 0
        correct_out = 0
        total_out = 0
        hallucinations = 0

        for tc in test_cases:
            answer, sources = rag_query(tc["question"], embedder, index, chunks, bm25)
            out_of_doc_answer = ("Not in text" in answer) or ("در متن نیست" in answer)

            if tc["in_doc"]:
                total_in += 1
                correct = (not out_of_doc_answer) and any(kw.lower() in answer.lower() for kw in tc["expected"])
                if correct:
                    correct_in += 1
            else:
                total_out += 1
                if out_of_doc_answer:
                    correct_out += 1
                else:
                    hallucinations += 1

            results.append({
                "Question": tc["question"],
                "Answer": (answer[:80] + "...") if len(answer) > 80 else answer,
                "Sources": ", ".join([f"{s[0]} p.{s[1]}" for s in sources]) if sources else "-",
                "Correct": correct if tc["in_doc"] else out_of_doc_answer,
            })

        in_acc = (correct_in / total_in * 100) if total_in else 0
        out_acc = (correct_out / total_out * 100) if total_out else 0
        hall_rate = (hallucinations / total_out * 100) if total_out else 0

        st.write(f"**In-document accuracy:** {in_acc:.1f}%")
        st.write(f"**Out-of-document accuracy:** {out_acc:.1f}%")
        st.write(f"**Hallucination rate:** {hall_rate:.1f}%")
        st.dataframe(results, use_container_width=True)

# --- حافظه چت ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.session_state.get("current_doc_key") != doc_key:
    st.session_state.messages = []
    st.session_state.current_doc_key = doc_key

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question... / سوال خودت را بنویس..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... / در حال فکر کردن..."):
            answer, sources = rag_query(prompt, embedder, index, chunks, bm25)
        if ("Not in text" in answer) or ("در متن نیست" in answer):
            full = f"❌ {answer}"
        else:
            src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
            full = f"{answer}\n\n---\n**Sources / منابع:**\n{src_text}"
        st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
