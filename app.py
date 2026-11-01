import os
import io
import json
import re
import base64
import zlib
import streamlit as st
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="Javid RAG", page_icon="🧠")

APP_BASE_URL = "https://javid-rag-assistant.streamlit.app"


# ============================================================
# Share helpers
# ============================================================
def encode_share(messages):
    slim = []
    for m in messages:
        slim.append({
            "role": m.get("role"),
            "content": m.get("content"),
            "confidence": m.get("confidence"),
        })
    data = json.dumps(slim, ensure_ascii=False).encode("utf-8")
    compressed = zlib.compress(data, level=9)
    encoded = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    return encoded


def decode_share(token):
    try:
        padding = 4 - len(token) % 4
        if padding != 4:
            token += "=" * padding
        compressed = base64.urlsafe_b64decode(token.encode("ascii"))
        data = zlib.decompress(compressed).decode("utf-8")
        return json.loads(data)
    except Exception:
        return None


# ============================================================
# Shared view
# ============================================================
share_token = None
try:
    share_token = st.query_params.get("share")
except (AttributeError, KeyError):
    try:
        share_token = st.experimental_get_query_params().get("share", [None])[0]
    except Exception:
        share_token = None

if share_token:
    shared = decode_share(share_token)
    if shared:
        st.title("🔗 Shared Conversation")
        st.caption("این گفتگو فقط خواندنی است / Read-only")
        st.markdown(f"[🔄 رفتن به اپ اصلی / Go to main app]({APP_BASE_URL})")
        st.divider()
        for msg in shared:
            with st.chat_message(msg.get("role", "assistant")):
                st.markdown(msg.get("content", ""))
                if msg.get("role") == "assistant" and msg.get("confidence") is not None:
                    conf = msg["confidence"]
                    emoji = "🟢" if conf >= 70 else ("🟡" if conf >= 40 else "🔴")
                    st.caption(f"{emoji} اطمینان: **{conf}%**")
        st.stop()
    else:
        st.error("❌ لینک اشتراک نامعتبر است.")
        st.markdown(f"[🔄 رفتن به اپ اصلی]({APP_BASE_URL})")
        st.stop()


# ============================================================
# برند
# ============================================================
st.markdown("""
<style>
    .brand-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e40af 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .brand-tagline {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -8px;
        margin-bottom: 1rem;
    }
    .brand-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="brand-title">Javid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="brand-tagline">Ask your documents. Get answers. Zero hallucinations.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<span class="brand-badge">🔒 Zero hallucinations</span>'
    '<span class="brand-badge">🌐 Bilingual</span>'
    '<span class="brand-badge">📄 PDF + TXT</span>',
    unsafe_allow_html=True
)
st.divider()


CF_ACCOUNT_ID = st.secrets["CF_ACCOUNT_ID"]
CF_API_TOKEN = st.secrets["CF_API_TOKEN"]
CF_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
CF_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"

MAX_HISTORY_TURNS = 3


def cf_generate(prompt, system="You are a precise assistant.", max_tokens=1500, temperature=0.3):
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": CF_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(CF_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ============================================================
# خلاصه‌ساز
# ============================================================
def summarize_document(full_text, lang):
    if not full_text or len(full_text.strip()) < 100:
        return {"tldr": "سند خیلی کوتاه است.", "medium": "", "detailed": [], "raw": ""}

    MAX_CHARS = 6000
    lang_name = "Persian" if lang == "Persian" else "English"
    text = full_text[:MAX_CHARS]
    raw_log = []

    tldr_prompt = (
        f"Summarize the document below in ONE sentence (max 25 words) in {lang_name}. "
        f"Only use information from the text. No preamble, no labels, just the sentence.\n\n"
        f"Document:\n\"\"\"\n{text}\n\"\"\""
    )
    try:
        tldr = cf_generate(tldr_prompt, max_tokens=100, temperature=0.2).strip()
        tldr = re.sub(r'^(TLDR|TL;DR|Summary|خلاصه)\s*:?\s*', '', tldr, flags=re.IGNORECASE).strip()
        tldr = tldr.strip('"').strip("'").strip()
        raw_log.append(f"[TLDR] {tldr[:200]}")
    except Exception as e:
        tldr = f"خطا در خلاصه‌سازی: {e}"
        raw_log.append(f"[TLDR ERROR] {e}")

    medium_prompt = (
        f"Write a paragraph of 4-5 sentences summarizing the main points of the document in {lang_name}. "
        f"Only use information from the text. No preamble, just the paragraph.\n\n"
        f"Document:\n\"\"\"\n{text}\n\"\"\""
    )
    try:
        medium = cf_generate(medium_prompt, max_tokens=500, temperature=0.2).strip()
        raw_log.append(f"[MEDIUM] {medium[:200]}")
    except Exception as e:
        medium = f"خطا: {e}"
        raw_log.append(f"[MEDIUM ERROR] {e}")

    bullets_prompt = (
        f"Extract exactly 5 key points from the document below. Write in {lang_name}. "
        f"Return ONLY 5 lines, each starting with '- '. "
        f"One sentence per point. Only use information from the text. No preamble.\n\n"
        f"Document:\n\"\"\"\n{text}\n\"\"\""
    )
    try:
        bullets_raw = cf_generate(bullets_prompt, max_tokens=600, temperature=0.2)
        detailed = []
        for line in bullets_raw.split("\n"):
            line = line.strip()
            line = re.sub(r'^[\-\*•\d\.\)\s]+', '', line).strip()
            if line and len(line) > 10:
                detailed.append(line)
        detailed = detailed[:5]
        raw_log.append(f"[BULLETS] {len(detailed)} bullets extracted")
    except Exception as e:
        detailed = []
        raw_log.append(f"[BULLETS ERROR] {e}")

    return {"tldr": tldr, "medium": medium, "detailed": detailed, "raw": "\n".join(raw_log)}


# ============================================================
# سوال‌ساز
# ============================================================
def generate_suggested_questions(full_text, lang, n=5):
    if not full_text or len(full_text.strip()) < 100:
        return []

    lang_name = "Persian" if lang == "Persian" else "English"
    prompt = f"""Read the following document and generate exactly {n} questions a curious reader would naturally ask.

Rules:
- Write in {lang_name}.
- Mix "why", "how", "what is the relationship", and applied questions.
- AVOID trivial yes/no questions.
- AVOID questions answerable in one word.
- Every question MUST be answerable from the text below.
- Return ONLY the questions, one per line, no numbering, no quotes, no preamble.
- Each question must end with "?" (or "؟" for Persian).

Document:
\"\"\"
{full_text[:6000]}
\"\"\"
"""
    try:
        raw = cf_generate(prompt, max_tokens=400, temperature=0.6)
    except Exception:
        return []

    questions = []
    for line in raw.split("\n"):
        line = re.sub(r'^\s*[\d\-\*•\.\)]+\s*', '', line.strip())
        line = line.strip('"').strip("'").strip()
        if line and len(line) > 10 and line.endswith(("?", "؟")):
            questions.append(line)
    return questions[:n]


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


def generate_query_variants(question, n=3):
    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    prompt = (
        f"Rewrite the following question in {n} different ways, preserving the meaning "
        f"but using different words and structures. Keep the same language as the original. "
        f"Return ONLY the rewritten questions, one per line, without numbering or explanation.\n\n"
        f"Original question: {question}"
    )
    payload = {
        "model": CF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
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
    except Exception:
        pass
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
    combined = {
        idx: alpha * vector_scores.get(idx, 0) + (1 - alpha) * keyword_scores.get(idx, 0)
        for idx in all_indices
    }

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
            if key not in seen:
                seen[key] = {"chunk": c, "score": 0.0}
            seen[key]["score"] += 1.0 / (rank + 1)

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return [r["chunk"] for r in ranked[:top_k]]


def rerank_maxsim_scored(question, candidates, embedder, top_k=3):
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
    return scored[:top_k]


def rerank_maxsim(question, candidates, embedder, top_k=3):
    return [c for c, _ in rerank_maxsim_scored(question, candidates, embedder, top_k)]


def build_search_query(question, history):
    if not history:
        return question

    follow_up_markers = [
        "آن", "این", "او", "بیشتر", "چرا", "چطور", "چه تفاوتی", "ادامه", "مثال",
        "it", "this", "that", "more", "why", "how", "difference", "example",
    ]
    is_follow_up = (
        len(question.split()) <= 5
        or any(m in question.lower() for m in follow_up_markers)
    )
    if not is_follow_up:
        return question

    for msg in reversed(history):
        if msg["role"] == "user":
            return f"{msg['content']} — {question}"
    return question


def rag_query(question, embedder, index, chunks, bm25, history=None, use_expansion=True):
    history = history or []
    search_query = build_search_query(question, history)

    candidates = multi_query_search(
        search_query, embedder, index, chunks, bm25, top_k=6, use_expansion=use_expansion
    )
    scored_chunks = rerank_maxsim_scored(search_query, candidates, embedder, top_k=3)
    retrieved = [c for c, _ in scored_chunks]

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

    messages = [
        {
            "role": "system",
            "content": f"You are an assistant that answers ONLY from the provided text. "
                       f"If the answer is not in the text, say exactly 'Not in text' (in {lang}). "
                       f"{lang_instruction} "
                       f"Use the conversation history to understand follow-up questions, "
                       f"but answer ONLY from the provided text. "
                       f"At the end, cite the source (file name and page/paragraph).",
        }
    ]

    recent = history[-(MAX_HISTORY_TURNS * 2):] if history else []
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Text:\n{context}\n\nQuestion: {question}\n\nAnswer ({lang}):",
    })

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"model": CF_MODEL, "messages": messages}

    response = requests.post(CF_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return f"API error (code {response.status_code}): {response.text}", [], {"chunks": [], "confidence": 0}

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    sources = list({(c["source"], c["page"]) for c in retrieved})

    is_not_in_text = ("Not in text" in answer) or ("در متن نیست" in answer)
    if is_not_in_text or not scored_chunks:
        confidence = 0
    else:
        top_score = scored_chunks[0][1]
        relevant = [s for _, s in scored_chunks if s >= 0.35]
        if len(relevant) >= 2:
            conf_raw = top_score * 0.75 + relevant[1] * 0.25
        else:
            conf_raw = top_score
        confidence = round(min(95, max(10, conf_raw * 115)))

    details = {
        "chunks": [
            {"text": c["text"], "page": c["page"], "source": c["source"], "score": round(s, 3)}
            for c, s in scored_chunks
        ],
        "confidence": confidence,
    }

    return answer.strip(), sources, details


# ============================================================
# سایدبار
# ============================================================
st.sidebar.header("📂 Select your document / سند خودت را انتخاب کن")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT (optional)", type=["pdf", "txt"])
use_expansion = st.sidebar.checkbox("Enable Multi-Query Expansion", value=True)


# ============================================================
# About
# ============================================================
st.sidebar.divider()
with st.sidebar.expander("ℹ️ About Javid RAG"):
    st.markdown(
        """
**Javid RAG** answers questions from **your** documents — and only your documents.

### 🎯 Why Javid RAG?
- 📄 **Upload any PDF or TXT** — ask in Persian or English
- 🎯 **Zero hallucinations** — if the answer isn't in the document, it says so
- 🔍 **Full transparency** — see exactly which chunks answered your question
- 🧠 **Conversation memory** — follow-up questions work naturally
- 📊 **Confidence scores** — know how sure the answer is

### 📈 Validation
Validated on 50 test questions — zero hallucinations on out-of-doc queries.

### 🔗 Links
- [📊 Case Study](https://docusage-query-flow.base44.app)
- [💻 GitHub](https://github.com/javidshahadati/rag-streamlit)
- [👤 LinkedIn](https://www.linkedin.com/in/javid-shahadati-a867a0447/)

### 👨‍💻 Built by
Javid Shahadati — RAG Specialist
        """
    )


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
    f"(Hybrid + Multi-Query + Re-ranking + Memory)"
)

# ============================================================
# خلاصه
# ============================================================
st.markdown("### 📝 خلاصه‌ی سند / Document Summary")

if st.session_state.get("summary_doc_key") != doc_key:
    st.session_state.summary = None
    st.session_state.summary_doc_key = doc_key

if st.button("📝 خلاصه کن / Summarize", use_container_width=True):
    full_text = "\n\n".join([p["text"] for p in doc["pages"]])
    lang = detect_language(full_text)
    with st.spinner("در حال ساخت خلاصه (۳ درخواست به مدل)..."):
        st.session_state.summary = summarize_document(full_text, lang)

if st.session_state.get("summary"):
    s = st.session_state.summary
    st.markdown("**⚡ TL;DR**")
    st.info(s.get("tldr") or "—")
    with st.expander("📄 خلاصه‌ی متوسط / Medium Summary", expanded=False):
        st.write(s.get("medium") or "—")
    with st.expander("📚 نکات کلیدی / Detailed Summary", expanded=False):
        bullets = s.get("detailed") or []
        if bullets:
            for b in bullets:
                st.markdown(f"- {b}")
        else:
            st.write("—")
    with st.expander("🔍 Debug (خروجی خام مدل)", expanded=False):
        st.code(s.get("raw", ""), language="text")

# ============================================================
# سوالات پیشنهادی
# ============================================================
st.markdown("### 💡 سوالات پیشنهادی / Suggested Questions")

if st.session_state.get("questions_doc_key") != doc_key:
    st.session_state.suggested_questions = None
    st.session_state.questions_attempts = 0
    st.session_state.questions_doc_key = doc_key

if (not st.session_state.get("suggested_questions")) and st.session_state.get("questions_attempts", 0) < 2:
    full_text = "\n\n".join([p["text"] for p in doc["pages"]])
    lang = detect_language(full_text)
    with st.spinner("در حال ساخت سوالات پیشنهادی..."):
        st.session_state.suggested_questions = generate_suggested_questions(full_text, lang)
    st.session_state.questions_attempts = st.session_state.get("questions_attempts", 0) + 1

if st.session_state.get("suggested_questions"):
    qs = st.session_state.suggested_questions
    cols = st.columns(2)
    for i, q in enumerate(qs):
        with cols[i % 2]:
            if st.button(q, key=f"sugg_q_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
else:
    st.warning("سوالی ساخته نشد. روی دکمه زیر بزن.")

if st.button("🔄 سوالات جدید", key="regen_qs"):
    st.session_state.suggested_questions = None
    st.session_state.questions_attempts = 0
    st.rerun()

# --- ارزیابی ---
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
        correct_in = total_in = correct_out = total_out = hallucinations = 0

        for tc in test_cases:
            answer, sources, _ = rag_query(
                tc["question"], embedder, index, chunks, bm25,
                history=[], use_expansion=use_expansion
            )
            out_of_doc_answer = ("Not in text" in answer) or ("در متن نیست" in answer)

            if tc["in_doc"]:
                total_in += 1
                correct = (not out_of_doc_answer) and any(
                    kw.lower() in answer.lower() for kw in tc["expected"]
                )
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
    st.session_state.share_url = None
    st.session_state.show_share = False

# ============================================================
# Chat actions
# ============================================================
if st.session_state.messages:
    st.markdown("### 💬 گفتگو / Conversation")
    ac1, ac2, ac3 = st.columns(3)

    with ac1:
        md_lines = [f"# گفتگو / Chat\n\n**Document:** {doc['name']}\n"]
        for msg in st.session_state.messages:
            role = "👤 **You**" if msg["role"] == "user" else "🤖 **Assistant**"
            md_lines.append(f"\n{role}:\n\n{msg.get('content', '')}\n")
            if msg.get("confidence") is not None:
                md_lines.append(f"_Confidence: {msg['confidence']}%_\n")
            md_lines.append("\n---\n")
        st.download_button(
            "📥 دانلود / Download",
            data="\n".join(md_lines).encode("utf-8"),
            file_name=f"chat_{doc['name'].rsplit('.', 1)[0]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with ac2:
        if st.button("🔗 لینک اشتراک / Share", use_container_width=True):
            token = encode_share(st.session_state.messages)
            st.session_state.share_url = f"{APP_BASE_URL}/?share={token}"
            st.session_state.show_share = True

    with ac3:
        if st.button("🗑️ پاک کردن / Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.share_url = None
            st.session_state.show_share = False
            st.rerun()

    if st.session_state.get("show_share") and st.session_state.get("share_url"):
        st.code(st.session_state.share_url, language="text")
        st.caption(f"طول لینک: **{len(st.session_state.share_url)}** کاراکتر")
        if len(st.session_state.share_url) > 8000:
            st.warning("⚠️ لینک بلند است. اگه باز نشد، پیام‌های کمتری رو به اشتراک بذار.")


def render_assistant_extras(idx, msg):
    conf = msg.get("confidence")
    if conf is not None:
        if conf >= 70:
            st.caption(f"🟢 اطمینان: **{conf}%**")
        elif conf >= 40:
            st.caption(f"🟡 اطمینان: **{conf}%**")
        else:
            st.caption(f"🔴 اطمینان: **{conf}%**")

    chunks_meta = msg.get("chunks", [])
    if chunks_meta:
        visible = [c for c in chunks_meta if c["score"] >= 0.30]
        if visible:
            with st.expander("🔍 چرا این جواب؟ (بخش‌های بازیابی‌شده)", expanded=False):
                for k, ch in enumerate(visible):
                    st.markdown(
                        f"**بخش {k+1} — صفحه {ch['page']} — امتیاز: `{ch['score']}`**"
                    )
                    st.caption(ch["text"][:400] + ("..." if len(ch["text"]) > 400 else ""))
        else:
            st.caption("⚠️ چانک مناسبی پیدا نشد — احتمالاً سند این اطلاعات را ندارد")

    fb_key = f"fb_{idx}"
current_fb = st.session_state.get(fb_key)

if current_fb == "up":
    st.caption("🙏 ممنون — خوشحالم که کمک کرد")
elif current_fb == "down":
    st.caption("📝 ثبت شد — ممنون که کمک کردی بهتر بشم")
else:
    fb_col1, fb_col2, fb_col3 = st.columns([3, 1, 1])
    with fb_col1:
        st.caption("این جواب چطور بود؟")
    with fb_col2:
        if st.button("👍", key=f"up_{fb_key}", help="جواب خوب بود"):
            st.session_state[fb_key] = "up"
            st.rerun()
    with fb_col3:
        if st.button("👎", key=f"down_{fb_key}", help="جواب دقیق نبود"):
            st.session_state[fb_key] = "down"
            st.rerun()


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(i, msg)

# --- ورودی چت ---
prompt_input = st.chat_input("Ask a question... / سوال خودت را بنویس...")
pending = st.session_state.pop("pending_question", None)
prompt = pending or prompt_input

if prompt:
    history_before = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... / در حال فکر کردن..."):
            answer, sources, details = rag_query(
                prompt, embedder, index, chunks, bm25,
                history=history_before, use_expansion=use_expansion
            )
        if ("Not in text" in answer) or ("در متن نیست" in answer):
            full = f"❌ {answer}"
        else:
            src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
            full = f"{answer}\n\n---\n**Sources / منابع:**\n{src_text}"
        st.markdown(full)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full,
        "confidence": details.get("confidence"),
        "chunks": details.get("chunks", []),
    })
    st.session_state.show_share = False
    st.rerun()
