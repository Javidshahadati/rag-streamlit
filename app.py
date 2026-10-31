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
st.caption("Ask in English or Persian — the assistant remembers the conversation.")

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
# خلاصه‌ساز سه‌سطحی — نسخه‌ی فیکس‌شده (خطی، نه JSON)
# ============================================================
def summarize_document(full_text, lang):
    if not full_text or len(full_text.strip()) < 100:
        return {"tldr": "سند خیلی کوتاه است.", "medium": "", "detailed": []}

    MAX_CHARS = 6000
    lang_name = "Persian" if lang == "Persian" else "English"

    prompt = f"""Analyze the document below and respond EXACTLY in this format (in {lang_name}):

TLDR: <one sentence, max 25 words>
MEDIUM: <one paragraph, 4-5 sentences>
BULLET: <key point 1>
BULLET: <key point 2>
BULLET: <key point 3>
BULLET: <key point 4>
BULLET: <key point 5>

Rules:
- Write everything in {lang_name}.
- Only use information from the text.
- No JSON, no markdown, no code fences, no preamble before TLDR:.
- Each line starts with exactly TLDR:, MEDIUM:, or BULLET:.
- Do NOT translate markers. Keep "TLDR:", "MEDIUM:", "BULLET:" in English.

Document:
\"\"\"
{full_text[:MAX_CHARS]}
\"\"\"
"""
    try:
        raw = cf_generate(prompt, max_tokens=1200, temperature=0.2)
    except Exception as e:
        return {"tldr": f"خطا در تماس با مدل: {e}", "medium": "", "detailed": []}

    result = {"tldr": "", "medium": "", "detailed": []}
    for line in raw.split("\n"):
        line = line.strip()
        up = line.upper()
        if up.startswith("TLDR:") or up.startswith("TL;DR:"):
            result["tldr"] = line.split(":", 1)[1].strip()
        elif up.startswith("MEDIUM:"):
            result["medium"] = line.split(":", 1)[1].strip()
        elif up.startswith("BULLET:"):
            b = line.split(":", 1)[1].strip()
            if b:
                result["detailed"].append(b)
        elif up.startswith(("-", "•", "*")) and len(result["detailed"]) < 5:
            b = line.lstrip("-•* ").strip()
            if b:
                result["detailed"].append(b)

    if not result["tldr"] and not result["medium"] and not result["detailed"]:
        result["tldr"] = raw[:250]

    return result


# ============================================================
# سوال‌ساز از سند
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
    retrieved = rerank_maxsim(search_query, candidates, embedder, top_k=3)

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
        return f"API error (code {response.status_code}): {response.text}", []

    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    sources = list({(c["source"], c["page"]) for c in retrieved})
    return answer.strip(), sources


# --- سایدبار ---
st.sidebar.header("📂 Select your document / سند خودت را انتخاب کن")
uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT (optional)", type=["pdf", "txt"])
use_expansion = st.sidebar.checkbox("Enable Multi-Query Expansion", value=True)

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
# بخش خلاصه‌ی سند
# ============================================================
st.markdown("### 📝 خلاصه‌ی سند / Document Summary")

if st.session_state.get("summary_doc_key") != doc_key:
    st.session_state.summary = None
    st.session_state.summary_doc_key = doc_key

if st.button("📝 خلاصه کن / Summarize", use_container_width=True):
    full_text = "\n\n".join([p["text"] for p in doc["pages"]])
    lang = detect_language(full_text)
    with st.spinner("در حال خواندن سند... / Reading document..."):
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

# ============================================================
# سوالات پیشنهادی — نسخه‌ی فیکس‌شده (retry + دکمه‌ی همیشه فعال)
# ============================================================
st.markdown("### 💡 سوالات پیشنهادی / Suggested Questions")

if st.session_state.get("questions_doc_key") != doc_key:
    st.session_state.suggested_questions = None
    st.session_state.questions_attempts = 0
    st.session_state.questions_doc_key = doc_key

# تلاش خودکار حداکثر ۲ بار
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

# این دکمه همیشه نمایش داده می‌شه (حتی اگه لیست خالی باشه)
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
            answer, sources = rag_query(
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
            answer, sources = rag_query(
                prompt, embedder, index, chunks, bm25,
                history=history_before, use_expansion=use_expansion
            )
        if ("Not in text" in answer) or ("در متن نیست" in answer):
            full = f"❌ {answer}"
        else:
            src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
            full = f"{answer}\n\n---\n**Sources / منابع:**\n{src_text}"
        st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
