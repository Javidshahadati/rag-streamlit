import os
import io
import json
import streamlit as st
import requests
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖")
st.title("🤖 RAG PDF Assistant — دستیار RAG")
st.caption("Ask in English or Persian — the assistant answers in the same language.")

# --- کلیدها ---
CF_ACCOUNT_ID = st.secrets["CF_ACCOUNT_ID"]
CF_API_TOKEN = st.secrets["CF_API_TOKEN"]
CF_MODEL = "@cf/meta/llama-3.1-8b-instruct-fast"
CF_URL = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1/chat/completions"


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
    return embedder, index, chunks


def rag_query(question, embedder, index, chunks, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    context = "\n\n".join(
        [f"[Source: {c['source']}, page/paragraph {c['page']}]\n{c['text']}" for c in retrieved]
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
                "content": "You are an assistant that answers ONLY from the provided text. "
                           "If the answer is not in the text, say exactly 'Not in text' "
                           "(in the same language as the question). "
                           "Answer in the same language as the question. "
                           "At the end, cite the source (file name and page/paragraph).",
            },
            {
                "role": "user",
                "content": f"Text:\n{context}\n\nQuestion: {question}\n\nAnswer:",
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


# --- انتخاب سند از سایدبار ---
st.sidebar.header("📂 Select your document / سند خودت را انتخاب کن")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT (optional)", type=["pdf", "txt"]
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

with st.spinner(f"Building vector index from «{doc['name']}»..."):
    embedder, index, chunks = build_index(doc_key, doc["name"], pages_json)

st.success(f"✅ Ready — {len(chunks)} chunks indexed from «{doc['name']}»")

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
            answer, sources = rag_query(prompt, embedder, index, chunks)
        if ("Not in text" in answer) or ("در متن نیست" in answer):
            full = f"❌ {answer}"
        else:
            src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
            full = f"{answer}\n\n---\n**Sources / منابع:**\n{src_text}"
        st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
