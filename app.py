import os
import torch
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖")
st.title("🤖 دستیار RAG — پرسش از سند")
st.caption("سوال خودت را بپرس. اگر جواب در سند نبود، می‌گویم «در متن نیست».")

@st.cache_resource(show_spinner=False)
def load_pipeline():
    FILE_PATH = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.lower().endswith((".pdf", ".txt")) and f != "requirements.txt":
                FILE_PATH = os.path.join(root, f)
                break
        if FILE_PATH:
            break

    if not FILE_PATH:
        st.error("هیچ فایل PDF یا TXT پیدا نشد.")
        st.stop()

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
            chunks.append({
                "text": chunk_text,
                "source": os.path.basename(FILE_PATH),
                "page": p["page"],
            })

    embedder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    return embedder, index, chunks, tokenizer, model


def rag_query(question, embedder, index, chunks, tokenizer, model, top_k=3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    context = "\n\n".join(
        [f"[منبع: {c['source']}، بخش {c['page']}]\n{c['text']}" for c in retrieved]
    )

    messages = [
        {
            "role": "system",
            "content": "تو یک دستیار هستی که فقط بر اساس متن داده‌شده جواب می‌دهد. "
                       "اگر جواب در متن نبود، دقیقاً بگو «در متن نیست». "
                       "همیشه فارسی جواب بده و در پایان، منبع را ذکر کن.",
        },
        {
            "role": "user",
            "content": f"متن:\n{context}\n\nسوال: {question}\n\nجواب:",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
        )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    sources = list({(c["source"], c["page"]) for c in retrieved})
    return answer.strip(), sources


with st.spinner("در حال بارگذاری مدل و ساخت پایگاه برداری... (چند دقیقه)"):
    embedder, index, chunks, tokenizer, model = load_pipeline()

st.success(f"آماده است — {len(chunks)} قطعه ایندکس شد.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("سوال خودت را بنویس..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("در حال فکر کردن..."):
            answer, sources = rag_query(
                prompt, embedder, index, chunks, tokenizer, model
            )
        if "در متن نیست" in answer:
            full = f"❌ {answer}"
        else:
            src_text = "\n".join([f"📄 {s}، بخش {p}" for s, p in sources])
            full = f"{answer}\n\n---\n**منابع:**\n{src_text}"
        st.markdown(full)

    st.session_state.messages.append({"role": "assistant", "content": full})
