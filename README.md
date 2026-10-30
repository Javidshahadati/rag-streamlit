# RAG PDF Assistant

**Stop hallucinating. Get answers only from your documents, with sources.**

## 🔗 Live Demo
**👉 [javid-rag-assistant.streamlit.app](https://javid-rag-assistant.streamlit.app)**

Upload your own PDF or TXT and ask questions — the assistant answers only from the document, cites the source, and says "not in text" when the answer isn't there.

## 📌 توضیح
یک سیستم RAG کامل که هر کاربر می‌تواند **سند خودش (PDF یا TXT)** را آپلود کند و از آن سوال بپرسد. جواب همراه با منبع (نام فایل و شماره صفحه) و در صورت نبود جواب، «در متن نیست». **دوزبانه** (فارسی و انگلیسی).

## 🎯 نتیجه
- **دقت روی سوالات در متن:** ۱۰۰٪
- **دقت روی سوالات خارج از متن:** ۱۰۰٪
- **تعداد توهم:** ۰
- **پشتیبانی از PDF و TXT**
- **آپلود سند توسط کاربر**
- **دوزبانه (فارسی + انگلیسی)**

## 🛠️ تکنولوژی
- Streamlit (رابط کاربری)
- Sentence Transformers (Embedding چندزبانه)
- FAISS (پایگاه برداری)
- Cloudflare Workers AI (Llama 3.1 8B)
- pypdf (استخراج متن از PDF)
- Python

## 🧠 نحوه‌ی کار
۱. کاربر فایل PDF یا TXT خودش را آپلود می‌کند.
۲. متن هر صفحه استخراج می‌شود.
۳. هر صفحه به قطعه‌های کوچک تقسیم می‌شود.
۴. هر قطعه به بردار تبدیل می‌شود.
۵. بردارها در FAISS ذخیره می‌شوند.
۶. سوال کاربر جستجو می‌شود.
۷. نزدیک‌ترین قطعه‌ها به مدل زبانی داده می‌شود.
۸. مدل فقط بر اساس متن جواب می‌دهد + منبع.

## 📊 نتایج تست (فارسی)
| سند | سوال | نتیجه |
|-----|-------|--------|
| ai-text.txt | Transformer چیست؟ | ✅ جواب درست + منبع |
| ai-text.txt | پایتخت فرانسه کجاست؟ | ✅ «در متن نیست» (۰ توهم) |
| مقاله GPR (۲۱ قطعه) | موضوع این مقاله چیست؟ | ✅ جواب درست + منبع |

## 📊 English test
| Document | Question | Result |
|----------|----------|--------|
| ai-text.txt | What is Transformer? | ✅ Correct answer + source |
| ai-text.txt | What is the capital of France? | ✅ "Not in text" (0 hallucination) |
| GPR research paper (21 chunks) | What is this paper about? | ✅ Correct answer + source |

## 📝 درس‌های گرفته‌شده
- استفاده از API به جای مدل محلی، محدودیت RAM را حل کرد.
- Streamlit Secrets برای امنیت کلیدها ضروری است.
- مدل‌های Cloudflare گاهی منقضی می‌شوند و باید به‌روز شوند.
- استخراج متن از PDF فارسی گاهی به‌هم‌ریخته است — برای متن‌های مهم، TXT بهتر است.
- مدل Llama 3.1 چندزبانه است و به هر زبانی که سوال بپرسی، همان زبان جواب می‌دهد.

## 👤 نویسنده
Javid Shahadati
