# 📄 Multi-PDF RAG Chatbot

> Upload multiple PDFs, ask questions, and get precise answers powered by **LLaMA 3.3** + **Google Gemini Embeddings** — all running in a clean Streamlit interface.

---

## 🧠 How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding (Gemini) → FAISS Index
                                                                      ↓
                                             User Question → Similarity Search
                                                                      ↓
                                             Context + Prompt → LLaMA 3.3 (Groq) → Answer
```

1. **Upload** one or more PDF files via the sidebar
2. **Process** — text is extracted, split into chunks, and embedded into a local FAISS vector store
3. **Ask** — your question is matched against the most relevant chunks
4. **Get answers** — LLaMA 3.3 (via Groq) generates a detailed, context-grounded response

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| 🖥️ UI | Streamlit |
| 📚 Document Parsing | PyPDF |
| ✂️ Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| 🔢 Embeddings | Google Gemini `gemini-embedding-2-preview` |
| 🗂️ Vector Store | FAISS (local) |
| 🤖 LLM | Groq — `llama-3.3-70b-versatile` |
| 🔗 Orchestration | LangChain LCEL (prompt → model → parser) |

---

## ✨ Features

- 📂 **Multi-file upload** — process several PDFs in one go
- ⚡ **Fast retrieval** — FAISS similarity search over embedded chunks
- 🧩 **RAG pipeline** — answers grounded strictly in your document context
- 🚦 **Rate-limit safe** — batched embedding with sleep intervals
- 🛡️ **Graceful error handling** — warns on empty uploads or failed text extraction
- 💬 **Conversational Q&A** — real-time answers directly in the Streamlit UI

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot
```

### 2. Set up your environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

- Get your **Google API key** → [Google AI Studio](https://aistudio.google.com/)
- Get your **Groq API key** → [console.groq.com](https://console.groq.com/)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
multi-pdf-rag-chatbot/
│
├── app.py                  # Main Streamlit application
├── .env                    # API keys (not committed)
├── requirements.txt        # Python dependencies
├── faiss_index/            # Auto-generated local vector store
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

---

## 📦 Requirements

```
streamlit
pypdf
langchain
langchain-google-genai
langchain-groq
langchain-community
faiss-cpu
google-generativeai
python-dotenv
```

> Or install all at once: `pip install -r requirements.txt`

---

## ⚠️ Notes

- The FAISS index is saved **locally** in a `faiss_index/` folder. Re-processing files will overwrite it.
- If text cannot be extracted from a PDF (e.g. scanned images without OCR), the app will display a warning.
- Answers are strictly based on document context — the model will say *"answer is not available in the context"* rather than hallucinate.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
