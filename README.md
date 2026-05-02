
<div align="center">

# ⚖️ CrimeScope AI

### RAG-Powered Indian Legal Advisor

*Describe a crime scene in plain English — get the exact Indian laws that apply.*

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=for-the-badge&logo=groq&logoColor=white)

</div>

---

## 🤔 What Is This?

Most people — including lawyers, police officers, and citizens — struggle to find which exact law applies to a given crime. You'd have to manually search through 17+ Indian legal acts, hundreds of sections, and thousands of pages of legal text.

**CrimeScope AI solves this.**

You describe what happened in plain language. CrimeScope reads your description, searches through a knowledge base of Indian legal acts using vector similarity, and returns a structured analysis — crime types identified, applicable acts, specific sections, and a plain-English justification for each.

It's not a chatbot. It doesn't guess. Every answer is grounded in actual retrieved legal text.

---

## ✨ Features

- 🔍 **Natural language input** — describe the crime in simple English, no legal knowledge required
- 📚 **17 Indian Legal Acts** — IPC, BNS, IT Act, POCSO, NDPS, Prevention of Corruption, and more
- 🧠 **RAG Pipeline** — Retrieval-Augmented Generation ensures answers are grounded in real law
- 🏗️ **Structured JSON output** — crime types, act names, section numbers, and justifications
- ⚡ **Fast** — Groq's llama-3.3-70b delivers results in under 5 seconds
- 🌐 **REST API + Web UI** — use it via FastAPI endpoints or the Streamlit interface
- 🔁 **Retry logic** — built-in tenacity retry for flaky API calls
- 📊 **LangSmith tracing** — every LLM call tracked for debugging and optimization

---

## 🏗️ How It Works
User Input (Crime Description)
│
▼
┌─────────────────────────┐
│ Query Understanding │ ← Groq generates multi-angle search queries
└────────────┬────────────┘
│
▼
┌─────────────────────────┐
│ Vector Retrieval │ ← ChromaDB similarity search (MMR, top-12)
│ (ChromaDB + MiniLM) │ sentence-transformers/all-MiniLM-L6-v2
└────────────┬────────────┘
│
▼
┌─────────────────────────┐
│ Legal Reasoning │ ← Groq llama-3.3-70b with structured Pydantic output
│ + JSON Generation │ System prompt = Senior Criminal Advocate persona
└────────────┬────────────┘
│
▼
Structured JSON Response
{crime_type, act, section, justification}

text

---

## 📁 Project Structure
CrimeScope-AI/
├── Data/
│ └── raw_pdfs/ # Drop all 17 legal act PDFs here
├── chroma_db/ # Auto-generated after running ingestion
├── logs/ # Auto-generated log files
├── src/
│ ├── config.py # Central config — all paths, models, settings
│ ├── ingestion.py # PDF → chunk → embed → ChromaDB
│ ├── retriever.py # ChromaDB vector search + MMR
│ ├── chain.py # Full RAG pipeline (retrieve → reason → generate)
│ ├── prompts.py # LLM prompt templates
│ └── schemas.py # Pydantic output schemas
├── tests/
│ └── test_cases.py # Automated test suite (10 scenarios)
├── api.py # FastAPI server
├── app.py # Streamlit web UI
├── ingest_run.py # Run this once to build vector store
├── test_cases.json # Sample test inputs and expected outputs
├── requirements.txt
├── .env.example
└── README.md

text

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Govindsingh072004/CrimeScope-AI.git
cd CrimeScope-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSy_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=CrimeScope-AI
```

> 🔑 Get free keys at: [Groq Console](https://console.groq.com) · [Google AI Studio](https://aistudio.google.com) · [LangSmith](https://smith.langchain.com)

### 5. Add the legal act PDFs

Place all 17 Indian legal act PDFs inside:
Data/raw_pdfs/

text

### 6. Build the vector store (run once)

```bash
python ingest_run.py
```

This reads all PDFs, creates embeddings, and saves them to `chroma_db/`. You only need to run this **once**.

---

## 🚀 Running the App

### Start the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit UI (in a new terminal)

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📡 API Usage

### `POST /analyze-crime`

**Request:**
```json
{
  "description": "A person broke into a house at night, threatened the owner with a knife, and stole cash worth ₹80,000."
}
```

**Response:**
```json
{
  "success": true,
  "processing_time_seconds": 3.42,
  "model_used": "llama-3.3-70b-versatile",
  "analysis": {
    "crime_type": ["Robbery", "House Trespass", "Criminal Intimidation"],
    "applicable_laws": [
      {
        "act": "Indian Penal Code, 1860",
        "section": "Section 390",
        "description": "Robbery — theft combined with use of force or threat of force.",
        "justification": "The accused threatened the owner with a knife while committing theft, escalating it from simple theft to robbery."
      },
      {
        "act": "Indian Penal Code, 1860",
        "section": "Section 448",
        "description": "Punishment for house trespass.",
        "justification": "The accused unlawfully entered a private residence without permission."
      }
    ],
    "ambiguity_note": null
  }
}
```

### `GET /health`
```json
{ "status": "ok", "service": "CrimeScope-AI" }
```

Swagger docs available at: `http://localhost:8000/docs`

---

## 🧪 Running Tests

```bash
# Make sure the API server is running first
python tests/test_cases.py
```

The test suite covers 10 crime scenarios including theft, cybercrime, corruption, domestic violence, and drug offenses.

---

## 📋 Legal Acts Covered

| Act | Year |
|-----|------|
| Bharatiya Nyaya Sanhita (BNS) | 2023 |
| Indian Penal Code (IPC) | 1860 |
| Information Technology Act | 2000 |
| POCSO Act | 2012 |
| Prevention of Corruption Act | 1988 |
| NDPS Act | 1985 |
| Domestic Violence Act | 2005 |
| Arms Act | 1959 |
| Prevention of Money Laundering Act | 2002 |
| Unlawful Activities Prevention Act | 1967 |
| *...and 7 more* | |

---

## ⚠️ Limitations

- **Not a substitute for legal advice.** This tool is for educational and research purposes only.
- Results depend on the quality and coverage of PDFs ingested.
- Very recent legal amendments may not be reflected if PDFs are outdated.
- Highly complex multi-party scenarios may produce incomplete analysis.

---

## 🔮 Future Work

- [ ] Hindi language input support
- [ ] Case law and court judgement database
- [ ] LangGraph agentic pipeline with hallucination validation loop
- [ ] Multi-document cross-referencing (e.g., IPC + BNS overlap analysis)
- [ ] Deployed public demo on Hugging Face Spaces

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq · llama-3.3-70b-versatile |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Observability | LangSmith |
| Validation | Pydantic v2 |

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

<div align="center">

Built with ❤️ for the Digixito Media GenAI Assignment

*"The law is reason, free from passion." — Aristotle*

</div>