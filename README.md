# LAW AI
### Pakistan Legal Information Assistant

LAW AI is a domain-specific Retrieval Augmented Generation (RAG) application. It searches a controlled set of Pakistani legal and regulatory documents and uses the Groq API to write an answer **only from the retrieved text**, with source references. If the documents do not contain the answer, it says so instead of guessing.

> **Disclaimer:** LAW AI provides information retrieved from the legal documents available in its knowledge base. It is intended for educational and informational purposes only and does not constitute legal advice. Laws and regulations may change. For important legal matters, verify the current official law or consult a qualified legal professional.

## Problem statement
General chatbots can state laws, sections and penalties that sound right but are invented. For legal information that is unacceptable. LAW AI restricts the model to documents you supply and refuses when evidence is missing.

## Features
- Built-in knowledge base loaded from `data/legal_documents/` (PDF and TXT)
- Optional session-only upload of a PDF or TXT (never saved)
- Retrieval with FAISS and `sentence-transformers/all-MiniLM-L6-v2`
- Relevance threshold: when nothing relevant is retrieved, Groq is **not called** and the refusal sentence is returned
- Strict grounding prompt, temperature 0
- Sources built from chunk metadata (document, page, possible section, organization, URL if provided), never written by the model
- Retrieved context shown in an expander
- Answers are labelled **Built-in Knowledge Base** or **User Uploaded Document**
- Friendly error messages (missing/invalid key, model unavailable, rate limit, bad or scanned PDF, empty file)

## Architecture
```text
Documents (PDF/TXT) -> text extraction (page numbers kept) -> cleaning -> chunking (800/150)
-> embeddings (MiniLM, 384-dim) -> FAISS index
Question -> embedding -> similarity search -> top-k chunks
-> relevance gate (MIN_SCORE) --no--> "I could not find this information in the provided legal documents."
                              --yes-> context + question -> Groq (llama-3.3-70b-versatile) -> answer + sources
```
The same functions run in the Colab notebook and in `app.py`.

## Data sources
This repository contains **no legal documents by default**. You add them. Use official or authoritative sources only, for example Pakistan Code (pakistancode.gov.pk), National Highways & Motorway Police (nhmp.gov.pk), Punjab Laws Online (punjablaws.gov.pk), and official police/government sites. Verify each document's title, issuing organization, URL and version/date before adding it. Blogs, Wikipedia, forums and social media are not legal sources.

Be careful about jurisdiction (federal vs provincial). Record it in `sources.json` so documents are not mixed up silently.

### `sources.json` (optional, written by you)
Place it in `data/legal_documents/`. Only fields you write are shown as sources. Nothing is guessed.
```json
{
  "Your_File_Name.pdf": {
    "title": "Exact title as printed on the document",
    "organization": "Publisher",
    "url": "Exact URL you downloaded it from",
    "version": "Date or version printed in the document",
    "category": "Criminal Law",
    "jurisdiction": "Federal"
  }
}
```

## Technology stack
Python, pypdf, sentence-transformers, FAISS (faiss-cpu), Groq SDK, Streamlit, GitHub, Streamlit Community Cloud, Google Colab for development. No LangChain, agents, extra databases or paid APIs besides your own Groq key.

## Installation (local)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables
Local: create `.streamlit/secrets.toml` (already git-ignored):
```toml
GROQ_API_KEY = "your_key_here"
```
or set the environment variable `GROQ_API_KEY`. Never commit a key.

## Google Colab development
Open `LAW_AI_RAG.ipynb` in Colab (or follow `LAW_AI_Colab_Cells.md`). Add `GROQ_API_KEY` under Colab Secrets (key icon) with notebook access on. Run the 25 cells in order. The Colab environment has many extra packages; `requirements.txt` lists only what the deployed app needs.

## Run locally
```bash
streamlit run app.py
```
The first start downloads the embedding model and indexes all documents, which can take a few minutes for large laws.

## GitHub
```bash
git init
git add .
git commit -m "Initial LAW AI RAG Assistant"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```
Check that no key is in any file before pushing.

## Streamlit Community Cloud
1. Push the project to GitHub.
2. Open Streamlit Community Cloud and connect your GitHub account.
3. Create an app: choose the repository and the `main` branch.
4. Set the main file to `app.py`.
5. Under *Advanced settings > Secrets*, add `GROQ_API_KEY = "your_key_here"`.
6. Deploy and test, including questions that must be refused.

## Project structure
```text
law-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LAW_AI_RAG.ipynb          # Colab development notebook
├── LAW_AI_Colab_Cells.md     # same cells, readable
└── data/
    └── legal_documents/
        └── sources.json      # optional metadata you write
```

## Example questions
- What is the procedure for registering an FIR?
- What does the law say about arrest?
- When can a search take place according to the provided documents?
- What are the responsibilities of a driver on a motorway?
- What are the responsibilities of NHMP?

Answers appear only if the relevant documents are loaded.

## Source citation system
Each answer lists the document, page, organization, version and URL **only when that metadata exists**. Section numbers are auto-detected from the chunk text and shown as "possible section(s)" because detection is a heuristic.

## Limitations
- Only as good as the loaded documents: they may be outdated, amended, or from the wrong jurisdiction.
- English-only embedding model; Urdu and Roman Urdu questions will retrieve poorly.
- Scanned PDFs are not supported (no OCR).
- Similarity thresholds reduce but cannot eliminate off-target retrieval; `MIN_SCORE` needs calibration on your documents.
- Section detection is heuristic. A chunk may contain several sections.
- An LLM can still misread context. Check the answer against the retrieved text and sources.
- The index is rebuilt on each app restart.

## Future improvements
Urdu / Roman Urdu support and multilingual embeddings, OCR, hybrid search (BM25), reranking, better section-aware retrieval, document version comparison, law update monitoring, provincial law databases, advanced RAG evaluation, citation verification, search filters.
