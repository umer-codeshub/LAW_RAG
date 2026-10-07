# LAW AI: Google Colab Cells
Run the cells top to bottom. Do not continue if a cell fails. The functions here are the exact same code that `app.py` uses.

---

## CELL 1: Install Dependencies

```python
!pip install -q faiss-cpu sentence-transformers groq pypdf
```

### What this does
Installs `pypdf` (PDF text extraction), `sentence-transformers` (embedding model), `faiss-cpu` (vector search) and `groq` (official Groq SDK). Colab already has numpy/torch. Streamlit is not needed here: we test the RAG logic first.

### Expected output
Mostly silent. Takes 30-90 seconds. If Colab says *Restart runtime*, click it, then continue.

---

## CELL 2: Verify Packages

```python
import io
import os
import re
import json
import numpy as np
import faiss
from pypdf import PdfReader

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"   # listed as a production model in Groq docs
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SCORE = 0.30   # starting value. Calibrate it in Colab (Cell 16).
NOT_FOUND_MESSAGE = "I could not find this information in the provided legal documents."
OPTIONAL_METADATA_FIELDS = ["title", "organization", "url", "version", "category", "jurisdiction"]

import importlib.metadata as md
for pkg in ["faiss-cpu", "sentence-transformers", "groq", "pypdf", "numpy"]:
    print(f"{pkg:22s} {md.version(pkg)}")
print("All packages imported OK")
print("Refusal sentence:", NOT_FOUND_MESSAGE)
```

### What this does
Imports everything and prints versions, so we know the environment works. Also loads the shared **configuration** (chunk size, model names, the refusal sentence).

### Expected output
Version numbers for each package and a final `All packages imported OK`.

---

## CELL 3: Configure Groq Secret

```python
import os
from google.colab import userdata

try:
    os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
    print("Secret loaded into environment (not printed).")
except Exception as error:
    print("Could not read GROQ_API_KEY. Check the secret name and 'Notebook access'.")
```

### What this does
Your API key must never be typed in a cell. In Colab: click the **key icon** in the left sidebar > *Add new secret* > Name: `GROQ_API_KEY`, Value: your key > turn on *Notebook access*. This cell reads it and keeps it in an environment variable, and prints nothing.

### Expected output
`Secret loaded into environment (not printed).` If you see an error, the secret name or notebook access is wrong.

---

## CELL 4: Verify API Key Without Printing It

```python
from groq import Groq

key = os.environ.get("GROQ_API_KEY", "")
print("Key present:", bool(key))
print("Key format looks right:", key.startswith("gsk_"))

try:
    available = sorted(m.id for m in Groq(api_key=key).models.list().data)
    print("Key works. Model available:", GROQ_MODEL_NAME in available)
    print("Models on your account:", available)
except Exception as error:
    print("Groq check failed:", type(error).__name__)
```

### What this does
Checks that the key exists and looks right, then asks Groq for its **live model list** to confirm the key works and that our chosen model (`llama-3.3-70b-versatile`) is available to your account.

### Expected output
`Key present: True`, `Key format looks right: True`, `Model available: True`. If the model is `False`, pick one from the printed list and change `GROQ_MODEL_NAME` in Cell 2.

---

## CELL 5: Load Documents

```python
from google.colab import files
DOCS_FOLDER = "/content/legal_documents"
os.makedirs(DOCS_FOLDER, exist_ok=True)

uploaded = files.upload()
for name, data in uploaded.items():
    with open(os.path.join(DOCS_FOLDER, name), "wb") as f:
        f.write(data)

def load_sources_metadata(folder):
    """Read optional sources.json: {"file.pdf": {"title": ..., "organization": ..., "url": ...}}.
    Only values YOU wrote there are used. Nothing is guessed."""
    path = os.path.join(folder, "sources.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_pdf(source, file_name):
    """Extract text page by page so page numbers are preserved.
    `source` can be a file path or a file-like object."""
    try:
        reader = PdfReader(source)
        if reader.is_encrypted:
            reader.decrypt("")
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "metadata": {"source": file_name, "page": page_number}})
    except Exception:
        raise ValueError(f"'{file_name}' could not be read. The file may be corrupted or protected.")
    if not pages:
        raise ValueError(f"No text could be extracted from '{file_name}'. It may be a scanned PDF (OCR is not supported in this version).")
    return pages


def load_txt(source, file_name):
    """Read a plain text file. TXT files have no page numbers."""
    try:
        if hasattr(source, "read"):
            raw = source.read()
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else raw
        else:
            with open(source, encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception:
        raise ValueError(f"'{file_name}' could not be read.")
    if not text.strip():
        raise ValueError(f"'{file_name}' is empty.")
    return [{"text": text, "metadata": {"source": file_name}}]


def load_document(source, file_name):
    """Choose the right loader from the file extension."""
    extension = os.path.splitext(file_name)[1].lower()
    if extension == ".pdf":
        return load_pdf(source, file_name)
    if extension == ".txt":
        return load_txt(source, file_name)
    raise ValueError(f"'{file_name}' is not supported. Please use PDF or TXT.")


def load_folder(folder):
    """Load every PDF/TXT in a folder and attach metadata from sources.json (if you wrote any).
    Returns (records, report, errors)."""
    records, report, errors = [], [], []
    sources_metadata = load_sources_metadata(folder)
    if not os.path.isdir(folder):
        return records, report, [f"Folder not found: {folder}"]
    for file_name in sorted(os.listdir(folder)):
        if not file_name.lower().endswith((".pdf", ".txt")):
            continue
        try:
            file_records = load_document(os.path.join(folder, file_name), file_name)
        except ValueError as error:
            errors.append(str(error))
            continue
        extra = sources_metadata.get(file_name, {})
        for record in file_records:
            for field in OPTIONAL_METADATA_FIELDS:
                if extra.get(field):
                    record["metadata"][field] = extra[field]
        records.extend(file_records)
        report.append({
            "file": file_name,
            "pages_or_docs": len(file_records),
            "characters": sum(len(r["text"]) for r in file_records),
            "metadata_from_sources_json": bool(extra),
        })
    return records, report, errors

records, report, errors = load_folder(DOCS_FOLDER)
for row in report:
    print(row)
for message in errors:
    print("PROBLEM:", message)
print(f"\n{len(records)} page/document records loaded")
```

### What this does
**Upload your official PDF/TXT files** (the file picker appears). Optionally also upload a `sources.json` that you write by hand (see README) with title, organization, URL, version/date and category for each file. Then the loaders read each file page by page and keep real page numbers. Unreadable, empty, scanned or unsupported files are reported, not silently skipped.

### Expected output
A file picker; afterwards a list of loaded files with page counts and character counts, and any error messages.

---

## CELL 6: Inspect Extracted Text

```python
for record in records[:3]:
    print(record["metadata"], "| characters:", len(record["text"]))
    print(record["text"][:400])
    print("-" * 60)
```

### What this does
Look at the raw extracted text yourself. Check that it is readable English, not garbled, and that page numbers look right. Bad extraction here means bad answers later.

### Expected output
For each of the first 3 records: file, page, length and the first 400 characters.

---

## CELL 7: Clean Text

```python
def clean_text(text):
    """Light cleaning only. We do NOT rewrite or 'fix' legal wording."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

sample = records[0]["text"]
print("Before:", len(sample), "characters | After:", len(clean_text(sample)), "characters")
```

### What this does
Light cleaning only: removes null bytes and repeated spaces/blank lines. We deliberately do **not** rewrite legal wording.

### Expected output
A before/after length for one page.

---

## CELL 8: Chunk Documents

```python
# Matches lines that start like a numbered provision, e.g. "154. Information in cognizable cases"
SECTION_PATTERN = re.compile(r"^\s*(\d{1,3}[A-Z]{0,2})\.\s+[A-Z]", re.MULTILINE)


def detect_sections(text):
    """Find provision numbers that literally appear in the chunk text. Heuristic: may be missed or wrong,
    so the app labels it 'possible'. Returns [] if nothing is found (never guessed)."""
    found = []
    for match in SECTION_PATTERN.finditer(text):
        if match.group(1) not in found:
            found.append(match.group(1))
    return found[:5]


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Cut text into overlapping pieces, preferring to end at a line break or sentence end."""
    pieces, start, length = [], 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window_start = start + int(chunk_size * 0.75)
            cut = max(text.rfind("\n", window_start, end), text.rfind(". ", window_start, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_documents(records, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk each page separately so every chunk keeps its real page number."""
    chunks, chunk_id = [], 0
    for record in records:
        for piece in split_text(clean_text(record["text"]), chunk_size, overlap):
            metadata = dict(record["metadata"])
            metadata["chunk_id"] = chunk_id
            sections = detect_sections(piece)
            if sections:
                metadata["sections"] = ", ".join(sections)
            chunks.append({"text": piece, "metadata": metadata})
            chunk_id += 1
    return chunks

chunks = chunk_documents(records)
lengths = [len(c["text"]) for c in chunks]
print(f"Created {len(chunks)} chunks | average {sum(lengths)//len(lengths)} chars | max {max(lengths)} chars")
```

### What this does
**Why chunk?** A whole law is far too big to embed as one vector, and one vector can only capture one general meaning. **Overlap (150)** repeats the end of one chunk at the start of the next so a sentence cut at the boundary is still found. **Too large** chunks blur several topics together and send too much text to the LLM. **Too small** chunks lose context (a penalty separated from the offence it belongs to). 800/150 is a starting point to tune. Each page is chunked separately so every chunk keeps its true page number. A provision number such as `154` is stored only if it literally appears in the chunk text.

### Expected output
`Created N chunks` with an average and maximum chunk length (max about 800).

---

## CELL 9: Inspect Chunks and Metadata

```python
import random
for chunk in random.sample(chunks, 3):
    print(chunk["metadata"])
    print(chunk["text"][:500])
    print("-" * 60)
print("Chunks with a detected section number:", sum(1 for c in chunks if "sections" in c["metadata"]), "of", len(chunks))
```

### What this does
Check three random chunks and their metadata. `sections` is a heuristic: it can be missing or imperfect, and the app labels it *possible*.

### Expected output
Three chunks with metadata, and how many chunks have a detected section.

---

## CELL 10: Load Embedding Model

```python
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model, texts, batch_size=64):
    """Turn texts into normalized float32 vectors (length 1), so inner product = cosine similarity."""
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")

embedder = load_embedding_model()
print("Model loaded")
```

### What this does
An **embedding model** converts text into a list of numbers (a vector) so that texts with similar *meaning* get similar vectors. We use `all-MiniLM-L6-v2` (English only). It is only used, not trained.

### Expected output
A download progress bar the first time, then `Model loaded`.

---

## CELL 11: Generate a Test Embedding

```python
vec = embed_texts(embedder, [chunks[0]["text"]])
print("Embedding shape:", vec.shape)
print("First 5 numbers:", vec[0][:5])

a, b, c = embed_texts(embedder, [
    "A car travelling too fast on the road.",
    "Driving a vehicle above the speed limit.",
    "A recipe for chocolate cake.",
])
print("car vs speeding :", round(float(a @ b), 3))
print("car vs cake     :", round(float(a @ c), 3))
```

### What this does
Embeds one chunk and shows its shape (384 numbers). Then compares three simple sentences to show *semantic similarity*: closer to 1.0 means closer meaning.

### Expected output
Shape `(1, 384)`. The two related sentences should score noticeably higher than the unrelated one.

---

## CELL 12: Create FAISS Index

```python
def build_faiss_index(chunks, model):
    """Embed all chunks and store them in a FAISS inner-product index."""
    if not chunks:
        raise ValueError("There are no chunks to index.")
    vectors = embed_texts(model, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

import time
start = time.time()
index = build_faiss_index(chunks, embedder)
print(f"Indexed {index.ntotal} vectors in {time.time()-start:.1f}s")
```

### What this does
**FAISS** is a library for fast nearest-neighbour search over vectors. An **index** is the structure holding all chunk vectors. `IndexFlatIP` compares the query to every vector with inner product; because our vectors are normalized, that equals cosine similarity (1.0 = same direction). It is exact, simple and fast enough for thousands of chunks.

### Expected output
`Indexed N vectors` equal to the chunk count, with the time taken (can take 1-3 minutes for big laws on CPU).

---

## CELL 13: Test FAISS

```python
test_vec = embed_texts(embedder, [chunks[10 % len(chunks)]["text"]])
scores, ids = index.search(test_vec, 3)
print("Top hit is the same chunk:", ids[0][0] == 10 % len(chunks))
print("Scores:", [round(float(s), 3) for s in scores[0]])
```

### What this does
Sanity test: searching with a chunk's own vector must return that same chunk first with a score near 1.0.

### Expected output
`Top hit is the same chunk: True` and a score close to 1.0.

---

## CELL 14: Create retrieve_documents()

```python
def retrieve_documents(query, index, chunks, model, k=TOP_K):
    """Embed the query, search FAISS, return top-k chunks with metadata and similarity scores."""
    if not query or not query.strip() or index is None or index.ntotal == 0:
        return []
    query_vector = embed_texts(model, [query.strip()])
    scores, ids = index.search(query_vector, min(k, index.ntotal))
    results = []
    for score, chunk_index in zip(scores[0], ids[0]):
        if chunk_index == -1:
            continue
        results.append({
            "text": chunks[chunk_index]["text"],
            "metadata": chunks[chunk_index]["metadata"],
            "score": float(score),
        })
    return results

print("retrieve_documents() ready")
```

### What this does
The retrieval function: embeds the question, searches FAISS, returns the top chunks with their metadata and similarity scores. **No Groq yet.**

### Expected output
No output (function defined).

---

## CELL 15: Test Retrieval

```python
test_questions = [
    "What is the procedure for registering an FIR?",
    "What does the law say about arrest?",
    "What are the rules related to search and seizure?",
    "What are the responsibilities of a driver on a motorway?",
    "What does the Pakistan Penal Code say about this offence?",
]
for q in test_questions:
    print("\nQUESTION:", q)
    for r in retrieve_documents(q, index, chunks, embedder, k=3):
        print(f"  score={r['score']:.2f} | {r['metadata'].get('source')} | page {r['metadata'].get('page')}")
        print("   ", r["text"][:200].replace("\n", " "))
```

### What this does
Runs your five mandatory test questions through retrieval only.

### Expected output
For each question, the top 3 chunks with score, document, page and a text preview.

---

## CELL 16: Inspect Retrieval Manually and Calibrate the Threshold

```python
in_scope = test_questions
out_of_scope = [
    "What is the weather in Lahore?",
    "Give me a recipe for biryani.",
    "Who won the 1992 cricket world cup?",
]
print("IN SCOPE (best score):")
for q in in_scope:
    print(f"  {retrieve_documents(q, index, chunks, embedder, k=1)[0]['score']:.2f}  {q}")
print("OUT OF SCOPE (best score):")
for q in out_of_scope:
    print(f"  {retrieve_documents(q, index, chunks, embedder, k=1)[0]['score']:.2f}  {q}")
print("\nCurrent MIN_SCORE:", MIN_SCORE)
# To change it:  MIN_SCORE = 0.35
```

### What this does
Read the retrieved text yourself: does it actually answer the question? If not, fix retrieval (chunk size, documents) **before** using the LLM. This cell also prints the best score for in-scope vs out-of-scope questions so you can choose `MIN_SCORE`: it should sit **between** the two groups.

### Expected output
Best scores per question. In-scope questions should clearly score above the out-of-scope ones. Adjust `MIN_SCORE` (default 0.30) if they do not.

---

## CELL 17: Connect Groq

```python
SYSTEM_PROMPT = f"""You are LAW AI, a document based Pakistani legal information assistant.

Answer ONLY using the provided context.
Do not use outside knowledge. Do not guess.
Do not invent laws, sections, penalties, fines, procedures, citations, cases, or sources.

If the answer is not supported by the provided context, reply with exactly this sentence and nothing else:
"{NOT_FOUND_MESSAGE}"

If the context only partly answers the question, give the supported part and clearly say what is not covered.
If the context contains different information on the same matter, say: "The retrieved documents contain different information on this matter." and describe each, naming its source.
Refer to sources as [Source 1], [Source 2], etc. as given in the context. Only mention section numbers or pages that appear in the context.
Do not claim that information is current unless the provided documents establish that.
This is informational assistance, not legal advice."""


def get_groq_client(api_key):
    from groq import Groq
    if not api_key:
        raise RuntimeError("The Groq API key is missing. Add GROQ_API_KEY to your secrets.")
    return Groq(api_key=api_key)


def build_context(results):
    """Label each retrieved chunk so the model (and the reader) can tell sources apart."""
    blocks = []
    for number, result in enumerate(results, start=1):
        meta = result["metadata"]
        header = f"[Source {number}] Document: {meta.get('title') or meta.get('source', 'Unknown')}"
        if meta.get("organization"):
            header += f" | Organization: {meta['organization']}"
        if meta.get("version"):
            header += f" | Version/date: {meta['version']}"
        if meta.get("page"):
            header += f" | Page: {meta['page']}"
        if meta.get("sections"):
            header += f" | Possible section(s): {meta['sections']}"
        blocks.append(header + "\n" + result["text"])
    return "\n\n---\n\n".join(blocks)

client = get_groq_client(os.environ.get("GROQ_API_KEY"))
print("Groq client ready\n")
print(SYSTEM_PROMPT)
```

### What this does
Creates the Groq client and defines the **strict system prompt** and the context builder. The prompt tells the model to use only the supplied context and to output the exact refusal sentence otherwise.

### Expected output
`Groq client ready`, then the system prompt text.

---

## CELL 18: Create generate_answer()

```python
def generate_answer(question, results, client, model_name=GROQ_MODEL_NAME):
    """Ask Groq to answer from the retrieved context only. Raises RuntimeError with a friendly message."""
    import groq
    context = build_context(results)
    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above."
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=900,
        )
        return response.choices[0].message.content.strip()
    except groq.AuthenticationError:
        raise RuntimeError("The Groq API key is invalid. Please check GROQ_API_KEY.")
    except groq.NotFoundError:
        raise RuntimeError(f"The model '{model_name}' is not available. Please choose a different Groq model.")
    except groq.RateLimitError:
        raise RuntimeError("Groq rate limit reached. Please wait a moment and try again.")
    except groq.APIConnectionError:
        raise RuntimeError("Could not reach Groq. Please check the internet connection.")
    except groq.APIStatusError:
        raise RuntimeError("Groq returned an error. Please try again.")

print("generate_answer() ready")
```

### What this does
Sends the retrieved context and question to Groq with `temperature=0` (no creative variation). Failures (bad key, model missing, rate limit, no connection) become simple messages instead of tracebacks.

### Expected output
No output (function defined).

---

## CELL 19: Test Answer Generation

```python
q = "What is the procedure for registering an FIR?"
hits = [r for r in retrieve_documents(q, index, chunks, embedder) if r["score"] >= MIN_SCORE]
print("Chunks passed to Groq:", len(hits))
print(generate_answer(q, hits, client))
print("\n--- CONTEXT SENT ---")
print(build_context(hits)[:2500])
```

### What this does
Retrieves for one question, keeps only chunks above `MIN_SCORE`, and asks Groq. **Read the answer against the chunks**: every claim must be in the text.

### Expected output
A grounded answer, with the context chunks printed below it for checking.

---

## CELL 20: Create rag_query()

```python
def rag_query(question, index, chunks, model, client, k=TOP_K, min_score=MIN_SCORE,
              origin="Built-in Knowledge Base"):
    """Full pipeline: retrieve -> relevance gate -> Groq -> answer + sources.
    Groq is NEVER called when no relevant context was retrieved."""
    result = {"question": question, "answer": "", "sources": [], "context": [],
              "retrieved": [], "found": False, "origin": origin, "error": None}

    if not question or not question.strip():
        result["answer"] = "Please type a question."
        return result

    try:
        retrieved = retrieve_documents(question, index, chunks, model, k=k)
    except Exception:
        result["error"] = "Search failed. Please try again."
        return result
    result["retrieved"] = retrieved

    relevant = [r for r in retrieved if r["score"] >= min_score]
    if not relevant:
        result["answer"] = NOT_FOUND_MESSAGE
        return result

    try:
        answer = generate_answer(question, relevant, client)
    except RuntimeError as error:
        result["error"] = str(error)
        return result

    result["answer"] = answer
    result["context"] = relevant
    if NOT_FOUND_MESSAGE in answer:
        result["context"] = []          # model refused, so show no sources
        return result
    result["found"] = True
    result["sources"] = format_sources(relevant)   # built from metadata, NOT written by the model
    return result

print("rag_query() ready")
```

### What this does
The full pipeline in one function: retrieve > **relevance gate** > Groq > answer + sources. If no chunk reaches `MIN_SCORE`, Groq is **never called**. Sources are built from chunk metadata, not written by the model, so it cannot invent a citation.

### Expected output
No output (function defined).

---

## CELL 21: Test Normal Questions

```python
def ask(q):
    return rag_query(q, index, chunks, embedder, client)

for q in test_questions[:3]:
    r = ask(q)
    print("\nQ:", q)
    print("A:", r["answer"] or r["error"])
    print("Sources:", r["sources"])
```

### What this does
Runs in-scope questions through the whole pipeline.

### Expected output
An answer and its sources for each question.

---

## CELL 22: Test Unknown Questions

```python
unknown_questions = [
    "What is the weather in Lahore?",
    "Give me a recipe for biryani.",
    "What is the current fine for speeding?",   # refuses ONLY if your documents do not contain it
    "What does the Companies Act say about share capital?",   # refuses if that law is not loaded
]
for q in unknown_questions:
    r = ask(q)
    print("\nQ:", q)
    print("A:", r["answer"] or r["error"])
    print("Refused:", r["answer"] == NOT_FOUND_MESSAGE, "| Sources:", r["sources"])
```

### What this does
The most important test. Each of these **must** return the refusal sentence and no sources.

### Expected output
Every line shows the exact sentence `I could not find this information in the provided legal documents.` If any question gets a real answer, tell me: we must investigate (threshold, prompt, or the documents really contain it).

---

## CELL 23: Add Source Display

```python
def show(result):
    print("QUESTION:", result["question"])
    print("ANSWER:\n" + (result["answer"] or result["error"] or ""))
    print("\nOrigin:", result["origin"])
    print("Sources")
    for line in result["sources"] or ["(none)"]:
        print(" -", line)
    print("=" * 70)

show(ask("What does the law say about arrest?"))
```

### What this does
A clean printout used for reading results: answer, origin, sources, and the retrieved context for verification.

### Expected output
A formatted answer block with Sources.

---

## CELL 24: Create Evaluation Questions

```python
eval_questions = [
    {"q": "What is the procedure for registering an FIR?", "type": "in_scope", "check": "Answer + sources should come from the criminal procedure document, if loaded."},
    {"q": "What does the law say about arrest?", "type": "in_scope", "check": "Retrieved text must mention arrest provisions."},
    {"q": "When can a search take place according to the documents?", "type": "in_scope", "check": "Retrieved text must be about search/seizure."},
    {"q": "What is the law on bail?", "type": "in_scope", "check": "Only if bail provisions are in your documents."},
    {"q": "What are the responsibilities of a driver on a motorway?", "type": "in_scope", "check": "Needs motorway/NHMP rules loaded."},
    {"q": "What are the responsibilities of NHMP?", "type": "in_scope", "check": "Needs NHMP/highway documents loaded."},
    {"q": "What are the duties of an investigating officer?", "type": "in_scope", "check": "Police Rules / procedure documents."},
    {"q": "What does the Pakistan Penal Code say about theft?", "type": "in_scope", "check": "Needs the Penal Code loaded."},
    {"q": "What is the weather in Lahore?", "type": "must_refuse", "check": "Must refuse."},
    {"q": "Give me a recipe for biryani.", "type": "must_refuse", "check": "Must refuse."},
    {"q": "Who is the current Prime Minister of Pakistan?", "type": "must_refuse", "check": "Must refuse (not a legal-document question)."},
]
print(len(eval_questions), "evaluation questions ready")
```

### What this does
About 11 questions: 8 in-scope and 3 that must be refused. The `check` column says what **you** must verify by reading the retrieved text. I cannot pre-write the correct legal answers, because the right answer is whatever your official documents say.

### Expected output
`11 evaluation questions ready`.

---

## CELL 25: Evaluate Retrieval and Grounding

```python
import pandas as pd

rows = []
for item in eval_questions:
    r = ask(item["q"])
    rows.append({
        "question": item["q"],
        "type": item["type"],
        "what_to_check": item["check"],
        "best_score": round(r["retrieved"][0]["score"], 2) if r["retrieved"] else None,
        "refused": r["answer"] == NOT_FOUND_MESSAGE or r["error"] is not None,
        "answer": r["answer"] or r["error"],
        "sources": " || ".join(r["sources"]),
        "retrieved_preview": " || ".join(x["text"][:120].replace("\n", " ") for x in r["context"][:3]),
        # fill in by hand after reading:
        "retrieval_relevant (y/n)": "", "grounded (y/n)": "", "hallucination (y/n)": "", "missing_info": "",
    })
df = pd.DataFrame(rows)
df.to_csv("law_ai_evaluation.csv", index=False)
pd.set_option("display.max_colwidth", 80)
display(df[["question", "type", "best_score", "refused", "answer"]])

must = df[df["type"] == "must_refuse"]
print(f"Must-refuse questions correctly refused: {int(must['refused'].sum())}/{len(must)}")
print("Saved law_ai_evaluation.csv (download it from the Files panel)")
```

### What this does
Runs every question, records the retrieved chunks, sources and answer, saves a CSV and counts refusals. **Then review manually**: for each in-scope row, mark relevance, groundedness, hallucination and missing info in the CSV.

### Expected output
A table plus `Must-refuse questions correctly refused: X/3`. Read each answer next to its retrieved text.
