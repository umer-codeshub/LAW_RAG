"""LAW AI - Pakistan Legal Information Assistant (Streamlit app).
RAG logic = the same functions tested step by step in the Colab notebook."""

import io
import os
import re
import json
import numpy as np
import faiss
from pypdf import PdfReader

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"   # GPT-OSS 120B on Groq (reasoning model)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SCORE = 0.30   # starting value. Calibrate it in Colab (Cell 16).
NOT_FOUND_MESSAGE = "I could not find this information in the provided legal documents."
OPTIONAL_METADATA_FIELDS = ["title", "organization", "url", "version", "category", "jurisdiction"]

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

def clean_text(text):
    """Light cleaning only. We do NOT rewrite or 'fix' legal wording."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

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

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model, texts, batch_size=64):
    """Turn texts into normalized float32 vectors (length 1), so inner product = cosine similarity."""
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")

def build_faiss_index(chunks, model):
    """Embed all chunks and store them in a FAISS inner-product index."""
    if not chunks:
        raise ValueError("There are no chunks to index.")
    vectors = embed_texts(model, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

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

def format_source(metadata):
    """Build a citation line using ONLY metadata that exists."""
    parts = [metadata.get("title") or metadata.get("source", "Unknown document")]
    if metadata.get("sections"):
        parts.append(f"Possible section(s): {metadata['sections']}")
    if metadata.get("page"):
        parts.append(f"Page {metadata['page']}")
    if metadata.get("organization"):
        parts.append(metadata["organization"])
    if metadata.get("version"):
        parts.append(f"Version/date: {metadata['version']}")
    if metadata.get("url"):
        parts.append(metadata["url"])
    return " | ".join(parts)


def format_sources(results):
    """Unique citation lines for a list of retrieved chunks."""
    lines = []
    for result in results:
        line = format_source(result["metadata"])
        if line not in lines:
            lines.append(line)
    return lines

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
            max_tokens=2500,   # reasoning models spend part of this on thinking, so keep it generous
            extra_body={"reasoning_effort": "low"},   # supported by gpt-oss models; keeps answers fast
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("The model returned an empty answer. Please try again.")
        return answer
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

# =====================================================================
# STREAMLIT USER INTERFACE (everything above is the same code tested in Colab)
# =====================================================================
import streamlit as st

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "legal_documents")
DISCLAIMER = (
    "LAW AI provides information retrieved from the legal documents available in its knowledge base. "
    "It is intended for educational and informational purposes only and does not constitute legal advice. "
    "Laws and regulations may change. For important legal matters, verify the current official law or "
    "consult a qualified legal professional."
)
BUILT_IN = "Built-in Knowledge Base"
UPLOADED = "User Uploaded Document"

st.set_page_config(page_title="LAW AI", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Building the legal knowledge base (first start only)...")
def get_builtin_knowledge_base():
    model = get_embedding_model()
    records, report, errors = load_folder(DATA_FOLDER)
    chunks = chunk_documents(records)
    index = build_faiss_index(chunks, model) if chunks else None
    return {"index": index, "chunks": chunks, "report": report, "errors": errors}


def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


@st.cache_resource
def get_cached_client(api_key):
    return get_groq_client(api_key)


def build_uploaded_knowledge_base(uploaded_file, model):
    """Temporary, in-memory only. Nothing is saved to disk."""
    records = load_document(io.BytesIO(uploaded_file.getvalue()), uploaded_file.name)
    chunks = chunk_documents(records)
    return {"index": build_faiss_index(chunks, model), "chunks": chunks, "name": uploaded_file.name}


def show_result_details(message):
    if message.get("origin"):
        st.caption(f"Answer source: {message['origin']}")
    if message.get("sources"):
        st.markdown("**Sources**")
        for line in message["sources"]:
            st.markdown(f"- {line}")
    if message.get("context"):
        with st.expander("Retrieved context"):
            for number, item in enumerate(message["context"], start=1):
                st.markdown(f"**[Source {number}]** {format_source(item['metadata'])}  \n_similarity: {item['score']:.2f}_")
                st.text(item["text"])


# ---------- Header ----------
st.title("LAW AI")
st.subheader("Pakistan Legal Information Assistant")
st.info(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []

embedding_model = get_embedding_model()
builtin_kb = get_builtin_knowledge_base()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Knowledge Base")
    if builtin_kb["report"]:
        for item in builtin_kb["report"]:
            st.markdown(f"✓ {item['file']}")
        st.caption(f"{len(builtin_kb['chunks'])} searchable chunks")
        categories = sorted({c["metadata"]["category"] for c in builtin_kb["chunks"] if c["metadata"].get("category")})
        if categories:
            st.markdown("**Legal categories**")
            for category in categories:
                st.markdown(f"- {category}")
    else:
        st.warning("No built-in legal documents found in data/legal_documents/.")
    for message in builtin_kb["errors"]:
        st.warning(message)

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("PDF or TXT (this session only)", type=["pdf", "txt"])
    uploaded_kb = None
    if uploaded_file is not None:
        cache_key = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("upload_key") != cache_key:
            try:
                with st.spinner("Processing document..."):
                    st.session_state.upload_kb = build_uploaded_knowledge_base(uploaded_file, embedding_model)
                    st.session_state.upload_key = cache_key
            except ValueError as error:
                st.session_state.pop("upload_kb", None)
                st.session_state.pop("upload_key", None)
                st.error(str(error))
            except Exception:
                st.session_state.pop("upload_kb", None)
                st.session_state.pop("upload_key", None)
                st.error("The document could not be processed.")
        uploaded_kb = st.session_state.get("upload_kb")
        if uploaded_kb:
            st.success(f"Ready: {uploaded_kb['name']} ({len(uploaded_kb['chunks'])} chunks)")
    else:
        st.session_state.pop("upload_kb", None)
        st.session_state.pop("upload_key", None)

    options = [BUILT_IN] + ([UPLOADED] if uploaded_kb else [])
    search_in = st.radio("Search in", options)

    st.divider()
    top_k = st.slider("Number of retrieved chunks", 1, 10, TOP_K)
    st.caption(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    st.caption("Vector database: FAISS")
    st.caption(f"LLM: Groq ({GROQ_MODEL_NAME})")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_result_details(message)

# ---------- New question ----------
question = st.chat_input("Ask a question about the legal documents...")
if question is not None:
    if not question.strip():
        st.warning("Please type a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            api_key = get_groq_key()
            active_kb = uploaded_kb if (search_in == UPLOADED and uploaded_kb) else builtin_kb
            origin = UPLOADED if active_kb is uploaded_kb and uploaded_kb else BUILT_IN

            if not api_key:
                reply = {"role": "assistant", "content": "The Groq API key is missing. Add GROQ_API_KEY to the Streamlit secrets.", "origin": None}
            elif active_kb["index"] is None:
                reply = {"role": "assistant", "content": NOT_FOUND_MESSAGE, "origin": origin}
            else:
                with st.spinner("Searching the legal documents..."):
                    result = rag_query(question, active_kb["index"], active_kb["chunks"],
                                       embedding_model, get_cached_client(api_key), k=top_k, origin=origin)
                reply = {
                    "role": "assistant",
                    "content": result["error"] or result["answer"],
                    "sources": result["sources"],
                    "context": result["context"],
                    "origin": origin if not result["error"] else None,
                }
            st.markdown(reply["content"])
            show_result_details(reply)
        st.session_state.messages.append(reply)
