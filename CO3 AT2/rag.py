import os
import re
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# SPORTS RULEBOOK RAG - PHASE 1
# PDF -> CHUNKS -> EMBEDDINGS -> FAISS -> RETRIEVAL
# ============================================================

PDF_FILE = "rulebook.pdf"

# Chunking configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Number of chunks retrieved for every question
TOP_K = 4

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 1. LOAD PDF
# ============================================================

def load_pdf(pdf_path):
    """
    Reads the rulebook PDF page by page.

    Each page is stored separately so that we can
    display the page number when showing sources.
    """

    if not os.path.exists(pdf_path):
        print(f"\nERROR: Could not find '{pdf_path}'.")
        print("Make sure rulebook.pdf is in the same folder as rag.py.")
        return []

    print("\nLoading rulebook...")

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text()

            if text and text.strip():
                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

        except Exception as error:
            print(f"Warning: Could not read page {page_number}: {error}")

    print(f"Total PDF pages: {len(reader.pages)}")
    print(f"Pages containing text: {len(pages)}")

    return pages


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Removes unnecessary whitespace while keeping
    the actual rulebook content.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 3. CHUNK DOCUMENT
# ============================================================

def create_chunks(pages):
    """
    Splits the rulebook into overlapping chunks.

    Chunk size: 800 characters
    Overlap:    150 characters
    """

    chunks = []

    for page_data in pages:

        page_number = page_data["page"]
        text = clean_text(page_data["text"])

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            # Move forward while preserving overlap
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    """
    Converts every text chunk into a numerical vector
    using the MiniLM embedding model.
    """

    print("\nLoading embedding model...")
    print(f"Model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")
    print(f"Number of chunks: {len(texts)}")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return model, embeddings


# ============================================================
# 5. CREATE FAISS VECTOR STORE
# ============================================================

def create_faiss_index(embeddings):
    """
    Creates a FAISS vector index.

    Since embeddings are normalized, inner product
    gives cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings.astype("float32"))

    print("\nFAISS vector store created.")
    print(f"Vector dimension: {dimension}")
    print(f"Vectors indexed: {index.ntotal}")

    return index


# ============================================================
# 6. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(query, model, index, chunks, top_k=TOP_K):
    """
    Converts the user question into an embedding and
    retrieves the most relevant rulebook chunks.
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number == -1:
            continue

        chunk = chunks[index_number]

        results.append({
            "score": float(score),
            "page": chunk["page"],
            "text": chunk["text"]
        })

    return results


# ============================================================
# 7. DISPLAY RETRIEVED SOURCES
# ============================================================

def display_results(query, results):

    print("\n" + "=" * 80)
    print("USER QUERY")
    print("=" * 80)

    print(query)

    print("\n" + "=" * 80)
    print("RETRIEVED RULEBOOK SOURCES")
    print("=" * 80)

    if not results:
        print("No relevant sources found.")
        return

    for number, result in enumerate(results, start=1):

        print("\n" + "-" * 80)

        print(f"SOURCE {number}")
        print(f"PDF Page       : {result['page']}")
        print(f"Similarity     : {result['score']:.4f}")

        print("\nRetrieved Text:")
        print(result["text"])

    print("\n" + "=" * 80)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 80)
    print("             CRICKET RULEBOOK RAG SYSTEM")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"Chunk size     : {CHUNK_SIZE} characters")
    print(f"Chunk overlap  : {CHUNK_OVERLAP} characters")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector store   : FAISS")
    print(f"Top-K          : {TOP_K}")

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    pages = load_pdf(PDF_FILE)

    if not pages:
        return

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print("\nCreating document chunks...")

    chunks = create_chunks(pages)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        print("ERROR: No chunks were created.")
        return

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    model, embeddings = create_embeddings(chunks)

    # --------------------------------------------------------
    # Create FAISS index
    # --------------------------------------------------------

    index = create_faiss_index(embeddings)

    print("\nRAG retrieval system is ready.")

    # --------------------------------------------------------
    # Interactive question answering retrieval
    # --------------------------------------------------------

    while True:

        print("\n" + "-" * 80)

        query = input(
            "\nEnter your cricket rule question "
            "(type 'exit' to stop): "
        ).strip()

        if query.lower() == "exit":
            print("\nRAG system stopped.")
            break

        if not query:
            print("\nERROR: Question cannot be empty.")
            continue

        # Retrieve relevant chunks
        results = retrieve(
            query,
            model,
            index,
            chunks,
            TOP_K
        )

        # Display sources
        display_results(query, results)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()