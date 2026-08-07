# ============================================================
# AI Curriculum Assistant
# Basic RAG using:
# PDF + ChromaDB + Sentence Transformers + Gemini
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from pypdf import PdfReader

import chromadb

from chromadb.utils import embedding_functions

from google import genai

from dotenv import load_dotenv

import os


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load variables from the .env file
load_dotenv()


# Get the Gemini API key from .env
api_key = os.getenv("GEMINI_API_KEY")


# Check whether the API key was found
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Check your .env file."
    )


# ============================================================
# 3. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=api_key
)


# Choose Gemini model
# gemini-2.5-flash-lite is a stable Gemini model.
MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# 4. LOAD PDF
# ============================================================

def load_pdf(path):
    """
    Reads the PDF file and extracts text
    from every page.
    """

    reader = PdfReader(path)

    text = ""

    # Go through every page
    for page in reader.pages:

        page_text = page.extract_text()

        # Some PDF pages may return None
        if page_text:
            text += page_text + "\n"

    return text


# Load curriculum PDF
text = load_pdf("curriculum_cse.pdf")


print()
print("=" * 50)
print(f"Loaded {len(text)} characters")
print("=" * 50)


# ============================================================
# 5. SPLIT TEXT INTO CHUNKS
# ============================================================

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits large text into smaller pieces.

    chunk_size:
        Maximum number of characters in each chunk.

    overlap:
        Number of characters shared between
        consecutive chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        # Create one chunk
        chunk = text[start:start + chunk_size]

        chunks.append(chunk)

        # Move forward
        # overlap keeps some previous context
        start += chunk_size - overlap

    return chunks


# Create chunks
chunks = chunk_text(text)


print(f"Created {len(chunks)} chunks")


# ============================================================
# 6. CREATE EMBEDDING FUNCTION
# ============================================================

# Sentence Transformers converts text into
# numerical vectors called embeddings.

embedding_function = (
    embedding_functions
    .SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)


# ============================================================
# 7. CREATE CHROMADB
# ============================================================

# PersistentClient stores the database
# permanently inside the ./db folder.

chroma_client = chromadb.PersistentClient(
    path="./db"
)


# Create or open the curriculum collection
collection = chroma_client.get_or_create_collection(
    name="curriculum",
    embedding_function=embedding_function
)


# ============================================================
# 8. STORE PDF CHUNKS IN CHROMADB
# ============================================================

if collection.count() == 0:

    print("Adding chunks to ChromaDB...")

    collection.add(

        # Actual text
        documents=chunks,

        # Extra information about each chunk
        metadatas=[
            {
                "source": "curriculum_cse.pdf",
                "chunk": i
            }
            for i in range(len(chunks))
        ],

        # Unique ID for every chunk
        ids=[
            f"chunk_{i}"
            for i in range(len(chunks))
        ]
    )

    print(
        f"ChromaDB now contains "
        f"{collection.count()} chunks"
    )

else:

    print(
        f"ChromaDB already contains "
        f"{collection.count()} chunks"
    )


# ============================================================
# 9. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(question, k=3):
    """
    Searches ChromaDB for the chunks
    most relevant to the user's question.
    """

    results = collection.query(

        query_texts=[question],

        n_results=k
    )

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    return documents, metadatas


# ============================================================
# 10. CREATE RAG PROMPT
# ============================================================

PROMPT = """
You are a college curriculum assistant.

Your job is to answer the student's question
using ONLY the context provided below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. If the answer is not present in the context,
   say exactly:

   I don't know.

3. Cite the chunk number after each important fact.

Example:

Unit 1 covers Python programming. [c3]

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# 11. BUILD THE PROMPT
# ============================================================

def build_prompt(question):
    """
    Retrieves relevant chunks and combines
    them with the question.
    """

    docs, metas = retrieve(question)

    context = ""

    # Add every retrieved document to the context
    for doc, meta in zip(docs, metas):

        context += (
            f"[c{meta['chunk']}]\n"
        )

        context += doc

        context += "\n\n"

    # Insert context and question
    # into our prompt template
    prompt = PROMPT.format(
        context=context,
        question=question
    )

    return prompt


# ============================================================
# 12. SEND PROMPT TO GEMINI
# ============================================================

def answer(question):
    """
    Sends the RAG prompt to Gemini
    and returns Gemini's answer.
    """

    # Build prompt using retrieved chunks
    prompt = build_prompt(question)

    # Send prompt to Gemini
    response = client.models.generate_content(

        model=MODEL_NAME,

        contents=prompt
    )

    return response.text


# ============================================================
# 13. CHAT LOOP
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 50)
    print("       AI CURRICULUM ASSISTANT")
    print("=" * 50)
    print("Type 'quit' to exit.")
    print()

    while True:

        # Ask user for a question
        question = input("Ask: ")

        # Exit condition
        if question.strip().lower() == "quit":

            print("Goodbye!")

            break

        # Generate answer
        try:

            result = answer(question)

            print()
            print("Answer:")
            print(result)
            print()

        except Exception as e:

            print()
            print("ERROR:")
            print(e)
            print()