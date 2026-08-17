import os
import time
import shutil

import numpy as np
import pandas as pd
import faiss
import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "dataset.csv"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

CHROMA_PATH = "./chroma_db"

FAISS_INDEX_FILE = "faiss_index.bin"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 70)
print("          TWEET TOPIC RETRIEVAL BENCHMARK")
print("=" * 70)

print("\nLoading embedding model...")
print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"{DATASET_FILE} was not found."
    )

df = pd.read_csv(DATASET_FILE)

if len(df) < 200:
    raise ValueError(
        f"Dataset contains only {len(df)} records. "
        "Minimum required is 200."
    )

texts = df["tweet_text"].astype(str).tolist()

ids = df["tweet_id"].astype(str).tolist()

relevance_labels = (
    df["relevant"]
    .astype(int)
    .tolist()
)

print(f"Total tweets: {len(df)}")

print("\nDataset columns:")
print(list(df.columns))


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("CREATING EMBEDDINGS")
print("=" * 70)

print(f"Embedding model: {MODEL_NAME}")
print(f"Number of tweets: {len(texts)}")

embedding_start = time.perf_counter()

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

embedding_time = (
    time.perf_counter() - embedding_start
)

embeddings = embeddings.astype(
    "float32"
)

dimension = embeddings.shape[1]

print(
    f"\nEmbedding generation time: "
    f"{embedding_time:.4f} seconds"
)

print(
    f"Embedding dimension: "
    f"{dimension}"
)


# ============================================================
# BUILD FAISS INDEX
# ============================================================

print("\n" + "=" * 70)
print("BUILDING FAISS INDEX")
print("=" * 70)

faiss_start = time.perf_counter()

# Inner Product + normalized embeddings
# gives cosine similarity.

faiss_index = faiss.IndexFlatIP(
    dimension
)

faiss_index.add(
    embeddings
)

faiss_indexing_time = (
    time.perf_counter() - faiss_start
)

print(
    f"FAISS vectors indexed: "
    f"{faiss_index.ntotal}"
)

print(
    f"FAISS indexing time: "
    f"{faiss_indexing_time:.6f} seconds"
)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

faiss.write_index(
    faiss_index,
    FAISS_INDEX_FILE
)

faiss_file_size = os.path.getsize(
    FAISS_INDEX_FILE
)

faiss_storage_mb = (
    faiss_file_size / (1024 * 1024)
)

print(
    f"FAISS index saved: "
    f"{FAISS_INDEX_FILE}"
)

print(
    f"FAISS storage footprint: "
    f"{faiss_storage_mb:.4f} MB"
)


# ============================================================
# BUILD CHROMADB INDEX
# ============================================================

print("\n" + "=" * 70)
print("BUILDING CHROMADB INDEX")
print("=" * 70)

# Remove previous ChromaDB database
if os.path.exists(CHROMA_PATH):

    shutil.rmtree(
        CHROMA_PATH
    )

chroma_start = time.perf_counter()

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

# Explicitly use cosine distance
collection = chroma_client.create_collection(
    name="tweet_collection",
    metadata={
        "description":
        "SaveThePlanet tweet benchmark",
        "hnsw:space":
        "cosine"
    }
)

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[
        {
            "topic": str(topic),
            "relevant": int(relevant)
        }
        for topic, relevant in zip(
            df["topic"],
            df["relevant"]
        )
    ]
)

chroma_indexing_time = (
    time.perf_counter() - chroma_start
)

print(
    f"ChromaDB documents indexed: "
    f"{collection.count()}"
)

print(
    f"ChromaDB indexing time: "
    f"{chroma_indexing_time:.6f} seconds"
)


# ============================================================
# MEASURE CHROMADB STORAGE
# ============================================================

def get_folder_size(folder):

    total = 0

    if not os.path.exists(folder):
        return 0

    for root, dirs, files in os.walk(folder):

        for file in files:

            file_path = os.path.join(
                root,
                file
            )

            try:
                total += os.path.getsize(
                    file_path
                )

            except OSError:
                pass

    return total


chroma_storage_bytes = get_folder_size(
    CHROMA_PATH
)

chroma_storage_mb = (
    chroma_storage_bytes /
    (1024 * 1024)
)

print(
    f"ChromaDB storage footprint: "
    f"{chroma_storage_mb:.4f} MB"
)


# ============================================================
# TEST QUERIES
# ============================================================

queries = [

    "How can people reduce plastic pollution?",

    "What can we do to fight climate change?",

    "How does renewable energy help the environment?",

    "What are the benefits of recycling?",

    "How can we protect oceans from pollution?",

    "Why should people reduce their carbon footprint?",

    "What actions can communities take to protect nature?",

    "How can sustainable transportation help the planet?",

    "Save the planet",

    "#SaveThePlanet"
]


# ============================================================
# RELEVANCE CALCULATION
# ============================================================

def calculate_relevance(result_ids):

    relevant_count = 0

    for result_id in result_ids:

        matching_rows = df[
            df["tweet_id"].astype(str)
            == str(result_id)
        ]

        if not matching_rows.empty:

            relevance = int(
                matching_rows.iloc[0]["relevant"]
            )

            if relevance == 1:
                relevant_count += 1

    if TOP_K == 0:
        return 0

    return (
        relevant_count /
        min(TOP_K, len(result_ids))
    )


# ============================================================
# FAISS SEARCH
# ============================================================

def search_faiss(query):

    # Query embedding generation is outside
    # the database timing so both databases
    # are compared fairly.

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    start = time.perf_counter()

    scores, indices = (
        faiss_index.search(
            query_embedding,
            TOP_K
        )
    )

    latency = (
        time.perf_counter() - start
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index == -1:
            continue

        results.append({

            "tweet_id":
                ids[index],

            "tweet":
                texts[index],

            "score":
                float(score)
        })

    return results, latency


# ============================================================
# CHROMADB SEARCH
# ============================================================

def search_chroma(query):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()

    start = time.perf_counter()

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K
    )

    latency = (
        time.perf_counter() - start
    )

    output = []

    result_ids = (
        results["ids"][0]
    )

    result_documents = (
        results["documents"][0]
    )

    distances = (
        results["distances"][0]
    )

    for result_id, document, distance in zip(
        result_ids,
        result_documents,
        distances
    ):

        # ChromaDB cosine distance:
        # similarity = 1 - distance

        similarity = (
            1 - float(distance)
        )

        output.append({

            "tweet_id":
                result_id,

            "tweet":
                document,

            "score":
                similarity
        })

    return output, latency


# ============================================================
# RUN BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("RUNNING SEMANTIC SEARCH BENCHMARK")
print("=" * 70)

benchmark_results = []

faiss_latencies = []

chroma_latencies = []

faiss_relevances = []

chroma_relevances = []


for number, query in enumerate(
    queries,
    start=1
):

    print("\n")
    print("#" * 70)

    print(
        f"QUERY {number}: {query}"
    )

    print("#" * 70)


    # ========================================================
    # FAISS
    # ========================================================

    faiss_results, faiss_latency = (
        search_faiss(query)
    )

    faiss_ids = [
        result["tweet_id"]
        for result in faiss_results
    ]

    faiss_relevance = (
        calculate_relevance(
            faiss_ids
        )
    )

    faiss_latencies.append(
        faiss_latency
    )

    faiss_relevances.append(
        faiss_relevance
    )

    print("\nFAISS TOP 5:")

    for rank, result in enumerate(
        faiss_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"[{result['score']:.4f}] "
            f"{result['tweet']}"
        )

    print(
        f"\nFAISS latency: "
        f"{faiss_latency * 1000:.4f} ms"
    )

    print(
        f"FAISS Top-5 relevance: "
        f"{faiss_relevance * 100:.1f}%"
    )


    # ========================================================
    # CHROMADB
    # ========================================================

    chroma_results, chroma_latency = (
        search_chroma(query)
    )

    chroma_ids = [
        result["tweet_id"]
        for result in chroma_results
    ]

    chroma_relevance = (
        calculate_relevance(
            chroma_ids
        )
    )

    chroma_latencies.append(
        chroma_latency
    )

    chroma_relevances.append(
        chroma_relevance
    )

    print("\nCHROMADB TOP 5:")

    for rank, result in enumerate(
        chroma_results,
        start=1
    ):

        print(
            f"{rank}. "
            f"[{result['score']:.4f}] "
            f"{result['tweet']}"
        )

    print(
        f"\nChromaDB latency: "
        f"{chroma_latency * 1000:.4f} ms"
    )

    print(
        f"ChromaDB Top-5 relevance: "
        f"{chroma_relevance * 100:.1f}%"
    )


    # ========================================================
    # STORE QUERY RESULTS
    # ========================================================

    benchmark_results.append({

        "query":
            query,

        "faiss_latency_ms":
            faiss_latency * 1000,

        "chroma_latency_ms":
            chroma_latency * 1000,

        "faiss_relevance_percent":
            faiss_relevance * 100,

        "chroma_relevance_percent":
            chroma_relevance * 100,

        "faiss_top5":
            " | ".join(
                faiss_ids
            ),

        "chroma_top5":
            " | ".join(
                chroma_ids
            )
    })


# ============================================================
# TOP-5 OVERLAP COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TOP-5 OVERLAP COMPARISON")
print("=" * 70)

overlap_results = []

# Compare first 3 queries as required

comparison_queries = queries[:3]

for query in comparison_queries:

    faiss_results, _ = (
        search_faiss(query)
    )

    chroma_results, _ = (
        search_chroma(query)
    )

    faiss_ids = {
        result["tweet_id"]
        for result in faiss_results
    }

    chroma_ids = {
        result["tweet_id"]
        for result in chroma_results
    }

    overlap = (
        faiss_ids.intersection(
            chroma_ids
        )
    )

    overlap_count = len(overlap)

    print("\nQuery:")
    print(query)

    print(
        f"FAISS results: "
        f"{len(faiss_ids)}"
    )

    print(
        f"ChromaDB results: "
        f"{len(chroma_ids)}"
    )

    print(
        f"Top-5 overlap: "
        f"{overlap_count}/5"
    )

    overlap_results.append({

        "query":
            query,

        "faiss_results":
            len(faiss_ids),

        "chroma_results":
            len(chroma_ids),

        "overlap":
            overlap_count,

        "overlap_percent":
            (overlap_count / 5) * 100
    })


# ============================================================
# FINAL METRICS
# ============================================================

# Latencies are stored in seconds.
# Convert average to milliseconds.

average_faiss_latency = (
    np.mean(
        faiss_latencies
    ) * 1000
)

average_chroma_latency = (
    np.mean(
        chroma_latencies
    ) * 1000
)

average_faiss_relevance = (
    np.mean(
        faiss_relevances
    ) * 100
)

average_chroma_relevance = (
    np.mean(
        chroma_relevances
    ) * 100
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL BENCHMARK SUMMARY")
print("=" * 70)

print(
    f"\nDataset size: "
    f"{len(df)} tweets"
)

print(
    f"Embedding model: "
    f"{MODEL_NAME}"
)

print(
    f"Embedding dimension: "
    f"{dimension}"
)

print("\nFAISS")
print("-" * 40)

print(
    f"Indexing time: "
    f"{faiss_indexing_time:.6f} seconds"
)

print(
    f"Average query latency: "
    f"{average_faiss_latency:.4f} ms"
)

print(
    f"Average Top-5 relevance: "
    f"{average_faiss_relevance:.2f}%"
)

print(
    f"Storage footprint: "
    f"{faiss_storage_mb:.4f} MB"
)


print("\nChromaDB")
print("-" * 40)

print(
    f"Indexing time: "
    f"{chroma_indexing_time:.6f} seconds"
)

print(
    f"Average query latency: "
    f"{average_chroma_latency:.4f} ms"
)

print(
    f"Average Top-5 relevance: "
    f"{average_chroma_relevance:.2f}%"
)

print(
    f"Storage footprint: "
    f"{chroma_storage_mb:.4f} MB"
)


# ============================================================
# SAVE DETAILED QUERY RESULTS
# ============================================================

results_df = pd.DataFrame(
    benchmark_results
)

results_df.to_csv(
    "benchmark_results.csv",
    index=False
)

print(
    "\nDetailed query results saved to:"
)

print(
    "benchmark_results.csv"
)


# ============================================================
# SAVE OVERLAP RESULTS
# ============================================================

overlap_df = pd.DataFrame(
    overlap_results
)

overlap_df.to_csv(
    "overlap_results.csv",
    index=False
)

print(
    "Overlap results saved to:"
)

print(
    "overlap_results.csv"
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame([

    {

        "database":
            "FAISS",

        "dataset_size":
            len(df),

        "embedding_model":
            MODEL_NAME,

        "embedding_dimension":
            dimension,

        "indexing_time_seconds":
            faiss_indexing_time,

        "average_query_latency_ms":
            average_faiss_latency,

        "average_top5_relevance_percent":
            average_faiss_relevance,

        "storage_mb":
            faiss_storage_mb
    },

    {

        "database":
            "ChromaDB",

        "dataset_size":
            len(df),

        "embedding_model":
            MODEL_NAME,

        "embedding_dimension":
            dimension,

        "indexing_time_seconds":
            chroma_indexing_time,

        "average_query_latency_ms":
            average_chroma_latency,

        "average_top5_relevance_percent":
            average_chroma_relevance,

        "storage_mb":
            chroma_storage_mb
    }
])

summary_df.to_csv(
    "benchmark_summary.csv",
    index=False
)

print(
    "Final summary saved to:"
)

print(
    "benchmark_summary.csv"
)


# ============================================================
# SPECIAL CASE
# ============================================================

print("\n" + "=" * 70)
print("SPECIAL CASE: HASHTAG-ONLY TWEET")
print("=" * 70)

special_query = "#SaveThePlanet"

faiss_special, _ = (
    search_faiss(
        special_query
    )
)

chroma_special, _ = (
    search_chroma(
        special_query
    )
)

print("\nQuery:")
print(special_query)


print("\nFAISS TOP 5:")

for rank, result in enumerate(
    faiss_special,
    start=1
):

    print(
        f"{rank}. "
        f"[{result['score']:.4f}] "
        f"{result['tweet']}"
    )


print("\nChromaDB TOP 5:")

for rank, result in enumerate(
    chroma_special,
    start=1
):

    print(
        f"{rank}. "
        f"[{result['score']:.4f}] "
        f"{result['tweet']}"
    )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK COMPLETED SUCCESSFULLY")
print("=" * 70)