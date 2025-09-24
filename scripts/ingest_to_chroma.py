# scripts/ingest_to_chroma.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# CONFIG
CSV_PATH = "data/insurance_faqs.csv"
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "insurance_faqs"

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH, dtype=str)
    df.fillna("", inplace=True)
    print(f"{len(df)} FAQs loaded.")

    print("Loading embedding model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)

    # create combined doc = question + answer (so retrieval returns both)
    docs = (df['question'] + " " + df['answer']).tolist()
    ids = df['faq_id'].tolist()
    metas = []
    for _, row in df.iterrows():
        metas.append({
            "faq_id": row['faq_id'],
            "question": row['question'],
            "tags": row.get('tags', ''),
            "last_updated": row.get('last_updated', '')
        })

    print("Computing embeddings...")
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True).tolist()

    print("Connecting to Chroma...")
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))

    # create or get collection
    try:
        collection = client.create_collection(name=COLLECTION_NAME)
        print("Created collection", COLLECTION_NAME)
    except Exception:
        collection = client.get_collection(name=COLLECTION_NAME)
        print("Using existing collection", COLLECTION_NAME)

    print("Upserting documents to Chroma...")
    collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )

    client.persist()
    print("Ingest complete and persisted to", PERSIST_DIR)

if __name__ == "__main__":
    main()
