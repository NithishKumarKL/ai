# scripts/retriever_and_llm.py
import os
import requests
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# CONFIG (set env vars or edit here)
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "insurance_faqs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
LITELLM_URL = os.getenv("LITELLM_URL", "https://aigen.ibn.in/v1/generate")  # example
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")

# PARAMETERS
TOP_K = 3
SIMILARITY_THRESHOLD = 0.2   # tune: smaller = stricter (depends on embedder & distance metric)

class RAGAssistant:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=PERSIST_DIR))
        self.collection = self.client.get_collection(name=COLLECTION_NAME)

    def retrieve(self, query, k=TOP_K):
        q_emb = self.model.encode([query], convert_to_numpy=True).tolist()[0]
        results = self.collection.query(query_embeddings=[q_emb], n_results=k)
        # results: dict with keys 'ids','documents','metadatas','distances'
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "faq_id": results["metadatas"][0][i].get("faq_id"),
                "question": results["metadatas"][0][i].get("question"),
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i]
            })
        return hits

    def build_prompt(self, query, hits):
        context_parts = []
        for h in hits:
            context_parts.append(f"[FAQ {h['faq_id']}] Q: {h['question']}\nA: {h['document']}")
        context = "\n\n".join(context_parts) if context_parts else "No context available."

        template = f"""
You are an insurance FAQ assistant. Use ONLY the facts provided in the 'Context' section below to answer the user's question. If the information is not present in the context, say you cannot answer and recommend escalation to a human agent.

Context:
{context}

User question: {query}

Produce:
1) A short answer (2–6 sentences) grounded only on the context.
2) A 'Sources' line listing FAQ ids you used in square brackets, e.g. Sources: [FAQ 3], [FAQ 9]

Answer:
"""
        return template

    def call_litelm(self, prompt, max_tokens=512, temperature=0.0):
        headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"} if LITELLM_API_KEY else {}
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = requests.post(LITELLM_URL, json=payload, headers=headers, timeout=30)
        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
        # expected: data['text'] or similar; adjust per your LiteLLM API
        # We'll try common fields:
        for key in ("text", "result", "output", "generated_text"):
            if key in data:
                return data[key]
        # fallback:
        return json.dumps(data)

    def answer(self, query):
        hits = self.retrieve(query)
        prompt = self.build_prompt(query, hits)
        llm_resp = self.call_litelm(prompt)
        return {
            "query": query,
            "answer": llm_resp,
            "sources": [h["faq_id"] for h in hits],
            "hits": hits
        }

# Example usage
if __name__ == "__main__":
    rag = RAGAssistant()
    q = "Does my policy cover flood damage?"
    out = rag.answer(q)
    print("ANSWER:\n", out["answer"])
    print("\nSOURCES:", out["sources"])
    for h in out["hits"]:
        print("\n--", h["faq_id"], h["question"])
