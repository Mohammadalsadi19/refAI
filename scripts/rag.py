import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class IFABRetriever:
    def __init__(self, index_path="knowledge/faiss.index",
                 chunks_path="knowledge/chunks.json",
                 meta_path="knowledge/metadata.json"):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)  # list of {"id": int, "text": str} — raw IFAB law text chunks
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)  # list of {"id": int, "file": str} — source page/chunk filename
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def search(self, question, top_k=5):
        query_vec = self.model.encode(question).astype("float32")
        query_vec = np.expand_dims(query_vec, axis=0)
        faiss.normalize_L2(query_vec)  # must match the normalization used when building the index
        similarities, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append({
                    "rank": rank + 1,
                    "similarity": float(similarities[0][rank]),
                    "text": self.chunks[idx]["text"],
                    "source": self.metadata[idx]["file"],
                })
        return results

    def top_k(self, question, k=5):
        return self.search(question, top_k=k)


if __name__ == "__main__":
    retriever = IFABRetriever()
    question = "What are the rules about handball?"
    results = retriever.top_k(question, k=5)

    for r in results:
        print(f"Rank {r['rank']} | Similarity: {r['similarity']:.4f}")
        print(f"Source: {r['source']}")
        print(f"Text: {r['text'][:200]}...\n")