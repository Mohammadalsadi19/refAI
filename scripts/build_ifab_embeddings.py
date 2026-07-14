import os
import pickle
from sentence_transformers import SentenceTransformer

# مجلد الـ chunks
CHUNKS_DIR = "knowledge/chunks"
# ملف الإخراج
OUTPUT_FILE = "knowledge/embeddings/ifab_embeddings.pkl"

# تأكد إن مجلد الإخراج موجود
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# نموذج الـ embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings_store = []

# نقرأ كل ملفات الـ chunks
for filename in os.listdir(CHUNKS_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(CHUNKS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # نعمل embedding للنص
        embedding = model.encode(text)

        # نخزن الناتج مع اسم الملف
        embeddings_store.append({
            "chunk_file": filename,
            "text": text,
            "embedding": embedding
        })

# نخزن كل الـ embeddings في ملف واحد
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embeddings_store, f)

print(f"Embeddings saved to {OUTPUT_FILE}")
