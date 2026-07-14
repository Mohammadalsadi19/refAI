import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# مجلد الإدخال والإخراج
CHUNKS_DIR = "knowledge/chunks"
OUTPUT_INDEX = "knowledge/faiss.index"
OUTPUT_CHUNKS = "knowledge/chunks.json"
OUTPUT_META = "knowledge/metadata.json"

# نموذج embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# تخزين النصوص والميتا
chunks_data = []
metadata = []

embeddings = []

# نقرأ كل ملفات الـ chunks
for filename in os.listdir(CHUNKS_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(CHUNKS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # نعمل embedding
        vector = model.encode(text)

        # نخزن البيانات
        chunks_data.append({"id": len(chunks_data), "text": text})
        metadata.append({"id": len(metadata), "file": filename})

        embeddings.append(vector)

# نحول embeddings لمصفوفة NumPy
embeddings_np = np.array(embeddings).astype("float32")

# نطبّع (normalize) كل embedding لطول واحد — عشان L2 distance الخام
# بيتأثر بحجم المتجه (magnitude) مش بس اتجاهه، وهاد بيطلع نتائج مش منطقية
# (مقاطع مش متعلقة بالسؤال تطلع "أقرب" من المقاطع الصح).
# بعد التطبيع، Inner Product = Cosine Similarity بالضبط.
faiss.normalize_L2(embeddings_np)

# نبني FAISS index بمقياس Cosine Similarity (Inner Product على متجهات مطبّعة)
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_np)

# نخزن الـ index
faiss.write_index(index, OUTPUT_INDEX)

# نخزن النصوص والميتا
with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"FAISS index saved to {OUTPUT_INDEX}")
print(f"Chunks saved to {OUTPUT_CHUNKS}")
print(f"Metadata saved to {OUTPUT_META}")