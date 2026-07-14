import os
import re

INPUT_DIR = "knowledge/text"
OUTPUT_DIR = "knowledge/chunks"
MIN_WORDS = 250
MAX_WORDS = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_into_chunks(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    # نقسم النص حسب العناوين إذا موجودة (Law, Handball, DOGSO...)
    sections = re.split(r"(Law\s+\d+.*|Handball|DOGSO|Direct Free Kick)", text)
    chunks = []
    buffer = []
    word_count = 0

    for section in sections:
        words = section.split()
        for word in words:
            buffer.append(word)
            word_count += 1
            if word_count >= max_words:
                chunks.append(" ".join(buffer))
                buffer = []
                word_count = 0
        if buffer and word_count >= min_words:
            chunks.append(" ".join(buffer))
            buffer = []
            word_count = 0

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks

# نقرأ كل الملفات النصية داخل knowledge/text
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            out_file = os.path.join(OUTPUT_DIR, f"{filename}_chunk{idx+1}.txt")
            with open(out_file, "w", encoding="utf-8") as out:
                out.write(chunk)

print("Chunks created for all text files.")
