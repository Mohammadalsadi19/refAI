import os
import fitz
from tqdm import tqdm

PDF_FILE = r"C:\Users\Moham\Projects\refAI\knowledge\pdf\IFAB_Laws_2025.pdf"

OUTPUT_DIR = r"C:\Users\Moham\Projects\refAI\knowledge\text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = fitz.open(PDF_FILE)

print(f"Pages: {len(doc)}")

for page_number in tqdm(range(len(doc))):

    page = doc.load_page(page_number)

    text = page.get_text("text")

    text = text.replace("\u00a0", " ")

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    filename = os.path.join(
        OUTPUT_DIR,
        f"page_{page_number+1}.txt"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(text)

print("Done.")