
import fitz
import os

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text_output = ""
    for i, page in enumerate(doc):
        text_output += f"## Page {i+1}\n\n" + page.get_text("text") + "\n\n--- Page Separator ---\n\n"
    
    markdown_filename = os.path.splitext(file_path)[0] + ".md"
    with open(markdown_filename, "w", encoding="utf-8") as f:
        f.write(text_output)
    print(f"Extracted text saved to: {markdown_filename}")