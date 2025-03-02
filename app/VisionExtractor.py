import fitz  # PyMuPDF
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()

page_separator = "--- Page Separator ---"

def pdfs_to_images(pdf_folder, image_folder, dpi=300):
    os.makedirs(image_folder, exist_ok=True)

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            try:
                doc = fitz.open(pdf_path)
                pdf_name = os.path.splitext(filename)[0]
                pdf_count = len(doc)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    matrix = fitz.Matrix(dpi / 100, dpi / 100)  # Calculate matrix based on DPI
                    pix = page.get_pixmap(matrix=matrix)
                    image_filename = os.path.join(image_folder, f'page_{page_num + 1}.png')
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save(image_filename, "PNG")
                    print(f'Saved: {image_filename}')
                doc.close()
                return pdf_name, pdf_count

            except Exception as e:
                print(f"Error processing {filename}: {e}")


def extract_text_from_images(image_path, output_path, pdf_name):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel(os.environ.get("FLASH_MODEL_NAME"))
    prompt = ("Extract all text, tabular data, and visual elements (such as graphs and charts) from the given image. "
              "Provide the extracted text first, followed by a structured description of any tables and charts found in the image. "
              "Format tables in a structured text representation.")
    os.makedirs(output_path, exist_ok=True)
    output_file_name = pdf_name + "_output.md"
    output_file = output_path + "/" + output_file_name
    with open(output_file, "w", encoding="utf-8") as f:
        # Sort the filenames numerically
        image_names = sorted(
            [filename for filename in os.listdir(image_path) if filename.endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        for image_name in image_names:
            image_path_full = os.path.join(image_path, image_name)
            image = Image.open(image_path_full)
            response = model.generate_content([prompt, image])
            text = response.text if response else ""
            f.write(f"Page {image_name}\n{text}\n\n{page_separator}\n\n")
            print(f'Extracted text from: {image_name}')

    return output_file, output_file_name


