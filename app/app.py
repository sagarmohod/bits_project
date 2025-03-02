import os
import uuid
from flask import Flask, request, jsonify
from pymongo import MongoClient
import fitz  # PyMuPDF
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from VisionExtractor import pdfs_to_images, extract_text_from_images 
from Embeddings import *
from langchain.chains import RetrievalQA


app = Flask(__name__)

# Load environment variables
load_dotenv()

# MongoDB setup
mongo_client = MongoClient(os.getenv("MONGO_STRING"))  # Replace with your MongoDB URI
db = mongo_client[os.getenv("MONGO_DATABASE")]  # Replace with your database name
fund_store = db["fund_store"]
chat_logs = db["chat_history"]

# Google Generative AI setup (if needed for text extraction)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    doc_name = request.form.get('doc_name') # Changed to doc_name
    doc_type = request.form.get('doc_type')

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    

    if fund_store.find_one({'doc_name': doc_name}):
        return jsonify({'message': f'Document name "{doc_name}" already exists. Please use a different name.'})

    if file:

        unique_id = str(uuid.uuid4())
        input_path = os.path.join("data", unique_id, "input_files")
        image_path = os.path.join("data", unique_id, "images")
        output_path = os.path.join("data", unique_id, "output_files")

        os.makedirs(input_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        file_path = os.path.join(input_path, file.filename) #used file.filename here
        file.save(file_path)

        pdf_name, pdf_count = pdfs_to_images(input_path, image_path)

        output_file, output_file_name = extract_text_from_images(image_path, output_path, pdf_name)

        # Create embeddings and store in Chroma
        vector_db_collection_name = f"{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_embeddings_and_store(output_file, vector_db_collection_name)
        print("**** embeddings generated successfully ****")

        size_in_mb = os.path.getsize(file_path)/(1024 * 1024)
        # Log file metadata to MongoDB
        file_metadata = {
            'doc_name': doc_name, # used doc_name here
            'timestamp': datetime.now(),
            'doc_type': doc_type,
            'input_file': {
                'input_file_name': file.filename, # used file.filename here
                'input_file_location': file_path,
                'file_size': f"File size: {size_in_mb:.2f} MB",
                'file_pages': pdf_count
            },
            'output_file': {
                'output_file_name': output_file_name,
                'output_file_location': output_file
            },
            "vector_db_collection_name" : vector_db_collection_name
        }
        result= fund_store.insert_one(file_metadata)
        file_metadata['_id'] = str(result.inserted_id)

        return jsonify({'message': 'File uploaded and processed successfully', "result": file_metadata})
    else:
        return jsonify({'error': 'Uploading failed'})


@app.route('/get_file_data', methods=['GET'])
def get_file_data():
    # Get all records from MongoDB
    data = list(fund_store.find())

    if not data:
        return jsonify({'message': 'No data found'}), 404

    # Group data by doc_type
    grouped_data = {}
    for item in data:
        doc_type = item.get('doc_type', 'unknown')  # Default to 'unknown' if doc_type is missing
        if doc_type not in grouped_data:
            grouped_data[doc_type] = []
        item['_id'] = str(item['_id'])  # Convert ObjectId to string for JSON serialization
        grouped_data[doc_type].append(item)

    # Prepare the response in the desired format
    result = [{'doc_type': doc_type, 'data': records} for doc_type, records in grouped_data.items()]

    return jsonify(result)


@app.route('/generate_summary', methods=['GET'])
def generate_summary():
    doc_name = request.form.get('doc_name')

    if not doc_name:
        return jsonify({'error': 'Document name not specified'})

    document = fund_store.find_one({'doc_name': doc_name})

    if not document:
        return jsonify({'message': f'Document "{doc_name}" not found.'})

    file_path = document['output_file']['output_file_location']
    doc_type = document['doc_type']

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        prompt = ""
        if doc_type == "mf":
            prompt = f"Generate a summary of the mutual fund document: {text}"
        elif doc_type == "bank":
            prompt = f"Generate a summary of the bank document: {text}"
        elif doc_type == "loan":
            prompt = f"Generate a summary of the loan document: {text}"
        else:
            return jsonify({'error': 'Unsupported document type'})

        llm = google_client.get_llm()
        response = llm.invoke(prompt)
        summary = response.content

        # Create summary file
        summary_path = os.path.dirname(file_path).replace("output_files", "summary_file")
        os.makedirs(summary_path, exist_ok=True)
        summary_file_location = os.path.join(summary_path, "summary.txt")

        with open(summary_file_location, 'w', encoding='utf-8') as summary_file:
            summary_file.write(summary)

        # Update document in MongoDB
        fund_store.update_one(
            {'doc_name': doc_name},
            {'$set': {
                'summary_file': {
                    'summary_file_name': "summary.txt",
                    'summary_file_location': summary_file_location,
                    'summary_creation_time': datetime.now()
                }
            }}
        )

        return jsonify({'summary': summary, 'summary_file_location': summary_file_location})

    except Exception as e:
        return jsonify({'error': f'Error generating summary: {e}'})


@app.route('/ask_question', methods=['POST'])
def ask_question():

    question = request.form.get('question')
    doc_name = request.form.get('doc_name')

    document = fund_store.find_one({'doc_name': doc_name})

    if not document:
        return jsonify({'message': f'Document "{doc_name}" not found.'})

    vector_db_collection_name = document.get('vector_db_collection_name') #Here we are using doc_name as collection name

    if not vector_db_collection_name:
        return jsonify({'error': 'Vector database collection name not found for this document'})

    try:
        # Initialize LLM and embeddings with the provided API key
        llm = google_client.get_llm()
        embeddings = google_client.get_embedding()

        vectordb = Chroma(persist_directory=f"chroma_db/{vector_db_collection_name}", embedding_function=embeddings)
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = qa_chain.invoke(question)

        # Log Q&A
        chat_logs.insert_one({
            'question': question,
            'doc_name': doc_name,
            'vector_db_collection_name': vector_db_collection_name,
            'answer': answer,
            'timestamp': datetime.now(),
            'doc_type': document.get('doc_type')
        })

        return jsonify({'answer': answer})

    except Exception as e:
        return jsonify({'error': f'Error answering question: {e}'})

if __name__ == '__main__':
    app.run(debug=True, port=8080)