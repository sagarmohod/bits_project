from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document
from ModelConfig import GoogleGenerativeAIClient
import uuid


page_separator = "--- Page Separator ---"

google_client = GoogleGenerativeAIClient()
embeddings = google_client.get_embedding()

def create_embeddings_and_store(file_path, doc_name):
    """Reads a file, splits based on page separator, creates embeddings, and stores them in Chroma."""

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split text by page separator
    chunks = text.split(page_separator)

    # Remove leading/trailing whitespace from chunks and filter out empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Create documents from chunks
    docs = [Document(page_content=chunk) for chunk in chunks]
    uuids = [str(uuid.uuid4()) for _ in range(len(docs))]

    # Create Chroma vector store
    vector_store = Chroma(
        collection_name=doc_name,
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally
    )
    vector_store.add_documents(documents=docs, ids=uuids)

