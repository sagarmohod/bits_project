from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

class GoogleGenerativeAIClient:
    def __init__(self):
        # Loading environment variables
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.llm_model_name = os.environ.get("LLM_MODEL_NAME")
        self.embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME")

    def get_llm(self):
        # configuring LLM Model
        llm = ChatGoogleGenerativeAI(
            api_key=self.api_key,
            model=self.llm_model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm

    def get_embedding(self):
        # Configuring Embedding Model
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model=self.embedding_model_name
        )
        return embeddings
