
Python

# agent_tools.py
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.vectorstores import Chroma
from ModelConfig import embeddings, llm  # Assuming ModelConfig.py is in the same directory

def retrieve_documents(query: str, doc_name: str) -> str:
    """Retrieves documents from a vector database based on a query."""
    vectordb = Chroma(persist_directory=f"chroma_db/{doc_name}", embedding_function=embeddings)
    results = vectordb.similarity_search(query)
    return "\n".join([doc.page_content for doc in results])

def generate_summary(text: str) -> str:
    """Generates a summary of the given text."""
    prompt = f"Generate a summary of the following text: {text}"
    response = llm.invoke(prompt)
    return response.content

tools = [
    Tool(
        name="RetrieveDocuments",
        func=lambda x: retrieve_documents(query=x["query"], doc_name=x["doc_name"]),
        description="useful for when you need to retrieve documents from a vector database"
    ),
    Tool(
        name="GenerateSummary",
        func=generate_summary,
        description="useful for when you need to generate a summary of a given text"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)