import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fetch_node import fetch_node


def retrieve_node(documents, persist_directory="chroma_db"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vector_store


if __name__ == "__main__":
    documents = fetch_node("data/engro_annual_report.pdf")
    vector_store = retrieve_node(documents)
    results = vector_store.similarity_search("What are ENGRO's biggest risks?", k=1)

    if results:
        print(results[0].page_content)