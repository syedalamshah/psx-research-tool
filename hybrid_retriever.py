import os

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fetch_node import fetch_node
from retrieve_node import retrieve_node


def build_hybrid_retriever(documents, vector_store):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 3

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


if __name__ == "__main__":
    documents = fetch_node("data/engro_annual_report.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        collection_name="psx_research",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    retriever = build_hybrid_retriever(chunks, vector_store)
    query = "What are ENGRO's biggest risks?"
    results = retriever.invoke(query)

    print("Results:")
    for result in results:
        print(result.page_content)
        print("\n---\n")
