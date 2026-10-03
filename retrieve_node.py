import os
import time

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

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        collection_name="psx_research",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    existing_count = vector_store._collection.count()
    if existing_count:
        chunks = chunks[existing_count:]
        print(
            f"Resuming from chunk {existing_count}, "
            f"{len(chunks)} chunks remaining"
        )

    batch_size = 7
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        print(f"Embedded batch {batch_index + 1}/{total_batches}")

        succeeded = False
        for attempt in range(3):
            try:
                vector_store.add_documents(batch)
                succeeded = True
                break
            except Exception as exc:
                if attempt < 2:
                    print(
                        f"Batch {batch_index + 1}/{total_batches} failed. "
                        f"Waiting 15s before retry {attempt + 1}/2..."
                    )
                    time.sleep(15)
                else:
                    print(
                        f"Batch {batch_index + 1}/{total_batches} failed after 2 retries; "
                        f"skipping it: {exc}"
                    )

        if not succeeded:
            continue

        if batch_index < total_batches - 1:
            time.sleep(5)

    return vector_store


if __name__ == "__main__":
    documents = fetch_node("data/engro_annual_report.pdf")
    vector_store = retrieve_node(documents)
    results = vector_store.similarity_search("What are ENGRO's biggest risks?", k=1)

    if results:
        print(results[0].page_content)