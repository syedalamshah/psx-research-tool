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

    batch_size = 10
    total_batches = max(1, (len(chunks) + batch_size - 1) // batch_size)

    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        print(f"Embedded batch {batch_index + 1}/{total_batches}")

        try:
            vector_store.add_documents(batch)
        except Exception as exc:
            error_text = str(exc).upper()
            is_quota_error = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text
            if not is_quota_error:
                raise

            print(
                f"Batch {batch_index + 1}/{total_batches} hit quota/rate limit. "
                "Waiting 30s and retrying once..."
            )
            time.sleep(5)

            try:
                vector_store.add_documents(batch)
            except Exception as retry_exc:
                retry_error_text = str(retry_exc).upper()
                retry_is_quota_error = (
                    "RESOURCE_EXHAUSTED" in retry_error_text or "429" in retry_error_text
                )
                if retry_is_quota_error:
                    print(
                        f"Batch {batch_index + 1}/{total_batches} failed again after retry; "
                        "skipping this batch."
                    )
                    continue
                raise

        if batch_index < total_batches - 1:
            time.sleep(15)

    return vector_store


if __name__ == "__main__":
    documents = fetch_node("data/engro_annual_report.pdf")
    vector_store = retrieve_node(documents)
    results = vector_store.similarity_search("What are ENGRO's biggest risks?", k=1)

    if results:
        print(results[0].page_content)