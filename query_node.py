import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from fetch_node import fetch_node
from retrieve_node import retrieve_node


def query_node(vector_store, question):
    documents = vector_store.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(document.page_content for document in documents)
    print("Retrieved context:\n", context[:2000])

    prompt = f"""
Answer the question using ONLY the provided context.
If the context does not contain enough information to answer the question, say
"I don't have enough information".

Context:
{context}

Question: {question}
"""

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
    response = llm.invoke(prompt)

    sources = [
        document.metadata.get("page", document.metadata)
        for document in documents
    ]
    answer = (
        "".join(item.get("text", "") for item in response.content)
        if isinstance(response.content, list)
        else response.content
    )
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = Chroma(
        collection_name="psx_research",
        embedding_function=embeddings,
        persist_directory="chroma_db",
    )

    result = query_node(vector_store, "What are ENGRO's biggest risks?")

    print("Answer:")
    print(result["answer"])
    print("Sources:")
    print(result["sources"])