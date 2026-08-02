from langchain_community.document_loaders import PyPDFLoader


def fetch_node(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


if __name__ == "__main__":
    documents = fetch_node("data/engro_annual_report.pdf")
    print(f"Loaded {len(documents)} pages")
