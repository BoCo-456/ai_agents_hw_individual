import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

llmod_api_key = open(r'llmod_api_key.txt', 'r')
pinecone_api_key = open(r'./pinecone_api_key.txt', 'r')
os.environ["OPENAI_API_KEY"] = llmod_api_key.read()
os.environ["OPENAI_API_BASE"] = r"https://api.llmod.ai/v1"
os.environ["PINECONE_API_KEY"] = pinecone_api_key.read()


def process_and_upload(file_path: str, index_name: str):
    loader = CSVLoader(file_path=file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(model="4UHRUIN-text-embedding-3-small")

    PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=embeddings
    )

    return len(docs)


if __name__ == "__main__":
    num_chunks = process_and_upload("val.csv", "your-pinecone-index-name")
    print(f"Successfully uploaded {num_chunks} chunks to Pinecone.")