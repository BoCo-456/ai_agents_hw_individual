import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

llmod_api_key = open(r'llmod_api_key.txt', 'r')
pinecone_api_key = open(r'./pinecone_api_key.txt', 'r')
os.environ["OPENAI_API_KEY"] = llmod_api_key.read()
os.environ["OPENAI_API_BASE"] = r"https://api.llmod.ai/v1"
os.environ["PINECONE_API_KEY"] = pinecone_api_key.read()


def process_and_upload(file_path: str, index_name: str):
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')

    data = []
    for _, row in df.iterrows():
        text_content = f"Title: {row['title']}\n\n{row['text']}"
        metadata = {
            "url": str(row['url']),
            "authors": str(row['authors']),
            "tags": str(row['tags'])
        }
        data.append(Document(page_content=text_content, metadata=metadata))

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
    num_chunks = process_and_upload("./val.csv", "medium-rag-index")
    print(f"Successfully uploaded {num_chunks} chunks to Pinecone.")
