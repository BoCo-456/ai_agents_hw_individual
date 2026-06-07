import os
import time
from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = open(r'./pinecone_api_key.txt', 'r')
os.environ["PINECONE_API_KEY"] = pinecone_api_key.read()


def create_pinecone_index(index_name: str):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    return pc.Index(index_name)


if __name__ == "__main__":
    index = create_pinecone_index("medium-rag-index")
    print(f"Index is ready.")