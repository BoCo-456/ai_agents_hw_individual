import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

llmod_api_key = open(r'llmod_api_key.txt', 'r')
pinecone_api_key = open(r'./pinecone_api_key.txt', 'r')
os.environ["OPENAI_API_KEY"] = llmod_api_key.read()
os.environ["OPENAI_API_BASE"] = r"https://api.llmod.ai/v1"
os.environ["PINECONE_API_KEY"] = pinecone_api_key.read()

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Context:
{context}"""


def run_pipeline(question: str, top_k: int = 3, index_name: str = "medium-rag-index"):
    embeddings = OpenAIEmbeddings(model="4UHRUIN-text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    retrieved_docs = vectorstore.similarity_search(question, k=top_k)

    context_chunks = []
    formatted_context_list = []

    for i, doc in enumerate(retrieved_docs):
        # Extract authors from Pinecone metadata
        authors = doc.metadata.get('authors', 'Unknown')

        # doc.page_content already contains the Title and the Text from our ingestion script
        chunk_text = f"Author(s): {authors}\nContent: {doc.page_content}"
        context_chunks.append(chunk_text)

        formatted_context_list.append({
            # The assignment JSON requires a title field, but since we didn't put it in
            # metadata during ingestion, we have to parse it out or leave it blank for now.
            # For the final run, you should add 'title' to your ingestion metadata!
            "title": "Title embedded in chunk",
            "chunk": chunk_text
        })

    formatted_context = "\n\n".join(context_chunks)

    chat_model = ChatOpenAI(model="4UHRUIN-gpt-5-mini")

    system_content = SYSTEM_PROMPT.format(context=formatted_context)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]

    response = chat_model.invoke(messages)

    return {
        "response": response.content,
        "context": formatted_context_list,
        "augmented_prompt": {
            "System": system_content,
            "User": question
        }
    }


if __name__ == "__main__":
    test_question = "Name an article that talks about data partitioning, cite the title and authors."
    result = run_pipeline(test_question, top_k=5)

    print("--- MODEL RESPONSE ---")
    print(result["response"])
