from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

app = FastAPI()

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Context:
{context}"""


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def run_prompt(request: PromptRequest):
    embeddings = OpenAIEmbeddings(model="4UHRUIN-text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name="medium-rag-index", embedding=embeddings)

    results = vectorstore.similarity_search_with_score(request.question, k=5)

    context_chunks = []
    formatted_context_list = []

    for doc, score in results:
        authors = doc.metadata.get('authors', 'Unknown')
        title = doc.metadata.get('title', 'Unknown')
        article_id = doc.metadata.get('url', 'Unknown')

        chunk_text = f"Author(s): {authors}\nContent: {doc.page_content}"
        context_chunks.append(chunk_text)

        formatted_context_list.append({
            "article_id": str(article_id),
            "title": str(title),
            "chunk": chunk_text,
            "score": float(score)
        })

    formatted_context = "\n\n".join(context_chunks)
    chat_model = ChatOpenAI(model="4UHRUIN-gpt-5-mini")

    system_content = SYSTEM_PROMPT.format(context=formatted_context)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=request.question)
    ]

    response = chat_model.invoke(messages)

    return {
        "response": response.content,
        "context": formatted_context_list,
        "Augmented_prompt": {
            "System": system_content,
            "User": request.question
        }
    }


@app.get("/api/stats")
def get_stats():
    return {
        "chunk_size": 512,
        "overlap_ratio": 0.1,
        "top_k": 5
    }