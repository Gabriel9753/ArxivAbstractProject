import os

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

CHROMA_DATA_PATH = "src/chroma_data"
CACHE_DIR = "src/cache"
EMBED_MODEL = "all-mpnet-base-v2"
COLLECTION_NAME = "arxiv_papers"
CHROMA_DATA_PATH = os.path.join(CHROMA_DATA_PATH, EMBED_MODEL)

# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
#     model_name=EMBED_MODEL, device="cuda", cache_folder=CACHE_DIR
# )


class Embedder:
    def __init__(self):
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL, device="cuda", cache_folder=CACHE_DIR
        )

    # def embed_documents(self, input):
    #     return [d.embedding for d in self.embedding_func.embed_documents(input)]

    def embed_query(self, query: str):
        return self.embedding_func(query)


def get_collection():
    assert os.path.exists(CHROMA_DATA_PATH), f"Chroma vectorstore path {CHROMA_DATA_PATH} does not exist"
    # client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    # return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)


if __name__ == "__main__":
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=Embedder(),
        persist_directory=CHROMA_DATA_PATH,
        create_collection_if_not_exists=False,
    )
    retriever = vectorstore.as_retriever()
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": "Give me a summary of the article."})
    print(response["answer"])

    from langchain.tools.retriever import create_retriever_tool

    tool = create_retriever_tool(
        retriever,
        "blog_post_retriever",
        "Searches and returns excerpts from the Autonomous Agents blog post.",
    )
    tools = [tool]
