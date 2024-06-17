import os
import re
import time

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from openai import OpenAI
import streamlit as st

from chromadb import Documents, EmbeddingFunction, Embeddings


CHROMA_DATA_PATH = r"C:\Users\ihett\OneDrive\Gabrilyi\arxiv_project\chroma_data"
EMBED_MODEL = "all-mpnet-base-v2"
# EMBED_MODEL = "all-MiniLM-L6-v2"
# EMBED_MODEL = "bge-base-en"
COLLECTION_NAME = "arxiv_papers"

CHROMA_DATA_PATH = os.path.join(CHROMA_DATA_PATH, EMBED_MODEL)
class Embedder(EmbeddingFunction):
    def __init__(self):
        self.client = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")
        self.model = "CompendiumLabs/bge-base-en-v1.5-gguf"

    def __call__(self, input:Documents) -> Embeddings:
        return [d.embedding for d in self.client.embeddings.create(input = input, model=self.model).data]


# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL,
    device="cuda",
)

# Streamlit App
st.title("ArXiv Paper Recommender")

# Input text
input_text = st.text_area("Enter text for recommendations", height=200)

# Recommendation settings
num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

# Other options (add as needed)
option1 = st.checkbox("Option 1")
option2 = st.checkbox("Option 2")


def text_processing(sample):
    title = sample["title"]
    abstract = sample["abstract"]

    # remove special characters
    title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    abstract = abstract.replace("\n", " ").replace("\r", " ").replace("\t", " ")

    # remove multiple spaces
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())

    return f"Title: {title} - Abstract: {abstract}"


def create_collection(client, collection_name, embedding_function):
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
        get_or_create=True,
    )

    return collection


def get_random_samples_from_collection(collection, n_samples):
    collection_ids = collection.get()["ids"]
    random_ids = np.random.choice(collection_ids, n_samples, replace=False).tolist()
    documents = collection.get(ids=random_ids)
    return documents


def st_color_same_words(text, words):
    # replace all words in text with the same word but colored (ignoring case)
    # color example:
    # temperature = "-10"
    # f"temprature: :blue[{temperature}]"

    for word in words:
        # just replace if full word match
        text = re.sub(rf"\b{word}\b", f":green[{word}]", text, flags=re.IGNORECASE)
    return text


collection = create_collection(client, COLLECTION_NAME, embedding_func) # Embedder()

if st.button("Get Recommendations"):
    if input_text:
        query_results = collection.query(query_texts=[input_text], n_results=num_recommendations)

        # Display results
        for _id, _doc, _dist, _meta in zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["distances"][0],
            query_results["metadatas"][0],
        ):
            # doc is built like: "Title: ... - Abstract: ..."
            doc_title = re.search("Title: (.*?) - Abstract:", _doc).group(1)
            doc_abstract = re.search("- Abstract: (.*?)$", _doc).group(1)
            doc_category = _meta["super_category"]
            doc_date = _meta["update_date"]

            st.subheader(f"ID: {_id}")
            st.write(f"**Category:** {doc_category}")
            st.write(f"**Date:** {doc_date}")
            st.write(f"**Title:** {st_color_same_words(doc_title, input_text.split())}")
            st.write(f"**Abstract:** {st_color_same_words(doc_abstract, input_text.split())}")
            st.write(f"**Distance:** {_dist}")
            st.write("---")
    else:
        st.error("Please enter some text for recommendations.")
