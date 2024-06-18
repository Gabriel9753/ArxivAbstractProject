import os
import re
import time

import chromadb
import numpy as np
import streamlit as st
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from openai import OpenAI

# CHROMA_DATA_PATH = r"C:\Users\ihett\OneDrive\Gabrilyi\arxiv_project\chroma_data"
CHROMA_DATA_PATH = "src/chroma_data"
EMBED_MODEL = "all-MiniLM-L12-v2"
# EMBED_MODEL = "all-mpnet-base-v2"
# EMBED_MODEL = "allenai-specter" # https://huggingface.co/sentence-transformers/allenai-specter
# EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
COLLECTION_NAME = "arxiv_papers"
CHROMA_DATA_PATH = os.path.join(CHROMA_DATA_PATH, EMBED_MODEL)
CACHE_DIR = "src/cache"

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL, device="cuda", cache_folder=CACHE_DIR
)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# --- Language Model ---

system_prompt = """
You are a chatbot assistant for arXiv papers. You are asked to provide a short summary of the paper based on the title and abstract.
Write in a friendly and informative tone. The summary should be concise and informative.
You can add additional useful information if needed.
Never mention that you are a chatbot or this is an AI-generated text. Just provide the information as if you were a human.
"""
user_prompt = """
This paper is from {} in the category {}.\nThe title is: {}\nThe abstract is: {}
"""
llm = OpenAI(base_url="http://localhost:5000/v1", api_key="lm-studio")

# Streamlit App
st.title("ArXiv Paper Recommender")

# Input text
input_text = st.text_area("Enter text for recommendations", height=200)

# Recommendation settings
num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)


def st_color_same_words(text, words):
    # replace all words in text with the same word but colored (ignoring case)
    # color example:
    # temperature = "-10"
    # f"temprature: :blue[{temperature}]"

    for word in words:
        # just replace if full word match
        text = re.sub(rf"\b{word}\b", f":blue[{word}]", text, flags=re.IGNORECASE)
    return text


def get_llm_summary(title, abstract, category, doc_date):
    # Generate a summary using the language model
    prompt = user_prompt.format(doc_date, category, title, abstract)
    completion = llm.chat.completions.create(
        model="lmstudio-ai/gemma-2b-it-GGUF",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return completion.choices[0].message.content


def format_arxiv_id(arxiv_id):
    # format arxiv id to have a consistent length of dddd:dddd
    # example: 2103.213 -> 2103.0213
    try:
        parts = arxiv_id.split(".")
        return f"{parts[0]:0>4}.{parts[1]:0>4}"
    except Exception as e:
        return arxiv_id


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
            # doc is built like: "... [SEP]"
            _id = format_arxiv_id(_id)
            doc = _doc.split(" [SEP] ")
            doc_title = doc[0].strip()
            doc_abstract = doc[1].strip()
            doc_category = _meta["super_category"]
            doc_date = _meta["update_date"]

            # st.subheader(f"ID: {_id}")
            # st.write(f"**URL:** [PDF Link](https://arxiv.org/pdf/{_id}.pdf)")
            st.markdown(f"### [{st_color_same_words(doc_title, input_text.split())}](https://arxiv.org/pdf/{_id}.pdf)")
            st.write(f"**Distance:** {_dist}")
            st.write(f"**Category:** {doc_category}")
            st.write(f"**Date:** {doc_date}")
            st.write(f"**Abstract:** {st_color_same_words(doc_abstract, input_text.split())}")
            st.write("")
            st.markdown(
                f'<div style="font-family: Arial; background-color: #3F4A99; padding: 10px; border-radius: 5px;">'
                f"{get_llm_summary(doc_title, doc_abstract, doc_category, doc_date)}"
                "</div>",
                unsafe_allow_html=True,
            )
            st.write("---")
    else:
        st.error("Please enter some text for recommendations.")
