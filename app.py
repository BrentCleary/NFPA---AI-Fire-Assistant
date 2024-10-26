import os
import fitz  # PyMuPDF
import openai
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit setup
st.title("NFPA Regulation Chat Interface")
st.write(
    "This is a chat app to help understand fire extinguisher regulations from the NFPA handbook."
)


# Load PDF
@st.cache_data
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text(
            "text",
            flags=fitz.TEXT_PRESERVE_IMAGES
            | fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_WHITESPACE,
        )
    return text


nfpa_text = load_pdf("NFPA-10-2022.pdf")
st.write("PDF Loaded Successfully.")

# Split PDF into paragraphs for more granularity, while excluding non-English text
import re


def split_pdf_into_paragraphs(text):
    paragraphs = text.split("\n\n")
    english_paragraphs = [
        p for p in paragraphs if detect(p) == "en"
    ]  # Keep only paragraphs that are in English
    return english_paragraphs


paragraphs = split_pdf_into_paragraphs(nfpa_text)


# Pre-process the paragraphs to remove irrelevant information and clean the text
def preprocess_paragraphs(paragraphs):
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph)  # Remove extra whitespace
        paragraph = paragraph.strip()  # Remove leading/trailing whitespace
        if len(paragraph) > 20:  # Keep only paragraphs with substantial content
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs


paragraphs = preprocess_paragraphs(paragraphs)

# Create TF-IDF Vectorizer and fit to the cleaned paragraphs of the PDF
vectorizer = TfidfVectorizer(stop_words="english").fit(paragraphs)


# Define a retrieval function that finds the top matching paragraphs
def retrieve_relevant_paragraphs(query, paragraphs, vectorizer, top_n=3):
    query_vec = vectorizer.transform([query])
    paragraph_vecs = vectorizer.transform(paragraphs)
    similarity_scores = cosine_similarity(query_vec, paragraph_vecs).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    top_paragraphs = [
        paragraphs[i] for i in top_indices if similarity_scores[i] > 0.1
    ]  # Filter to ensure relevance
    return top_paragraphs


# Function to ask OpenAI for more detailed answers
def ask_openai(api_key, question, context):
    openai.api_key = api_key
    # Concatenate top paragraphs to create context (while limiting token size)
    combined_context = "\n".join(context)
    if len(combined_context) > 1500:  # Ensure that context isn't too long
        combined_context = combined_context[:1500] + "..."

    messages = [
        {
            "role": "system",
            "content": "You are an assistant helping with NFPA regulations.",
        },
        {
            "role": "user",
            "content": f"Use the following NFPA regulations to answer the question:\n\n{combined_context}\n\nQuestion: {question}",
        },
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.2
    )
    return response["choices"][0]["message"]["content"].strip()


# Integrate with Streamlit
query = st.text_input("Ask a question about NFPA regulations:")
if query:
    relevant_paragraphs = retrieve_relevant_paragraphs(query, paragraphs, vectorizer)
    if relevant_paragraphs:
        answer = ask_openai(api_key, query, relevant_paragraphs)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("I don't know because it's not in the handbook.")
