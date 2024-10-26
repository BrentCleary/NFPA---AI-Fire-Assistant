import os
import fitz  # PyMuPDF
from openai import OpenAI

import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

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
    for page_num, page in enumerate(doc, start=1):
        text += f"\n\n[Page {page_num}]\n"
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
    english_paragraphs = []
    for p in paragraphs:
        try:
            if detect(p) == "en":
                english_paragraphs.append(p)
        except LangDetectException:
            continue
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
        (paragraphs[i], i) for i in top_indices if similarity_scores[i] > 0.1
    ]  # Filter to ensure relevance
    return top_paragraphs


# Function to ask OpenAI for more detailed answers
def ask_openai(api_key, question, context_with_indices):
    # Concatenate top paragraphs to create context (while limiting token size)
    combined_context = "\n".join(
        [
            f"{context} (from Page {index + 1})"
            for context, index in context_with_indices
        ]
    )
    if len(combined_context) > 1500:  # Ensure that context isn't too long
        combined_context = combined_context[:1500] + "..."

    messages = [
        {
            "role": "system",
            "content": "You are an assistant helping with NFPA regulations. Only provide information that directly references the handbook, including the exact quote and the page number. If the information is not found in the provided context, respond that it is not available.",
        },
        {
            "role": "user",
            "content": f"Use the following NFPA regulations to answer the question:\n\n{combined_context}\n\nQuestion: {question}",
        },
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.2
    )
    return response.choices[0].message.content.strip()


# Integrate with Streamlit
query = st.text_input("Ask a question about NFPA regulations:")
if query:
    relevant_paragraphs_with_indices = retrieve_relevant_paragraphs(
        query, paragraphs, vectorizer
    )
    if relevant_paragraphs_with_indices:
        answer = ask_openai(api_key, query, relevant_paragraphs_with_indices)
        st.write(f"**Answer:** {answer}")
    else:
        st.write("I don't know because it's not in the handbook.")
