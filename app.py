import os
import fitz  # PyMuPDF
import openai
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

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
        text += page.get_text()
    return text


nfpa_text = load_pdf("NFPA-10-2022.pdf")
st.write("PDF Loaded Successfully.")


# Split PDF into pages
def split_pdf_into_pages(text):
    return text.split("\n\n")


pages = split_pdf_into_pages(nfpa_text)

# Create TF-IDF Vectorizer and fit to the pages of the PDF
vectorizer = TfidfVectorizer().fit(pages)


# Define a search function that finds the best matching page
def search_handbook(query, pages, vectorizer):
    query_vec = vectorizer.transform([query])
    page_vecs = vectorizer.transform(pages)
    similarity_scores = np.dot(page_vecs, query_vec.T).toarray()
    best_idx = np.argmax(similarity_scores)
    best_score = similarity_scores[best_idx][0]

    THRESHOLD = 0.2  # Set a threshold to identify if the answer is good enough
    if best_score < THRESHOLD:
        return None, None
    return pages[best_idx], best_idx


# Function to ask OpenAI for more detailed answers
def ask_openai(api_key, question, context):
    openai.api_key = api_key
    prompt = f"Use the following NFPA regulations to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant helping with NFPA regulations.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"].strip()


# Integrate with Streamlit
query = st.text_input("Ask a question about NFPA regulations:")
if query:
    context, page_number = search_handbook(query, pages, vectorizer)
    if context:
        answer = ask_openai(api_key, query, context)
        st.write(f"**Answer from Page {page_number + 1}:** {answer}")
    else:
        st.write("I don't know because it's not in the handbook.")
