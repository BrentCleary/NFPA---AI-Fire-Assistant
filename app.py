import os
import fitz  # PyMuPDF
from openai import OpenAI
import arize.phoenix as arize
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, QueryEngine
from openinference.instrumentation import OpenInferenceInstrumentation

import streamlit as st
import numpy as np
from dotenv import load_dotenv
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


# Create and index the NFPA document using LlamaIndex
@st.cache_data
def create_llama_index(paragraphs):
    reader = SimpleDirectoryReader(input_text=paragraphs)
    index = GPTVectorStoreIndex.from_documents(reader.load_data())
    return index


index = create_llama_index(paragraphs)

# Setup Query Engine with LlamaIndex
query_engine = QueryEngine(index=index)


# Define a retrieval function using LlamaIndex
@st.cache_data
def retrieve_relevant_context(query, query_engine, top_n=3):
    response = query_engine.query(query, top_k=top_n)
    results = [(item.text, item.metadata.get("page", "unknown")) for item in response]
    return results


# Setup Arize Phoenix for monitoring
arize.init(project_name="NFPA_Chat_App", api_key=api_key)

# Setup openinference instrumentation
instrumentation = OpenInferenceInstrumentation()


# Function to ask OpenAI for more detailed answers
def ask_openai(api_key, question, context_with_indices):
    # Concatenate top paragraphs to create context (while limiting token size)
    combined_context = "\n".join(
        [f"{context} (from Page {index})" for context, index in context_with_indices]
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
    # Retrieve context using LlamaIndex
    relevant_paragraphs_with_indices = retrieve_relevant_context(query, query_engine)
    if relevant_paragraphs_with_indices:
        # Instrumentation start for monitoring
        instrumentation.start()
        answer = ask_openai(api_key, query, relevant_paragraphs_with_indices)
        # Instrumentation end for monitoring
        instrumentation.end()
        st.write(f"**Answer:** {answer}")
        # Log to Arize for monitoring purposes
        arize.log_response(query, answer, relevant_paragraphs_with_indices)
    else:
        st.write("I don't know because it's not in the handbook.")
