import streamlit as st
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF


def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.title("NFPA Regulation Chat Interface")
st.write(
    "This is a chat app to help understand fire extinguisher regulations from the NFPA handbook."
)
nfpa_text = load_pdf("NFPA-10-2022.pdf")
st.write("PDF Loaded Successfully.")
