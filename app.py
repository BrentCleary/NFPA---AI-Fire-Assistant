import os
import fitz  # PyMuPDF
import phoenix as px
from phoenix.otel import register
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

import streamlit as st
import numpy as np
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
import re
import logging

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=api_key, model="gpt-4")

# Configure OpenTelemetry for tracing
tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# Create Phoenix session
session = px.launch_app()

# Streamlit setup
st.title("NFPA Regulation Chat Interface")
st.write(
    "This is a chat app to help understand fire extinguisher regulations from the NFPA handbook."
)


# Load PDF
@st.cache_data
# Load PDF
@st.cache_data
def load_pdf(file_path):
    doc = fitz.open(file_path)
    paragraphs = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text(
            "text",
            flags=fitz.TEXT_PRESERVE_IMAGES
            | fitz.TEXT_PRESERVE_LIGATURES
            | fitz.TEXT_PRESERVE_WHITESPACE,
        )
        for paragraph in text.split("\n\n"):
            paragraphs.append({"text": paragraph, "page": page_num})
    return paragraphs


nfpa_text = load_pdf("NFPA-10-2022.pdf")
st.write("PDF Loaded Successfully.")


# Split PDF into paragraphs for more granularity, while excluding non-English text
def split_pdf_into_paragraphs(text):
    paragraphs = text.split("\n\n")
    english_paragraphs = []
    for p in paragraphs:
        try:
            if detect(p) == "en":
                # Adding metadata with section numbers if available
                section_match = re.match(r"^(\d+(\.\d+)*)", p)
                if section_match:
                    section = section_match.group(0)
                    english_paragraphs.append({"text": p, "section": section})
                else:
                    english_paragraphs.append({"text": p, "section": None})
        except LangDetectException:
            continue
    return english_paragraphs


paragraphs = nfpa_text


# Pre-process the paragraphs to remove irrelevant information and clean the text
# Pre-process the paragraphs to remove irrelevant information and clean the text
def preprocess_paragraphs(paragraphs):
    cleaned_paragraphs = []
    for paragraph_dict in paragraphs:
        paragraph = paragraph_dict["text"]
        paragraph = re.sub(r"\s+", " ", paragraph)  # Remove extra whitespace
        paragraph = paragraph.strip()  # Remove leading/trailing whitespace
        if len(paragraph) > 20:  # Keep paragraphs that are long enough to be useful
            cleaned_paragraphs.append(paragraph)
    return cleaned_paragraphs


cleaned_paragraphs = preprocess_paragraphs(paragraphs)


# Load or build the index
@st.cache_data
def load_or_build_index(paragraphs):
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage/nfpa")
        nfpa_index = load_index_from_storage(storage_context)
        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        from llama_index.core import Document

        logging.info(f"Total paragraphs to index: {len(paragraphs)}")
        nfpa_docs = [
            Document(text=p["text"], metadata={"page": p["page"]}) for p in paragraphs
        ]
        nfpa_index = VectorStoreIndex.from_documents(nfpa_docs, show_progress=True)
        nfpa_index.storage_context.persist(persist_dir="./storage/nfpa")

    return nfpa_index


nfpa_index = load_or_build_index(cleaned_paragraphs)

# Setting up the query engine
nfpa_engine = nfpa_index.as_query_engine(similarity_top_k=5, llm=llm)

# Define query engine tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=nfpa_engine,
        metadata=ToolMetadata(
            name="NFPA",
            description=(
                "Provides information about Fire regulations for year 2022. "
                "Use a detailed plain text question as input to the tool. "
            ),
        ),
    )
]

# Creating the Agent
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    max_turns=10,
)

# Streamlit Input for Querying
user_query = st.text_input("Ask about NFPA regulations:")
if user_query:
    # Pre-process query to handle section-specific requests
    section_ref = re.findall(r"\d+(\.\d+)*", user_query)
    if section_ref:
        user_query += f" Please look specifically at section {section_ref[0]} for this information."
    response = agent.chat(user_query)
    # Extracting exact quotes and page numbers from the response
    if response.source_nodes:
        for node in response.source_nodes:
            if node.node and "page" in node.node.extra_info:
                st.write(
                    f"Exact quote from page {node.node.extra_info['page']}: {node.node.text}"
                )
    st.write(response)

# Track session interactions
logging.basicConfig(level=logging.INFO)
logging.info(
    f"User query: {user_query}, Response: {str(response) if user_query else ''}"
)
