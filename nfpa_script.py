# %% [markdown]
# # Fire Agent with LlamaIndex
# 
# ## Install Dependencies

# %%
!pip install uv
!uv pip install --system -qU llama-index==0.11.6 llama-index-llms-openai llama-index-readers-file llama-index-embeddings-openai llama-index-llms-openai-like "openinference-instrumentation-llama-index>=2" arize-phoenix python-dotenv

# %% [markdown]
# ## Setup API Keys
# To run the rest of the notebook you will need access to an OctoAI API key. You can sign up for an account [here](https://octoai.cloud/). If you need further guidance you can check OctoAI's [documentation page](https://octo.ai/docs/getting-started/how-to-create-octoai-access-token).

# %%
from os import environ
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

# %% [markdown]
# ## Import libraries and setup LlamaIndex

# %%
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# Create an llm object to use for the QueryEngine and the ReActAgent
llm = OpenAI(model="gpt-4")

# %% [markdown]
# # Set up Phoenix

# %%
import phoenix as px
session = px.launch_app()

# %%
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

tracer_provider = register()
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# %% [markdown]
# ## Load Documents

# %%
try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/nfpa"
    )
    nfpa_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False

# %% [markdown]
# This is the point we create our vector indexes, by calculating the embedding vectors for each of the chunks. You only need to run this once.

# %%
if not index_loaded:
    # load data
    nfpa_docs = SimpleDirectoryReader(
        input_files=["./NFPA10-2022.pdf"]
    ).load_data()

    # build index
    nfpa_index = VectorStoreIndex.from_documents(nfpa_docs, show_progress=True)

    # persist index
    nfpa_index.storage_context.persist(persist_dir="./storage/nfpa")

# %% [markdown]
# Now create the query engines.

# %%
nfpa_engine = nfpa_index.as_query_engine(similarity_top_k=3, llm=llm)

# %% [markdown]
# We can now define the query engines as tools that will be used by the agent.
# 
# As there is a query engine per document we need to also define one tool for each of them.

# %%
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

# %% [markdown]
# ## Creating the Agent
# Now we have all the elements to create a LlamaIndex ReactAgent

# %%
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    max_turns=10,
)

# %% [markdown]
# Now we can interact with the agent and ask a question.

# %%
response = agent.chat("What chemicals are in a 2A fire extinguisher?")
print(str(response))

# %%
!pip install dill


# %%
import dill

with open('react_agent.dill', 'wb') as f:
    dill.dump(agent, f)



