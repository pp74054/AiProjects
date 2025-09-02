from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_file=["metagpt.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

settings.llm = OpenAI(model = "gpt-3.5-turbo")
settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

from llama_index.core import SummaryIndex, VectorStoreIndex
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

from llama_index.core.tools import QueryEngineTool
summary_tool = QueryEngineTool.from_defaults (
    query_engine=summary_query_engine,
    description=("useful for when you need to answer questions about the Metagpt paper."), 
)

vector_tool = QueryEngineTool.from_defaults (
    query_engine=vector_query_engine,
    description = ("useful for when you need to answer questions about the Metagpt paper."),
)

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector

query_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool
    ],
)

response = query_engine.query("What is the summary of the document?")

print(str(response))

print (len(response.source_nodes))

response = query_engine.query(
    "How does Metagpt work?"
)

print(str(response))

from utils import get_router_equery_engine
query_engine = get_router_equery_engine("metagpt.pdf ")



