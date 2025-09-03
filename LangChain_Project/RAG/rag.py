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

response = query_engine.query("Tell me about Metagpt.")
print(str(response))

from helper import gen_openai_api_key
OPENAI_API_KEY = gen_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

from llama_index.core.tools import FunctionTool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def mystery_function(x: int, y: int) -> int:
    """Multiply two numbers."""
    return (x + y) * (x + y)

add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")
response = llm.chat(
    "What is the sum of 3 and 5? ",
    verbose=True
)

print(str(response))

from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_file=["metagpt.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

print(nodes[0].get_content(metadata="all"))

from llama_index.core import VectorStoreIndex
query_engine = vector_index.as_query_engine(similarity_top_k=2)
    


from llama_index.core.vector_stores import MetadataFilter
query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filter=MetadataFilter.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)
response = query_engine.query("What is Metagpt?")

print(str(response))


for n in response.source_nodes:
    print(n.metadata)
    
from typing import List
from llama_index.core.vector_stores import FilterCondition
def vector_query(
    query: str,
    page_numbers: List[str]
) -> str:
    """Perform a vector query with metadata filtering."""
    
    metedata_dicts = [
        {"key": "page_label", "value": page_number} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilter.from_dicts(
            metedata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response

vector_query_tool = FunctionTool.from_defaults(
    name= "vector_query",
    fn=vector_query
    )

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.predict_and_call(
    "What is Metagpt? Use the vector_query function to find the answer.",
    tools=[vector_query_tool],
    verbose=True
)

for n in response.source_nodes:
    print(n.metadata)
    
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool
summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults (
    name="summary_tool",
    query_engine=summary_query_engine,
    description=("useful for when you need to answer questions about the Metagpt paper."),
)

response = llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "What is Metagpt? Use the vector_query function to find the answer. If you cannot find the answer, use the summary_tool to find the answer.",
    verbose=True
)

for n in response.source_nodes:
    print(n.metadata)
    
response = llm.predict_and_call(
    [vector_query_tool, summary_tool],
    "what is the summary of the Metagpt paper? Use the vector_query function to find the answer. If you cannot find the answer, use the summary_tool to find the answer.",
    verbose=True
)

