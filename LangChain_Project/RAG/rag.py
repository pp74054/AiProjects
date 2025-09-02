from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_file=["metagpt.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)




