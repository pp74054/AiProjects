# Update the import path if helpers.py is in a different directory, for example:
# from Agents.helpers import get_openai_api_key
from .helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

from .utils import get_doc_tools
vetcor_tool, summary_tool = get_doc_tools("metagpt.pdf, metagpt")

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

from llama_index.core.agent import Agent, FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    llm=llm,
    vetcor=True
)
Agent = AgentRunner(agent_worker)

response = Agent.quary("Tell me about Metagpt.")

print (response.source_nodes[0].get_content(metadata_mode="all"))

response = Agent.chat("How does Metagpt work?")

response = Agent.chat("tell me the results over one of the above datasets.")

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vetcor_tool, summary_tool],
    llm=llm,
    vetcor=True
)
Agent = AgentRunner(agent_worker)

task = Agent.create_task(
    "tell me about the agent roles in MetaGPT."
    "and then how they communicate with each other?"
)

step_output = Agent.run_step(task.task_id)

complete_steps = Agent.get_completed_steps(task.task_id)
print(f"Num completed for task: {task.task_id}: {len(complete_steps)}")
print(complete_steps[0].output.sources[0].raw_output)

upcoming_steps = Agent.get_upcoming_steps(task.task_id)
print (f"Num upcoming for task: {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

step_output = Agent.run_step(task.task_id, input = "What about how agents share information?")

step_output = Agent.run_step(task.task_id)
print(step_output.is_last)

response = Agent.finalize_response(task.task_id)
print (str(response))

from .helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/pdf?id=H1eA5A2tPB",
    "https://openreview.net/pdf?id=rJ4km2EKwH",
    "https://openreview.net/pdf?id=rye9n2VtDB",
    
] 

papers = [
    "metagpt.pdf",
    "longchat.pdf",
    "selfrag.pdf"
]

from utils import get_doc_tools
from pathlib import Path
paper_to_tools_dict = {}
for paper in papers:
    print (f"Loading tools for {paper}")
    vetcor_tool, summary_tool = get_doc_tools(paper, str(Path(paper).stem))
    paper_to_tools_dict[paper] = [vetcor_tool, summary_tool]
    

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI 
llm = OpenAI(model="gpt-3.5-turbo")

len(initial_tools)


 


