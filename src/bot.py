import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from langchain_core.messages import HumanMessage

# load .env variables
load_dotenv()

# load web pages
'''
# How to Load Web page
# Link: https://python.langchain.com/docs/how_to/document_loader_web/
'''
url = "https://codebasics.io"
loader = WebBaseLoader(web_paths=[url])
docs = []
for doc in loader.lazy_load():
  docs.append(doc)

# creating a retriever tool
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
vstore = InMemoryVectorStore.from_documents(documents=docs, embedding=embeddings)
retriever = vstore.as_retriever()
my_tool = create_retriever_tool(retriever, "web_page_info_retriever", "Searches for answer in the given website")

# initialize llm
llm = ChatOpenAI()

# initialize memory
memory = MemorySaver()

# define tools
tools = [my_tool]

# bind tools
llm.bind_tools(tools)

 # config
config = {"configurable": {"thread_id": "a1"}}

# inititate agent
agent = create_react_agent(llm, tools, checkpointer=memory)

# call agent
res = agent.invoke({"messages": [HumanMessage(content="hi I am Siam.")]}, config)
print(res["messages"][-1].pretty_print())

res = agent.invoke({"messages": [HumanMessage(content="Total Paid Learners ?")]}, config)
print(res["messages"][-1].pretty_print())