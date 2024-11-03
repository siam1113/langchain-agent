from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import create_react_agent
from dotenv import load_dotenv
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

# load .env variables
load_dotenv()

# initiate the llm
llm = ChatOpenAI(model="gpt-3.5-turbo")

# define tools
# NOTE: .bind_tools is required when you want to the llm to do tool calling
tools = [multiply]

# inititate the memory make the model 'stateful'
memory = MemorySaver()

# create agent
agent = create_react_agent(llm, tools, checkpointer=memory)

# config
config = {"configurable": {"thread_id": "abc123"}}

# ask question
response = agent.invoke({ "messages": """My Intro: 👋 This is Siam Hasan an aspiring entrepreneur with a strong passion for programming & technology.

💫 Enthusiastic about learning and exploring new cutting-edge tools and technologies to create digital solutions.

🎯 Focused on delivering high quality solutions at fast pace having the ability to contribute in all stages of process.

💼 Working as SQA Engineer, Test Automation Engineer with multiple startups around the globe.

💬 Like to talk about anything related to Technology , Business , Startup and Entrepreneurship.""" }, config)
print(response)

response = agent.invoke({ "messages": "What's my current working role ?" }, config)
print(response)
