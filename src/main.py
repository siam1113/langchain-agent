from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,
)
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain import hub
from langchain_core.messages import HumanMessage

# load .env variables
load_dotenv()

# initiate llm with model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# define tools
# playwright
sync_browser = create_sync_playwright_browser()
playwright_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
playwright_tools = playwright_toolkit.get_tools()

# bind tools with llms
tool_binded_llm = llm.bind_tools(playwright_tools)

# ask tool binded llm (respond should be i) general ii) which tool to use)
# res = tool_binded_llm.invoke("Following count of a user from github, username 'siam1113'")
# print(res.content)
# print(res.tool_calls)

# initializing agent
prompt = hub.pull("hwchase17/structured-chat-agent") # ?? learn about promting
agent = create_structured_chat_agent(
    llm,
    playwright_tools,
    prompt
)
agent_executor = AgentExecutor(agent=agent, tools=playwright_tools)

# # input
# response  = agent_executor.invoke({ "input": "Follow count of a user from github, username 'siam1113'"})

# # output
# print(response)

# streaming messages
for chunk in agent_executor.stream(
     {"input": [HumanMessage(content="Follow count of a user from github, username 'siam1113")]}
):
    print(chunk)
    print("----")


