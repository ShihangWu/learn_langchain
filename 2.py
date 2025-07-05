from dotenv import load_dotenv
load_dotenv()  # 自动加载 .env 文件
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_core.messages import HumanMessage

model.invoke([HumanMessage(content="Hi! I'm Bob")])

# ...existing code...
# result_1 = model.invoke([HumanMessage(content="Hi! I'm Bob")])
# print(result_1)
#
# model.invoke([HumanMessage(content="What's my name?")])
# result_2 = model.invoke([HumanMessage(content="What's my name?")])
# print(result_2)

from langchain_core.messages import HumanMessage, AIMessage
# 假设 model 已初始化

# messages = [
#     HumanMessage(content="Hi! I'm Bob"),
#     AIMessage(content="Hello Bob! How can I assist you today?"),
#     HumanMessage(content="What's my name?"),
# ]
#
# response = model.invoke(messages)
# print(response.content)  # 仅文本
# print("-" * 40)
# print(response)          # 对象的repr格式

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc234"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
# output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc123"}}

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


  # # Async function for node:
  # async def call_model(state: MessagesState):
  #     response = await model.ainvoke(state["messages"])
  #     return {"messages": response}
  #
  #
  # # Define graph as before:
  # workflow = StateGraph(state_schema=MessagesState)
  # workflow.add_edge(START, "model")
  # workflow.add_node("model", call_model)
  # app = workflow.compile(checkpointer=MemorySaver())
  #
  # # Async invocation:
  # output = await app.ainvoke({"messages": input_messages}, config)
  # output["messages"][-1].pretty_print()
