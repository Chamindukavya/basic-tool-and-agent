from datetime import datetime
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import os

load_dotenv()

class TimeResponse(BaseModel):
    conditions: str

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY in your .env file")

def get_current_time(_=None):
    """Returns the current time in a human-readable format."""
    current_time = datetime.now().strftime("%I:%M %p %A, %B %d, %Y")
    return f"The current time is {current_time}."

checkpointer = InMemorySaver()


model = init_chat_model(
    "gpt-4.1-nano",
    temperature=0.5
)

agent = create_react_agent(
    tools=[get_current_time],
    model=model,
    prompt="you are a helpfull assistance",
    checkpointer=checkpointer,
    response_format=TimeResponse
)


while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        {"configurable": {"thread_id": "1"}}
    )
    
    print(f"Assistant: {response['structured_response'].conditions}")

