import chainlit as cl
from backend import rag

@cl.on_message
async def on_message(message: cl.Message):
    user_prompt = message.content
    response = rag.run_rag(user_prompt)

    # Send a response back to the user
    await cl.Message(
        content=f"{response}",
    ).send()