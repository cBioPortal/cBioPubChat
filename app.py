import chainlit as cl
from backend import rag

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Driver events in Glioma",
            message="What are the driver events in Glioma?",
            icon="/public/icons/brain.svg",
        ),

        cl.Starter(
            label="Pi3K pathway alteration",
            message="How is the Pi3k pathway altered in cancer?",
            icon="/public/icons/dna.svg",
        ),
        cl.Starter(
            label="Cancer types with NTRK fusions",
            message="In which cancer types does one see NTRK fusions?",
            icon="/public/icons/cancer.svg",
        ),
        cl.Starter(
            label="Erlotinib response in lung cancer patients",
            message="Which lung cancer patients respond to Erlotinib?",
            icon="/public/icons/lungs.svg",
        )
    ]

@cl.on_message
async def on_message(message: cl.Message):
    user_prompt = message.content
    response = rag.run_rag(user_prompt)

    # Send a response back to the user
    await cl.Message(
        content=f"{response}",
    ).send()