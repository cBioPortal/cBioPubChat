import chainlit as cl
from backend.Agent import Agent
from typing import Optional

# Instantiate RAG Agent
rag_agent = Agent(embedding_dir='data/cBioPortal_data_chromadb', embedding_model_name="text-embedding-ada-002", llm_model_name="gpt-4o-mini", llm_model_provider="openai")

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("cbio-user", "cbio-password"):
        return cl.User(
            identifier="cbio-user", metadata = {
                "role": "admin",
                "provider": "credentials"
            }
        )
    else:
        return None

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
    response, context = rag_agent.ask(user_prompt, return_context=True)
    relevant_studies = rag_agent.get_studies_from_context(context)

    # Format the response
    response = response + "\n\nRelevant Studies:\n"
    for doc in relevant_studies:
        response = response + f"* [{doc.get('name')}]({doc.get('url')})\n"

    # Send a response back to the user
    await cl.Message(
        content=f"{response}",
    ).send()