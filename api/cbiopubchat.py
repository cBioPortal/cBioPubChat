from fastapi import FastAPI
from pydantic import BaseModel
from backend import run_rag
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/cbiopubchat")
def cbiopubchat_endpoint(request: PromptRequest):
    response = run_rag(request.prompt)
    return {"response": response}