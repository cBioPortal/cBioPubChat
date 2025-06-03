import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
else:
    print("OpenAI Key found!")


prompt = PromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks related to cBioPortal publications. Use the following extracted content from cBioPortal publications to answer the question. If you don't know the answer, just say that you don't know.
    
    ---
    Context:
    {context}
    ---
    
    Question:
    {question}
    
    Answer:
    """
)
persist_directory = "./data/vectordb/chroma/pubmed/paper_and_pdf"
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_function
)

print("Number of documents:", len(vector_store._collection.get()['ids']))

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What pathways are altered in breast cancer."})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')