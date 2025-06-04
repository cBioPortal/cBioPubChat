import getpass
import json
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def run_rag(user_prompt: str):
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    prompt = PromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks related to cBioPortal publications. Use the following context from cBioPortal publications to answer the question. If you don't know the answer, just say that you don't know. In your response, don't mention the word 'context' or refer to the context explicitly. Provide a concise answer.
        
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

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": user_prompt})

    cbioportal_study_url = "https://www.cbioportal.org/study/summary?id="
    filtered_metadata = [
        {
            "name": doc.metadata.get("name"),
            "studyId": doc.metadata.get("studyId"),
            "url": cbioportal_study_url + doc.metadata.get("studyId")
        }
        for doc in result["context"]
    ]
    seen_ids = set()
    unique_studies = []
    for item in filtered_metadata:
        study_id = item.get("studyId")
        if study_id and study_id not in seen_ids:
            seen_ids.add(study_id)
            unique_studies.append(item)

    result = result["answer"] + "\n\n"
    result = result + "Citations:\n"
    for doc in unique_studies:
        result = result + f"* [{doc.get('name')}]({doc.get('url')})\n"

    return result