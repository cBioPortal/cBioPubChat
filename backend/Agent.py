from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from backend.State import State

class Agent:
    cbioportal_study_url = "https://www.cbioportal.org/study/summary?id="
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

    def __init__(self, embedding_dir, embedding_model_name, llm_model_name, llm_model_provider, k=10):
        self.embedding_dir = embedding_dir
        self.embedding_fn = OpenAIEmbeddings(model=embedding_model_name)
        self.llm = init_chat_model(llm_model_name, model_provider=llm_model_provider)
        self.vector_store = Chroma(
            persist_directory=embedding_dir,
            embedding_function=self.embedding_fn
        )
        self.k = k

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"], k=self.k)
        return {"context": retrieved_docs}

    def generate(self, state:State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def ask(self, question, return_context=False):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        result = graph.invoke({"question": question})

        if return_context:
            return result["answer"], result["context"]

        return result["answer"]

    def get_studies_from_context(self, context, num_studies=3):
        filtered_metadata = [
            {
                "name": doc.metadata.get("name"),
                "studyId": doc.metadata.get("studyId"),
                "url": self.cbioportal_study_url + doc.metadata.get("studyId")
            }
            for doc in context
        ]

        seen_ids = set()
        unique_studies = []
        for item in filtered_metadata:
            study_id = item.get("studyId")
            if study_id and study_id not in seen_ids:
                seen_ids.add(study_id)
                unique_studies.append(item)
        unique_studies = unique_studies[:num_studies]
        return unique_studies