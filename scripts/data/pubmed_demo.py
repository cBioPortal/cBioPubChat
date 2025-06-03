# === Standard Library ===
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union
#import time
#from typing import Dict, Iterator,

# === Third-Party Libraries ===
from dotenv import load_dotenv, find_dotenv
#import chromadb

# === LangChain Core ===
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# === LangChain OpenAI API ===
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# === LangChain Community ===
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma

# === LangChain Standard Modules ===
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.docstore.document import Document as DocstoreDocument
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.schema.document import Document as SchemaDocument
# from langchain.text_splitter import CharacterTextSplitter

from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


def load_json_file(path):
    with open(path) as f:
        data = json.load(f)
    return data


pmid_data = load_json_file('../../data/pmcid_list.json')
study_data = load_json_file('../../data/data_raw.json')
pmid_dict = {}
study_dict = defaultdict(list)

# Pair PMID and PMCID in dict
for data in pmid_data:
    pmcid = data.get('pmcid')
    if pmcid:
        pmid = data.get('pmid')
        pmid_dict[pmcid] = pmid


for study in study_data:
    pmid = study.get('pmid')
    if pmid:
        pmid = pmid.strip()
        if len(pmid) > 8:  # more than one pmid
            curr = pmid.split(',')
            for id in curr:
                study_dict[id].append(study)
        else:
            study_dict[pmid].append(study)


class pubmedLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],

    ):
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:
        """Load and return documents from the pubmed file."""
        docs: List[Document] = []

        with open(self.file_path, encoding="utf-8") as f:
            pubmed_content = f.read()
        # extract paper title
        with open(self.file_path, encoding="utf-8") as file:
            title = file.readline().strip()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            separators=["\n", "Results", "Discussion", "Method Summary", "Tissue Source Sites", "Supplementary Material"]
        )
        pubmed_splits = text_splitter.split_text(pubmed_content)
        #convert string to doc, otherwise it cannot have metaData
        pubmed_splits_doc = text_splitter.create_documents(pubmed_splits)
        for split in pubmed_splits_doc:
            docs.append(split)

        #extract pmcid from file name, and get pmid & study content
        pmcid = Path(self.file_path).stem
        pmid = pmid_dict[pmcid]
        study = {}
        if pmid in study_dict:
            study = study_dict[pmid]
        if len(study) > 1:  # for one paper used in multi-study
            i = 2
            # Process each study and store it in the study_dict
            content_dict = {}
            for s in study:
                for key, value in s.items():
                    if key in content_dict:
                        content_dict[(key + str(i))] = value
                    else:
                        content_dict[key] = value
                i += 1
            study_dict[pmid] = content_dict
            study = study_dict[pmid]
        else:
            study = study[0]

        # add metaData
        for doc in docs:
            doc.metadata['pmc_id'] = pmcid
            doc.metadata['paper_title'] = title
            for k, v in study.items():
                doc.metadata[k] = v
            doc.metadata['pmid'] = pmid
        return docs


def load_docs(dir, loader):
    loader = DirectoryLoader(path=dir, loader_cls=loader, show_progress=True)
    docs = loader.load()
    print(len(docs))
    with open(f"{dir}.txt", 'w') as f:
        f.write(str(docs))
    return docs


# docs = load_docs('demo/loaded_pmc', pubmedLoader)
# print(len(docs))

load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

# LLM
llm_openai = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
)

# Embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-ada-002"
)

# # first part of embeddings
# vectordb_total = Chroma.from_documents(
#     documents=docs[:3],
#     embedding=embeddings,
#     persist_directory="vectordb/chroma/pubmed/pdf2"
# )
# vectordb_total.persist()


def update_vectordb_with_docs(docs, embeddings, base_persist_directory):
    # the db created for first part of db
    vectordb_total = Chroma(persist_directory=base_persist_directory, embedding_function=embeddings)

    # Iterate through the documents and update the vectordb
    for i in range(int(len(docs) / 3)):
        start, end = i * 3, (i + 1) * 3
        docs_to_embed = docs[start:end]
        vectordb_new = Chroma.from_documents(
            documents=docs_to_embed,
            embedding=embeddings,

        )

        new_data = vectordb_new._collection.get(include=['documents', 'metadatas', 'embeddings'])
        vectordb_total._collection.add(
            embeddings=new_data['embeddings'],
            metadatas=new_data['metadatas'],
            documents=new_data['documents'],
            ids=new_data['ids']
        )
        print(vectordb_total._collection.count())
    return vectordb_total


vectordb_total = Chroma(persist_directory="data/data_chromadb", embedding_function=embeddings)

retriever = vectordb_total.as_retriever(k=3)

# build prompt template
ANSWER_PROMPT = ("You are a professional assistant."
                 "Answer questions content & metadata, and chat history if needed."
                 "Below is a set of related Q&A examples that includes both good and bad examples. For each example:"
                 "If it is marked as a 'Good example,' you may refer the conversation. Sometimes user can give important info"
                 "If it is marked as a 'Bad example,' improve the answer to better meet the user's needs."
                 "Also, ignore the context if it is a reference."
                 "return the pmc_id, pmid, studyID of all the contexts :"
"{related_QA}"
"{context}"
"Question: {question}"
)


prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", ANSWER_PROMPT),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )

# history_aware_retriever = create_history_aware_retriever(llm_azure, retriever, qa_prompt)


def get_pubmed_chain():
    chain = (
        RunnableLambda(lambda x: x['question']) |
        {"related_QA": RunnablePassthrough(), "context": retriever, "question": RunnablePassthrough()}
        | prompt  # choose a prompt
        | llm_openai  # choose a llm
        | StrOutputParser()
    )
    # question_answer_chain = create_stuff_documents_chain(llm_azure, qa_prompt)
    # final_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return chain


def main():
    # Get response using the chain
    def get_response(question):
        chain = get_pubmed_chain()
        ans = chain.invoke(question)
        return ans

    # Chat prediction logic
    def predict(message, history):
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        history_langchain_format.append(HumanMessage(content=message))

        gpt_response = get_response(message)
        return gpt_response

    print("=== Welcome to the cBioPortal PubMed ChatBot ===")
    print("Type 'exit' to end the session.\n")

    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = predict(user_input, history)
        print("Bot:", response)
        history.append((user_input, response))


if __name__ == '__main__':
    main()
