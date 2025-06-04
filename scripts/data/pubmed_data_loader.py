# TODO: Remove unnecessary imports

# === Standard Library ===
import json
import argparse
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
from langchain_core.runnables import RunnablePassthrough

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

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


def load_json_file(path):
    """TODO: REVIEW 20250602

    Load and parse a JSON file.

    This function opens a file at the specified path and parses its contents
    as JSON, returning the resulting data structure.

    Parameters
    ----------
    path : str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict or list
        The parsed JSON data structure from the file.
    """
    with open(path) as f:
        data = json.load(f)
    return data


def get_pubmed_chain():
    """TODO: REVIEW 20250602

    Create and return a processing chain for PubMed queries.

    This function constructs a LangChain processing pipeline that takes a question,
    retrieves relevant context from a vector database, and generates a response
    using an LLM. The chain processes the question, retrieves related documents,
    and formats them for the prompt template before passing to the LLM.

    Parameters
    ----------
    None

    Returns
    -------
    RunnableSequence
        A LangChain runnable sequence that can be invoked with a question.
    """
    # Create the chain with proper pipe operators
    chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt  # Choose a prompt
        | llm  # Choose a LLM
        | StrOutputParser()
    )

    return chain


# Chat prediction logic
def predict(question, history):
    """TODO: REVIEW 20250602

    Process a user message and generate a response using chat history.

    This function formats the chat history into LangChain message format
    and sends the current user message along with history context to the
    PubMed response generation pipeline.

    Parameters
    ----------
    message : str
        The current user message to be processed.
    history : list
        List of tuples containing (user_message, bot_response) pairs from
        previous interactions.

    Returns
    -------
    str
        The generated response from the language model.
    """
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=question))

    chain = get_pubmed_chain()
    gpt_response = chain.invoke(question)

    return gpt_response


def update_vectordb_with_docs(docs, embeddings, base_persist_directory):
    """TODO: REVIEW 20250602

    Update a vector database with new document embeddings.

    This function takes a list of documents, creates embeddings for them in batches,
    and adds them to an existing Chroma vector database. The function processes
    documents in small batches to manage memory usage.

    Parameters
    ----------
    docs : List[Document]
        List of Document objects to add to the vector database.
    embeddings : Embeddings
        The embedding model to use for creating document embeddings.
    base_persist_directory : str
        Directory path where the Chroma database is stored.

    Returns
    -------
    Chroma
        The updated Chroma vector database instance.
    """
    # A DB created for first part of DB
    vectordb_total = Chroma(persist_directory=base_persist_directory, embedding_function=embeddings)

    # Iterate through the documents and update the vector DB
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


class PubmedLoader(BaseLoader):
    def __init__(
        self,
        file_path: Union[str, Path],

    ):
        self.file_path = Path(file_path).resolve()

    def load(self) -> List[Document]:
        """TODO: REVIEW 20250602

        Load and return documents from the pubmed file.

        This function reads a PubMed file, extracts its content and metadata,
        splits the text into manageable chunks, and returns a list of document
        objects with appropriate metadata attached.

        Parameters
        ----------
        None

        Returns
        -------
        List[Document]
            A list of Document objects containing chunks of the PubMed article
            with associated metadata.
        """
        docs: List[Document] = []

        with open(self.file_path, encoding="utf-8") as f:
            pubmed_content = f.read()
        # extract paper title
        with open(self.file_path, encoding="utf-8") as file:
            title = file.readline().strip()

        # Define separators for text splitting
        separators = [
            "\n", "Results", "Discussion", "Method Summary",
            "Tissue Source Sites", "Supplementary Material"
        ]
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            separators=separators
        )
        pubmed_splits = text_splitter.split_text(pubmed_content)

        # Convert string to doc, otherwise it cannot have metaData
        pubmed_splits_doc = text_splitter.create_documents(pubmed_splits)
        for split in pubmed_splits_doc:
            docs.append(split)

        # Extract pmcid from file name, and get pmid & study content
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

        # Add metadata
        for doc in docs:
            doc.metadata['pmc_id'] = pmcid
            doc.metadata['paper_title'] = title
            for k, v in study.items():
                doc.metadata[k] = v
            doc.metadata['pmid'] = pmid
        return docs


def load_docs(dir, loader):
    """TODO: REVIEW 20250602

    Load documents from a directory using the specified loader.

    This function uses the DirectoryLoader to load all files in a specified directory
    using the provided loader class. It prints the number of documents loaded
    and saves a text representation of the documents to a file.

    Parameters
    ----------
    dir : str
        The directory path containing documents to load.
    loader : BaseLoader
        The document loader class to use for loading the documents.

    Returns
    -------
    List[Document]
        A list of loaded Document objects.
    """
    loader = DirectoryLoader(path=dir, loader_cls=loader, show_progress=True)
    docs = loader.load()
    print(len(docs))

    # TODO: 20250603 Keep debugging here?
    #with open(f"{dir}.txt", 'w') as f:
    #    f.write(str(docs))

    return docs


# TODO: 20250603 Both needed?
load_dotenv()
_ = load_dotenv(find_dotenv())  # read local .env file

# SETUP LLM AND EMBEDDING MODELS ----
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

# READ METADATA ----
# Load publication ID information
# TODO: 20250603 Generated by GSOC getStudy.py file
pmid_data = load_json_file('data/data_raw/pmcid_list.json')

# Contains study metadata; from cBioPortal api/studies API
study_data = load_json_file('data/data_raw/studies.json')
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

# SETUP VECTORDB
# Load DB
vectordb_total = Chroma(persist_directory="data/data_chromadb", embedding_function=embeddings)

# Setup Retriever
retriever = vectordb_total.as_retriever(k=3)

# Build prompt template
ANSWER_PROMPT = """
Answer question given the content & metadata and chat history if needed.
Return the pmc_id, pmid, studyID of all the contexts: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)


if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="PubMed Document Processing and RAG Query System")
    parser.add_argument('--test', '-t', action='store_true', help='Run test query on the system')
    parser.add_argument('--load', '-l', metavar='DIR', nargs='?', const='data/data_raw/txt',
                        help='Load documents from directory (default: data/data_raw/txt)')

    args = parser.parse_args()

    # Process arguments
    if args.test:
        # Run test query
        history = []
        user_input = "Human: Give me an example of a breast cancer biomarker?"
        response = predict(user_input, history)
        print("AI:", response)
        history.append((user_input, response))

    if args.load:
        # Load documents from specified directory
        directory = args.load
        print(f"Loading documents from {directory}...")
        docs = load_docs(directory, PubmedLoader)
        print(f"Loaded {len(docs)} document chunks")

        # Update vector database with loaded documents
        persist_dir = "data/data_chromadb"
        print(f"Updating vector database at {persist_dir}...")
        update_vectordb_with_docs(docs, embeddings, persist_dir)
        print("Database update complete")

    # Show help if no arguments provided (argparse handles -h/--help automatically)
    if not args.test and not args.load:
        parser.print_help()
