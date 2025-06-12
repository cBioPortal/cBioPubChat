# 🧬 cBioPubChat

**Ask questions. Explore cancer publications. Discover insights.**

> ⚠️ _This project is a **work in progress** being developed for the **cBioPortal Hackathon 2025**._

cBioPubChat is an AI-powered chatbot designed to help researchers, clinicians, and enthusiasts interact with publications from cBioPortal studies. By combining vector search with large language models, the chatbot can:

- Retrieve the most relevant studies based on a user question
- Summarize the key findings from those studies
- Provide direct links to the studies in [cBioPortal](https://www.cbioportal.org/)

## Usage

### Environment setup
```shell
# Run init script to setup a python venv and install requirements
source scripts/init_env.sh

# Everytime you start your ide/terminal, source the venv to use correct dependencies
source venv/bin/activate

# Download the embeddings
./scripts/init_db.sh
```

### Run test script
To test your environment, export your OpenAI API Key and run the test script.
```shell
py tests/env.py

# If all is successful, the test script should put a smile on your face!
```

### Start app
```shell
chainlit run app.py -w
```

## Sample Use Case

> _“Which pathways are most commonly altered in ovarian cancer?”_

cBioPubChat will:
- Search all study publication text using embedding similarity
- Summarize relevant findings with an LLM
- Provide links to those studies in cBioPortal

## Planned Tech Stack

- **[Chainlit](https://github.com/Chainlit/chainlit)** – Interactive chat UI
- **[LangChain](https://www.langchain.com/)** – LLM pipeline & orchestration
- **[ChromaDB](https://www.trychroma.com/)** – Vector store for publication embeddings
- **Python** 3.10+
- **LLMs** – OpenAI or other LangChain-compatible providers

## Planned Project Structure
```shell
cBioPubChat/
├── app/                          # Chainlit app frontend and config
│   ├── main.py                   # Chainlit entrypoint (UI + LangChain agent)
│   └── config.toml               # Chainlit config (title, theme, etc.)
├── backend/                      # Core logic: embeddings, indexing, QA
│   ├── ingest/
│   │   ├── parse_publications.py # PDF, HTML, or plain text loader
│   │   ├── embed_and_store.py    # Convert text → embeddings → store in ChromaDB
│   │   └── __init__.py
│   ├── qa/
│   │   ├── query_engine.py       # Embedding search + summarization pipeline
│   │   └── __init__.py
│   └── __init__.py
├── data/                         # Raw and processed publication data
│   ├── raw/                      # Raw PDFs or metadata
│   └── processed/                # Text chunks or cleaned files
├── chroma/                       # Local ChromaDB index directory (auto-created)
├── notebooks/                    # (Optional) Jupyter notebooks for exploration
│   └── analysis.ipynb
├── tests/                        # Unit and integration tests
│   ├── test_ingest.py
│   ├── test_query.py
│   └── ...
├── scripts/                      # Convenience scripts (e.g., bootstrap)
│   └── run_ingest.sh
├── .env                          # API keys, secrets (ignored by git)
├── .gitignore
├── README.md
└── requirements.txt              # Pip dependencies
```