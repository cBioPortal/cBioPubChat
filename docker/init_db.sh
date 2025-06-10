#!/bin/bash

if [ ! -d "data/vectordb" ]; then
  echo "Downloading vector db..."
  wget -O chromadb.zip 'https://zenodo.org/records/15557780/files/cbioportal_chroma_zenodo.zip?download=1'
  unzip chromadb.zip
  mv cbioportal_chroma_zenodo/vectordb data/vectordb
  rm chromadb.zip
fi

exec chainlit run app.py --host 0.0.0.0