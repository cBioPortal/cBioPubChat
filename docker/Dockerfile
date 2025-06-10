# Use the official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Download embeddings
#RUN wget -O chromadb.zip 'https://zenodo.org/records/15557780/files/cbioportal_chroma_zenodo.zip?download=1'
#RUN unzip chromadb.zip && \
#    cp -r cbioportal_chroma_zenodo/vectordb data/vectordb

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the default Chainlit port
EXPOSE 8000

# Command to run the Chainlit app
COPY docker/init_db.sh /init_db.sh
RUN chmod +x /init_db.sh
ENTRYPOINT ["/init_db.sh"]