import os
# Import the Settings class from the chromadb.config module
from chromadb.config import Settings

# Get the value of the PERSIST_DIRECTORY environment variable. If it's not set, default to 'db'.
# This is the directory where the Chroma database will be stored.
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')

# Define the settings for the Chroma database.
# persist_directory is the directory where the Chroma database will be stored.
# anonymized_telemetry is set to False, which means the database will not collect anonymized telemetry data.
# Even if the data is anonymized, we won't take any risks
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
)

# Chroma in the code is used as a vector store!

# The information we save to our chroma database is derived from the files we upload to it via our ingest.py program
# The Chroma database stores vector representations of the text data from the uploaded files
# The embeddings are generated using the HuggingFace model specified in the ingest.py program
# They capture the contextual meaning of the text and allow the system to perform efficient similarity searches.
# It converts our info into something that our LLM can understand
# When you query the database, it finds the vectors that are most similar to your query
# Then, it retrieves the most relevant pieces of info from the uploaded files
