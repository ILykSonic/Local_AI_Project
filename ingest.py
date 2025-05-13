
import glob  # Used to find all the file paths that match a specified pattern, useful for loading documents.
from multiprocessing import Pool  # Allows for parallel processing to speed up document loading and processing.
from tqdm import tqdm  # A library for displaying progress bars in loops, making it easier to monitor the progress of tasks.
import torch  # PyTorch library, used for working with our GPU
import time  # Provides various time-related functions, used for measuring execution time.
import os  # Provides a way to interact with the operating system, used for environment variables and path manipulations.
import warnings  # Used to control the display of warnings.
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits the document text into smaller chunks before turning it into vectors.
# This ensures that each chunk is small enough to be processed by the embedding model without exceeding token limits.
# As a result, it creates multiple smaller vectors instead of one large vector for the entire document.
from constants import CHROMA_SETTINGS
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings  # Used to configure settings for the Chroma database.
from chromadb import Client  # Chroma database client for interacting with the vector store.
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings  # Base class for embeddings, used to define custom embedding models.
from PIL import Image  # Python Imaging Library, used for opening, manipulating, and saving images.
import pytesseract  # OCR tool for extracting text from images.
import fitz  # PyMuPDF library, used for working with PDF documents.
import io  # Core tools for working with streams (like in-memory files).
import shutil  # Deletes old database if a new one is created with the same embedding model
from sentence_transformers import SentenceTransformer  # Library for generating sentence embeddings.
import re
import uuid  # Generates unique identifier for each chunk before writing it to our vector store. Ensures that each chunk has its own identifier (preventing duplicate id issues)

# This suppresses unnecessary warnings when running the code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 suppresses all logs (set to '2' to see only errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Path for Tesseract OCR (This was the only way to get it working)
os.environ['PATH'] += os.pathsep + r'C:\Users\njmadmin\AppData\Local\Programs\Tesseract-OCR'
# CUDA settings (Both these settings are determined by what is displayed in the task manager on your pc)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
# (check to make sure the number is the same when switching hardware)

# Prints all cuda info (mainly for testing purposes)
def print_cuda_info():
    print("CUDA Available:", torch.cuda.is_available())
    print("Current CUDA Device Index:", torch.cuda.current_device())
    print("Current CUDA Device Name:", torch.cuda.get_device_name(0))
    print("PyTorch Version:", torch.__version__)

# Document Loading Config
# Import various document loaders from the langchain_community package
# These loaders are used to read and process different types of documents
# I have specifically tested pdf and txt document loaders
from langchain_community.document_loaders import (
    CSVLoader,  # Loader for CSV files
    EverNoteLoader,  # Loader for EverNote files
    PyMuPDFLoader,  # Loader for PDF files using PyMuPDF
    TextLoader,  # Loader for plain text files
    UnstructuredEmailLoader,  # Loader for unstructured email files
    UnstructuredEPubLoader,  # Loader for unstructured EPUB files
    UnstructuredHTMLLoader,  # Loader for unstructured HTML files
    UnstructuredMarkdownLoader,  # Loader for unstructured Markdown files
    UnstructuredODTLoader,  # Loader for unstructured ODT (OpenDocument Text) files
    UnstructuredPowerPointLoader,  # Loader for unstructured PowerPoint files
    UnstructuredWordDocumentLoader,  # Loader for unstructured Word document files
)

# Directory paths and model settings
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db')  # Directory to persist the Chroma database
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')  # Directory containing source documents
EMBEDDINGS_MODEL_NAME = "all-mpnet-base-v2"  # Name of the embeddings model to use

# Maximum number of tokens in each text chunk.
chunk_size = 400
# Ensure the token limit of the embedding model isn't reached, as this can cause errors
# Too low of a chunk size, we can lose valuable information
# Too High we can encounter errors.

# Number of overlapping tokens between consecutive chunks, the overlap helps maintain context between chunks.
chunk_overlap = 100
# Too high of an overlap, and we slow down processing time and introduce redundancy
# Too low, and we can lose important context and information between our chunks

# Chroma database settings, same settings as before
CHROMA_SETTINGS = Settings(
    persist_directory=PERSIST_DIRECTORY,  # Directory to store the Chroma database
    anonymized_telemetry=False  # Disable anonymized telemetry data collection
)

# Mapping of file extensions to their respective document loaders and any required arguments
LOADER_MAPPING = {
    ".pdf": (PyMuPDFLoader, {}),  # Use PyMuPDFLoader for PDF files
    ".txt": (TextLoader, {"encoding": "utf8"}),  # Use TextLoader for plain text files with UTF-8 encoding
    ".docx": (UnstructuredWordDocumentLoader, {}),  # Use UnstructuredWordDocumentLoader for Word document files
}

# Document class to encapsulate the content and metadata of a document
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content  # Contains the text of each page in our documents
        self.metadata = metadata  # Metadata is details about the document that aren't the contents of the document (e.g., file name source file path)
        # This helps in organization as well as finding the source file for the information
# End of Document Loading Config

# OCR Document and processing
def preprocess_image_for_ocr(image):
    # Convert the image to RGB mode if it is in CMYK mode
    if image.mode == 'CMYK':
        image = image.convert('RGB')
    # Resize the image to double its original dimensions using the LANCZOS filter for better OCR accuracy
    # common practice as larger text is easier for the OCR to read
    return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)


def extract_text_with_ocr(file_path: str) -> list:
    """
    Extract text (native + OCR) from *each* page of a PDF and return a
    list[Document] where every Document represents one page and carries
    {"source": <pdf-path>, "page": <1-based-page-number>} in its metadata.
    """
    doc = fitz.open(file_path)
    documents = []  # ← this replaces `full_text`

    for page_num, page in enumerate(doc, start=1):

        # ---- 1) native text -------------------------------------------------
        page_text = page.get_text()

        # ---- 2) (optional) OCR if the page is mostly images -----------------
        blocks = page.get_text("blocks")
        if blocks and len(page_text) / len(blocks) < 50:
            for xref, *_ in page.get_images(full=True):
                img_bytes = doc.extract_image(xref)["image"]
                image = Image.open(io.BytesIO(img_bytes))
                processed = preprocess_image_for_ocr(image)
                page_text += pytesseract.image_to_string(processed, lang="eng")

        # ---- 3) wrap this single page into a Document -----------------------
        documents.append(
            Document(
                page_content=page_text,
                metadata={"source": file_path, "page": page_num}
            )
        )

    doc.close()
    return documents  # list[Document]  (not one big blob)


def detect_headers(text):
    # Split the text into lines
    lines = text.split('\n')
    headers = {}
    current_header = None
    # Simplified pattern for headers
    header_pattern = re.compile(r'^\s*(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$')

    for line in lines:
        # Check if the line matches the header pattern
        if header_pattern.match(line):
            current_header = line.strip()
            headers[current_header] = []
        elif current_header:
            headers[current_header].append(line)

    return headers

def load_single_document(file_path):
    try:
        ext = "." + file_path.rsplit(".", 1)[-1]

        # ── PDF ───────────────────────────────────────────────
        if ext == ".pdf":
            # ▸ returns a list[Document] where each .metadata already
            #   contains {"source": file_path, "page": page_num}
            return extract_text_with_ocr(file_path)

        # ── all other supported types ─────────────────────────
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            docs = loader.load()

            wrapped = []
            for doc in docs:
                wrapped.append(
                    Document(
                        page_content=doc.page_content,
                        # keep whatever metadata the loader produced
                        metadata={**doc.metadata, "source": file_path}
                    )
                )
            return wrapped

        # anything else → error
        raise ValueError(f"Unsupported file extension: {ext}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# End of OCR Document Processing

# Loading and processing documents

# Function to load documents from a source directory, excluding any ignored files
def load_documents(source_dir, ignored_files=[]):
    all_files = []
    # Iterate through each file extension supported by LOADER_MAPPING
    for ext in LOADER_MAPPING:
        # Use glob to find all files with the current extension in the source directory and its subdirectories
        # os.path.join combines the source directory with the pattern to search for files with the given extension
        # '**/*{ext}' searches for all files with the given extension in the directory and its subdirectories
        # recursive=True ensures the search includes all subdirectories
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    # Filter out any files that are in the ignored_files list
    # List comprehension iterates through all_files and includes only those not in ignored_files
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    documents = []
    # Use multiprocessing to load documents in parallel, utilizing all available CPU cores
    with Pool(processes=os.cpu_count()) as pool:
        # Use tqdm to display a progress bar for loading the documents
        # 'total=len(filtered_files)' sets the total length of the progress bar
        # 'desc='Loading new documents'' sets the description shown next to the progress bar
        # 'ncols=80' sets the width of the progress bar
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            # Load documents in parallel using imap_unordered
            # imap_unordered applies the load_single_document function to each item in filtered_files in parallel
            for result in pool.imap_unordered(load_single_document, filtered_files):
                # if load_single_document returns a result (i.e., it successfully processes a document)
                if result:
                    # extend the documents list with the results
                    # We add the processed document(s) to the documents list
                    documents.extend(result)
                # this then updates the progress bar
                # pbar.update() increments the progress bar by one step
                pbar.update()
    # Return the list of loaded documents
    return documents

# Function to process documents by loading, splitting them into chunks, and returning the chunks
def process_documents(ignored_files=[]):
    # Print a message to indicate the start of the loading process
    print(f"Loading documents from {source_directory}")
    # Record the start time to measure how long the loading process takes
    start_time = time.time()
    # Call the function to load documents from the specified source directory, excluding any ignored files
    documents = load_documents(source_directory, ignored_files)
    # Check if no documents were loaded
    if not documents:
        # Print a message indicating no documents were loaded and exit the function
        print("No new documents to load")
        exit(0)
    # Record the end time to measure the total loading duration
    end_time = time.time()
    # Calculate the duration of the loading process
    load_duration = end_time - start_time
    # Print a message showing how many documents were loaded and how long it took
    print(f"Loaded {len(documents)} new documents from {source_directory} in {load_duration / 60:.2f} minutes")

    # Chunk Stuff

    # Initialize a RecursiveCharacterTextSplitter to split the documents into chunks
    # chunk_size and chunk overlap were defined earlier
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # line sets up a tool we use
    # Record the start time to measure how long the splitting process takes
    start_time = time.time()
    # Split the loaded documents into smaller chunks of text,
    # This makes it easier to process the text later, especially for tasks like machine learning or natural language processing
    texts = text_splitter.split_documents(documents) # This line actually divides them
    # Record the end time to measure the total splitting duration
    end_time = time.time()
    # Calculate the duration of the splitting process
    split_duration = end_time - start_time
    # Print a message showing how many text chunks were created, and how long it took
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each) in {split_duration:.2f} seconds")
    # Return the list of text chunks created from the documents
    return texts

# End of loading and processing documents

# Function to check if a vector store exists
def does_vectorstore_exist(persist_directory, embedding_model):
    # Construct the path to the vector store directory
    vectorstore_dir = os.path.join(persist_directory, embedding_model)

    # Check if the 'index' directory exists within the vector store directory
    if os.path.exists(os.path.join(vectorstore_dir, 'index')):
        # Check if the essential .parquet files exist within the vector store directory
        # .parquet files are used for efficient data storage and retrieval
        # - chroma-collections.parquet stores metadata about the collections within the vector store
        # - chroma-embeddings.parquet stores the actual embeddings, which are the vector representations of the text data
        if os.path.exists(os.path.join(vectorstore_dir, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(vectorstore_dir, 'chroma-embeddings.parquet')):
            # Use glob to find all .bin files in the 'index' directory
            # .bin files are binary files that store serialized data, often used for index structures
            list_index_files = glob.glob(os.path.join(vectorstore_dir, 'index/*.bin'))
            # Add any .pkl files in the 'index' directory to the list
            # .pkl files are pickle files used to serialize and deserialize Python objects, storing index data or other metadata
            list_index_files += glob.glob(os.path.join(vectorstore_dir, 'index/*.pkl'))
            # Check if there are more than 3 index files (.bin and .pkl combined)
            if len(list_index_files) > 3:
                # Return True if all checks pass
                return True
    # Return False if any of the checks fail
    return False

# End of vector store check

# Main Function
def main():
    print_cuda_info()

    # Use CUDA for embedding generation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME, model_kwargs={"device": device})

    vectorstore_dir = os.path.join(PERSIST_DIRECTORY, EMBEDDINGS_MODEL_NAME)
    print(f"Persisting data to directory: {vectorstore_dir}")

    start_time = time.time()
    if does_vectorstore_exist(PERSIST_DIRECTORY, EMBEDDINGS_MODEL_NAME):
        print(f"Appending to existing vectorstore at {vectorstore_dir}")
        db = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")

        embedding_start_time = time.time()  # Start time for embedding creation
        db.add_documents(texts)
        embedding_end_time = time.time()  # End time for embedding creation
    else:
        if os.path.exists(vectorstore_dir):
            shutil.rmtree(vectorstore_dir)
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")

        batch_size = 2000
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        embedding_start_time = time.time()  # Start time for embedding creation
        for i, text_batch in enumerate(text_batches):
            print(f"Processing batch {i + 1} of {len(text_batches)}")
            if i == 0:
                db = Chroma.from_documents(text_batch, embeddings, persist_directory=vectorstore_dir)
            else:
                db.add_documents(text_batch)
        embedding_end_time = time.time()  # End time for embedding creation

    end_time = time.time()
    ingestion_duration = end_time - start_time
    embedding_duration = embedding_end_time - embedding_start_time  # Duration for embedding creation

    print(f"Embedding creation process took {embedding_duration:.2f} seconds.")  # Print embedding creation time
    print(f"Ingestion process took {ingestion_duration / 60:.2f} minutes.")
    print(f"Data ingestion has finished, you can ask the chatbot questions regarding your files.")

# Run Main Function
if __name__ == "__main__":
    main()

