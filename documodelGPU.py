# Import necessary modules and libraries
import os  # For interacting with the operating system
import warnings  # For managing warning messages
import torch  # Pytorch library for GPU support
import time  # For measuring time intervals
import argparse  # For parsing command-line arguments
import requests  # For making HTTP requests
from yaspin import yaspin  # For creating loading spinners in the console
from langchain.chains import RetrievalQA  # For creating a retrieval-based QA chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # For streaming output callbacks
from langchain_community.vectorstores import Chroma  # For managing a Chroma vector store
from langchain_community.llms import Ollama  # For using the Ollama language model
from constants import CHROMA_SETTINGS  # Custom settings for Chroma
from sentence_transformers import SentenceTransformer, models  # For sentence embedding models
import socket  # For network connections

# Suppress specific warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore user warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore future warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow

# Set CUDA environment variables to specify which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Order CUDA devices by PCI bus ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify to use the first GPU (index 0)

# Environment variables to configure the model and embeddings
model = os.environ.get("MODEL", "llama3")  # Set the LLM model name, defaulting to "llama3"
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", 'all-mpnet-base-v2')  # Set the embeddings model name, defaulting to "all-mpnet-base-v2"
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")  # Set the directory for persisting Chroma DB, defaulting to "db"
embedding_model_directory = 'EModels/sentence-transformers/all-mpnet-base-v2'  # Path to local embedding model files
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 25))  # Number of source chunks to retrieve for an answer, defaulting to 20

# Define a function to find the directory for the embeddings model
def find_embedding_directory(base_directory, model_name):
    return os.path.join(base_directory, model_name)  # Construct and return the directory path based on the base directory and model name

# Define a class for local HuggingFace embeddings to ensure local model usage
class LocalHuggingFaceEmbeddings:
    def __init__(self, model_path, device):
        print(f"Initializing transformer model from path: {model_path}")  # Print the model path being used
        self.block_network_access()  # Block network access to ensure no internet usage
        # Initialize the transformer model and pooling layer
        word_embedding_model = models.Transformer(model_name_or_path=model_path)  # Load the transformer model from the local path
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())  # Create a pooling layer
        # Combine the transformer model and pooling layer into a SentenceTransformer
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])  # Initialize the SentenceTransformer with the modules
        self.device = device  # Set the device to use (GPU or CPU)
        self.restore_network_access()  # Restore network access after initialization

    # Block network access to prevent the model from using the internet
    def block_network_access(self):
        def guard(*args, **kwargs):
            raise RuntimeError("Network access is disabled")  # Raise an error if network access is attempted

        socket.socket = guard  # Block socket connections
        requests.get = guard  # Block HTTP GET requests
        requests.post = guard  # Block HTTP POST requests
        requests.put = guard  # Block HTTP PUT requests
        requests.patch = guard  # Block HTTP PATCH requests
        requests.delete = guard  # Block HTTP DELETE requests

    # Restore network access after model initialization
    def restore_network_access(self):
        import importlib  # Import the module for reloading
        importlib.reload(socket)  # Reload the socket module
        importlib.reload(requests)  # Reload the requests module

    # Embed documents using the local model
    def embed_documents(self, documents):
        print("Embedding documents locally")  # Print that documents are being embedded locally
        embeddings = self.model.encode(documents, show_progress_bar=False, device=self.device)  # Generate embeddings for the documents
        return [embedding.tolist() for embedding in embeddings]  # Convert embeddings to list format and return

    # Embed a query using the local model
    def embed_query(self, query):
        # print("Embedding query locally")  # Print that the query is being embedded locally
        embedding = self.model.encode([query], show_progress_bar=False, device=self.device)[0]  # Generate an embedding for the query
        return embedding.tolist()  # Convert embedding to list format and return

# Define a function to block network access for the entire script
def block_network_access():
    def guard(*args, **kwargs):
        raise RuntimeError("Network access is disabled")  # Raise an error if network access is attempted

    socket.socket = guard  # Block socket connections
    requests.get = guard  # Block HTTP GET requests
    requests.post = guard  # Block HTTP POST requests
    requests.put = guard  # Block HTTP PUT requests
    requests.patch = guard  # Block HTTP PATCH requests
    requests.delete = guard  # Block HTTP DELETE requests

# Define the main function
def main():
    #block_network_access()  # Block network access for the entire script

    args = parse_arguments()  # Parse command-line arguments

    # Construct the embeddings directory path
    embeddings_directory = find_embedding_directory(persist_directory, embeddings_model_name)

    # Check if the directory exists
    if not os.path.isdir(embeddings_directory):
        raise ValueError(f"No directory found for embeddings model {embeddings_model_name}")  # Raise an error if the directory does not exist

    print(f"Using embeddings directory: {embeddings_directory}")  # Print the directory being used for embeddings

    # Initialize the local embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set the device to GPU if available, otherwise CPU
    print(f"Using device: {device}")
    embeddings = LocalHuggingFaceEmbeddings(model_path=embedding_model_directory, device=device)  # Initialize local HuggingFace embeddings

    # Initialize Chroma DB with the embeddings
    print("Initializing Chroma DB")  # Print that Chroma DB is being initialized
    db = Chroma(persist_directory=embeddings_directory, embedding_function=embeddings)  # Initialize Chroma DB
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})  # Set up the retriever with the specified number of source chunks

    # Set up callbacks for streaming output
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]  # Set up streaming output callbacks if not muted

    # Initialize the LLM
    print("Initializing LLM")  # Print that the LLM is being initialized
    llm = Ollama(model=model, callbacks=callbacks)  # Initialize the Ollama LLM

    # Set up the QA chain with the retriever and LLM
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)  # Set up the QA chain

    # Main loop to handle user queries
    while True:
        query = input("\nEnter a query, or type 'exit' when done asking questions. ")  # Prompt the user for a query
        if query.lower() in ["exit"]:
            break  # Exit the loop if the user types "exit"
        if query.strip() == "":
            continue  # Continue if the input is empty

        with yaspin(text="Generating answer...", color="green") as spinner:  # Display a spinner while generating the answer
            start = time.time()  # Record the start time
            try:
                # Generate an answer for the query
                print("Generating answer")  # Print that the answer is being generated
                res = qa.invoke(query)  # Invoke the QA chain with the query
            except Exception as e:
                print(f"Error generating answer: {e}")  # Print an error message if an exception occurs
                continue  # Continue to the next iteration
            end = time.time()  # Record the end time
            spinner.ok("✅ ")  # Mark the spinner as complete

            duration = end - start  # Calculate the duration
            print(f"\n> Time taken to generate the answer: {duration} seconds")  # Print the time taken to generate the answer

        answer = res.get('result', 'No answer found')  # Get the answer from the response
        docs = res.get('source_documents', [])  # Get the source documents from the response

        # Print the question and the generated answer
        print("\n" + "-" * 50 + "\n")
        print("\n\n> Question:")
        print(query)  # Print the query
        print("\n> Answer:")
        print(answer)  # Print the answer
        print("\n" + "-" * 50 + "\n")

        # Print the relevant source documents if not hidden
        if not args.hide_source and docs:
            print("\n> Relevant Sources:")  # Print the relevant sources header
            for document in docs[:args.max_sources]:  # Limit the number of sources printed
                source = document.metadata.get("source", "Unknown Source")  # Get the source metadata
                print(f"\n> {source}:")
                print(document.page_content)  # Print the content of the source document
            print("\n" + "-" * 50 + "\n")

# Define a function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='DocuModel: Ask questions to your documents without an internet connection.')  # Create a parser with a description
    parser.add_argument("--hide-source", "-H", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')  # Add a flag to hide source documents
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')  # Add a flag to mute streaming output
    parser.add_argument("--max-sources", "-S", type=int, default=1,
                        help='Specify the maximum number of source documents to display.')  # Add an argument to specify the maximum number of source documents

    return parser.parse_args()  # Parse and return the arguments

# If this script is executed, run the main function
if __name__ == "__main__":
    main()  # Call the main function to start the program
