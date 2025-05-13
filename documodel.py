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
model = os.environ.get("MODEL", "mistral:7b-instruct-q4_0")  # Set the LLM model name, defaulting to "llama3"
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", 'all-mpnet-base-v2')  # Set the embeddings model name, defaulting to "all-mpnet-base-v2"
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")  # Set the directory for persisting Chroma DB, defaulting to "db"
# all-MiniLM-L12-v2
embedding_model_directory = 'EModels/sentence-transformers/all-mpnet-base-v2'  # Path to local embedding model files
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 20))  # Number of source chunks to retrieve for an answer, defaulting to 20

# Define a function to find the directory for the embeddings model
def find_embedding_directory(base_directory, model_name):
    return os.path.join(base_directory, model_name)  # Construct and return the directory path based on the base directory and model name


# Define a class for local HuggingFace embeddings to ensure local model usage
class LocalHuggingFaceEmbeddings:
    def __init__(self, model_path, device):
        print(f"Initializing transformer model from path: {model_path}")  # Print the model path being used

        # Initialize the transformer model and pooling layer
        word_embedding_model = models.Transformer(model_name_or_path=model_path)  # Load the transformer model from the local path
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())  # Create a pooling layer
        # Combine the transformer model and pooling layer into a SentenceTransformer
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])  # Initialize the SentenceTransformer with the modules
        self.device = device  # Set the device to use (GPU or CPU)





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
    print("Initializing Chroma DB")
    db = Chroma(persist_directory=embeddings_directory, embedding_function=embeddings)
    print(f"Collection name: {db._collection.name}")
    print(f"Documents in collection: {db._collection.count()}")

    # Build a plain retriever (no score_threshold here)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Relevance cut‑off for manual filtering (0‒1, higher = stricter)
    RELEVANCE_CUTOFF = 0.05




    # Set up the retriever with the specified number of source chunks

    # Set up callbacks for streaming output
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]  # Set up streaming output callbacks if not muted

    # Initialize the LLM
    print("Initializing LLM")  # Print that the LLM is being initialized
    print(f"Ollama is using model: {model}")
    SYSTEM_PROMPT = (
        "You are an assistant whose ONLY knowledge source is the context. "
        "If the context does not contain the answer, respond exactly:\n"
        "'I don’t know based on the files I have.'"
    )

    llm = Ollama(model=model, system=SYSTEM_PROMPT, callbacks=callbacks)
    # Initialize the Ollama LLM

    # Set up the QA chain with the retriever and LLM
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)  # Set up the QA chain

    # Main loop to handle user queries
    # ---------------- MAIN LOOP ----------------
    DEBUG = False  # Set to True if you want to see raw distance scores for tuning

    while True:
        # Prompt the user for a query
        try:
            query = input("\nEnter a query, or type 'exit' when done asking questions. ")
        except (EOFError, KeyboardInterrupt):   # Graceful exit on Ctrl‑C / Ctrl‑D
            break

        if query.lower() == "exit":             # Exit the loop if the user types "exit"
            break
        if query.strip() == "":                 # Ignore blank lines
            continue

        # ---------- similarity search ----------
        # ---------- similarity search ----------
        hits = db.similarity_search_with_relevance_scores(query, k=35)

        # 🚀 DROP THE FILTER — give the LLM every hit that Chroma returned
        filtered_docs = [doc for doc, dist in hits]  # <-- keep all

        # (if you ever want a lenient cutoff, use e.g.  dist < 0.4  instead.)
        # ---------------------------------------

        print(f"Retriever pulling from: {retriever.search_kwargs}")

        if not filtered_docs:
            print("I don’t know based on the files I have.")
            continue
        # ----------------------------------------

        # ---------- generate answer ----------
        SYSTEM_PROMPT = (
            "You are an assistant whose ONLY knowledge source is the context. "
            "If the context lacks an answer, reply exactly:\n"
            "I don’t know based on the files I have."
        )
        # (prompt stays where it is)

        with yaspin(text="Generating answer...", color="green"):
            start = time.time()
            answer_dict = qa.combine_documents_chain.invoke(  # <- use invoke, not run
                {"input_documents": filtered_docs,
                 "question": query}
            )
            answer = answer_dict["output_text"]
        elapsed = time.time() - start
        # ----------------------------------------

        # Extract answer and the source documents
        #answer = res.get("result", "No answer found")
        #docs   = res.get("source_documents", [])
        docs = filtered_docs

        # ----------------- printing -----------------
        print("\n" + "-" * 50 + "\n")
        print("\n\n> Question:")
        print(query)         # Print the query
        print("\n> Answer:")
        print(answer)        # Print the answer
        print(f"\n> Time taken: {elapsed:.2f}s")
        print("\n" + "-" * 50 + "\n")


        # Print the relevant source documents if not hidden
        if not args.hide_source and docs:
            print("\n> Relevant Sources:")  # Print the relevant sources header
            for document in docs[:args.max_sources]:  # Limit the number of sources printed
                source = document.metadata.get("source", "Unknown Source")  # Get the source metadata
                page = document.metadata.get("page", "Unknown Page")  # Get the page number metadata
                print(f"\n> {source}, Page {page}:")  # Print the source and page number
                print(document.page_content)  # Optionally print the content of the source document
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
    parser.add_argument("--max-sources", "-S", type=int, default=3,
                        help='Specify the maximum number of source documents to display.')  # Add an argument to specify the maximum number of source documents

    return parser.parse_args()  # Parse and return the arguments

# If this script is executed, run the main function
if __name__ == "__main__":
    main()  # Call the main function to start the program
