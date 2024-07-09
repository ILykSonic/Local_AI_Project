import os
import glob
import torch
import time
import argparse
import warnings
from yaspin import yaspin
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from constants import CHROMA_SETTINGS

# Suppress specific warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set CUDA environment variables to specify which GPU to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
print("GPU is being used:", torch.cuda.is_available())

# Environment variables to configure the model and embeddings
model = os.environ.get("MODEL", "llama3")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", 'all-mpnet-base-v2')
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 5))

def find_embedding_directory(base_directory, model_name):
    # List all directories in the base directory
    all_dirs = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    # Find the directory that contains the embedding model name
    for directory in all_dirs:
        if model_name in directory:
            return os.path.join(base_directory, directory)
    return None

def main():
    args = parse_arguments()

    embeddings_directory = find_embedding_directory(persist_directory, embeddings_model_name)
    if embeddings_directory is None:
        raise ValueError(f"No directory found for embeddings model {embeddings_model_name}")

    print(f"Using embeddings directory: {embeddings_directory}")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=embeddings_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    while True:
        query = input("\nEnter a query, or type 'finished' when done asking questions. ")
        if query.lower() == "exit" or query.lower() == "finished":
            break
        if query.strip() == "":
            continue

        with yaspin(text="Generating answer...", color="green") as spinner:
            start = time.time()
            try:
                res = qa.invoke(query)
            except Exception as e:
                print(f"Error generating answer: {e}")
                continue
            end = time.time()
            spinner.ok("✅ ")

            duration = end - start
            print(f"\n> Time taken to generate the answer: {duration} seconds")

        answer = res.get('result', 'No answer found')
        docs = res.get('source_documents', [])

        print("\n" + "-" * 50 + "\n")
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        print("\n" + "-" * 50 + "\n")

        if not args.hide_source and docs:
            print("\n> Relevant Sources:")
            for document in docs:
                source = document.metadata.get("source", "Unknown Source")
                print(f"\n> {source}:")
                print(document.page_content)
            print("\n" + "-" * 50 + "\n")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    main()
