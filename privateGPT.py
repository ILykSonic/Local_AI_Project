from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
import os
import argparse
import time
from yaspin import yaspin
import torch
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_101"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
print("GPU is being used:", torch.cuda.is_available())

model = os.environ.get("MODEL", "llama3")  # Change each time you switch models
# https://www.sbert.net/docs/pretrained_models.html

#os.environ["EMBEDDINGS_MODEL_NAME"] = "multi-qa-mpnet-base-cos-v1"

# These three get information from chroma. If it cannot be obtained it will default to the information after the comma
# You can set an Embedding model yourself, or use the MiniLM-L6. I suggest the LM-L6, its MTEB benchmark is great
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
# If you set PD, you can choose where you store your chroma database. Otherwise, it will save to a folder titled "db"
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
# This determines the number of chunks the chatbot tries to receive from the database.
# Higher = Slower & More Accurate/More Detailed  Lower = Faster & Less Accurate/Less Detailed
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 10000))

from constants import CHROMA_SETTINGS


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query, or type 'finished' when done asking questions. ")
        if query == "exit":
            break
        if query.lower() == "finished":
            break
        if query.strip() == "":
            continue

        with yaspin(text="Generating answer...", color="green") as spinner:
            start = time.time()
            res = qa.invoke(query)
            end = time.time()
            spinner.ok("✅ ")

            # Calculate the time it took to generate the answer
            duration = end - start
            # Print the time it took to generate the answer
            print(f"\n> Time taken to generate the answer: {duration} seconds")

        # Get the answer from the chain
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        print("\n" + "-" * 50 + "\n")
        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        print("\n" + "-" * 50 + "\n")

        # Print the relevant sources used for the answer
        if not args.hide_source:
            print("\n> Relevant Sources:")
            for document in docs[:2]:  # Change this line depending on how many sources you want.
                print("\n> " + document.metadata["source"] + ":")
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
