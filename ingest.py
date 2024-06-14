import glob
from multiprocessing import Pool
from tqdm import tqdm
import torch
import os
import warnings
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from constants import CHROMA_SETTINGS
from sympy import sympify
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

persist_directory = os.environ.get('PERSIST_DIRECTORY', 'db')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
chunk_size = 1000
chunk_overlap = 100

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    def load(self):
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            raise type(e)(f"{self.file_path}: {e}") from e
        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def extract_text_with_ocr(page):
    text = page.get_text("text")
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = page.document.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        ocr_text = pytesseract.image_to_string(image)
        text += f"\n[Image OCR]:\n{ocr_text}\n"
    return text

def extract_equations(text):
    equations = []
    for line in text.split('\n'):
        try:
            equation = sympify(line)
            equations.append(str(equation))
        except:
            continue
    return '\n'.join(equations)

def process_document(doc):
    if isinstance(doc, list):
        processed_docs = []
        for sub_doc in doc:
            if hasattr(sub_doc, 'pages'):
                for page in sub_doc.pages:
                    page_text = extract_text_with_ocr(page)
                    equations = extract_equations(page_text)
                    sub_doc.text += page_text + "\n" + equations
            processed_docs.append(sub_doc)
        return processed_docs
    else:
        if hasattr(doc, 'pages'):
            for page in doc.pages:
                page_text = extract_text_with_ocr(page)
                equations = extract_equations(page_text)
                doc.text += page_text + "\n" + equations
        return [doc]

def load_single_document(file_path):
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        docs = loader.load()
        return process_document(docs)
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir, ignored_files=[]):
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                print(f"Processing file: {filtered_files[i]}")
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files=[]):
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory):
    index_files = glob.glob(os.path.join(persist_directory, 'index/*'))
    return os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')) and len(index_files) > 3

def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if does_vectorstore_exist(persist_directory):
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        batch_size = 5000
        text_batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        for i, text_batch in enumerate(text_batches):
            print(f"Processing batch {i + 1} of {len(text_batches)}")
            if i == 0:
                db = Chroma.from_documents(text_batch, embeddings, persist_directory=persist_directory)
            else:
                db.add_documents(text_batch)
        print("Data ingestion has finished, you can ask the chatbot questions regarding your files.")

if __name__ == "__main__":
    main()
