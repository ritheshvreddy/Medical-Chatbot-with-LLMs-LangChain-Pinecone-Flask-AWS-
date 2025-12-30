from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # New dedicated text splitter package
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. Load PDF Function
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# 2. Text Split Function (Uses new langchain_text_splitters package)
def text_split(extracted_data):
    # These chunk sizes must be integers
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# 3. Download Embeddings Function
def download_hugging_face_embeddings():
    # HuggingFaceEmbeddings is now found in the community package
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings