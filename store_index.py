from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# 1. Load the Data (Fixed function name)
extracted_data = load_pdf_file(data='data/')

# 2. Split the Text (Skipping the missing 'filter' step)
text_chunks = text_split(extracted_data)

# 3. Download Embeddings
embeddings = download_hugging_face_embeddings()

# 4. Initialize Pinecone Client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# 5. Create Index (if it doesn't exist)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 6. Upload Vectors to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)