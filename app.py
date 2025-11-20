from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq # NEW IMPORT
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# --- APP SETUP ---
app = Flask(__name__)

# Load keys from .env file
load_dotenv()

# --- KEY SETUP ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY') # NEW GROQ KEY
# Set environment variables for the current session
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# --- RAG CHAIN SETUP ---
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot" 

# Initialize vector store (retriever's memory)
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize Groq Model
chatModel = ChatGroq(
    model_name="llama-3.1-8b-instant", # Fast, high-quality free model
    groq_api_key=GROQ_API_KEY
)

# Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # Assuming system_prompt is defined in src.prompt
        ("human", "{input}"),
    ]
)

# Assemble the RAG Chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- FLASK ROUTES ---
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User Query: ", input)
    
    # Invoke the RAG Chain
    response = rag_chain.invoke({"input": msg})
    
    print("AI Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    # Changed port to 5000 to avoid "forbidden socket" errors
    app.run(host="0.0.0.0", port=5000, debug=True)