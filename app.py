from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os
from pinecone import Pinecone

app = Flask(__name__)
load_dotenv()

# =========================
# ENVIRONMENT VARIABLES
# =========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

missing_keys = []
if not PINECONE_API_KEY:
    missing_keys.append("PINECONE_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")
if not GEMINI_API_KEY:
    missing_keys.append("GEMINI_API_KEY")

if missing_keys:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing_keys)}")

# Set env vars explicitly for downstream SDKs
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# =========================
# PINECONE SETUP
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# =========================
# PROMPT
# =========================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# =========================
# MODEL SELECTION
# =========================
PORT = int(os.environ.get("PORT", 5000))
print(f"🔥 Running on port {PORT}")

use_groq = PORT == 5000
use_gemini = PORT == 5001

if use_groq:
    print("🧠 Using GROQ Llama-3.1")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )

elif use_gemini:
    print("🧠 Using Gemini 2.5 Flash")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )

else:
    raise RuntimeError("Invalid PORT configuration. Use 5000 (Groq) or 5001 (Gemini).")

# =========================
# RAG CHAIN
# =========================
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    name = "Groq (Llama-3)" if use_groq else "Gemini 2.5 Flash"
    return render_template("chat.html", model_name=name, endpoint="/ask")

@app.route("/ask", methods=["POST"])
def ask():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return response["answer"]

# =========================
# APP ENTRYPOINT
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
