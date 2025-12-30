from flask import Flask, render_template, request, jsonify
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

# --- KEYS ---
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# --- SHARED MEMORY (Pinecone) ---
pc = Pinecone(api_key=PINECONE_API_KEY) 
index_name = "medical-chatbot"
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# --------------------------------------------------
# DECIDE MODEL BASED ON PORT
# --------------------------------------------------

PORT = int(os.environ.get("PORT", 5000))
print(f"\n🔥 RUNNING APP ON PORT: {PORT}\n")

use_groq = (PORT == 5000)
use_gemini = (PORT == 5001)

# --------------------------------------------------
# BRAIN 1 → GROQ (Port 5000)
# --------------------------------------------------
if use_groq:
    print("🧠 Using GROQ Llama-3.1-8B-Instant")
    groq_llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=GROQ_API_KEY
    )
    qa_chain = create_stuff_documents_chain(groq_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

# --------------------------------------------------
# BRAIN 2 → GEMINI (Port 5001)
# --------------------------------------------------
elif use_gemini:
    print("🧠 Using GEMINI 2.5 Flash")
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=GEMINI_API_KEY
    )
    qa_chain = create_stuff_documents_chain(gemini_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    name = "Groq (Llama-3)" if use_groq else "Gemini 2.5 Flash"
    return render_template("chat.html", model_name=name, endpoint="/ask")

@app.route("/ask", methods=["POST"])
def ask():
    msg = request.form["msg"]
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])


if __name__ == '__main__':
    print(f"🚀 Starting Flask on 0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
