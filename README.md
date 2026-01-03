
# Medical Chatbot – Intelligent Healthcare Assistant 

A Retrieval-Augmented Generation (RAG) based Medical Chatbot that allows users to ask health-related questions and receive accurate, context-aware responses by combining Large Language Models (LLMs) with a vector database.
The system supports multiple LLM providers (GROQ & Gemini), scalable cloud deployment, and automated CI/CD using Docker and AWS.

## Key Features

* Retrieval-Augmented Generation (RAG) for factual accuracy.

* Dual LLM Support
  * Groq – Llama-3.1-8B (Ultra-fast responses)
  * Gemini 2.5 Flash (Deeper clinical reasoning)

* Vector Database with Pinecone.

* HuggingFace Sentence Transformers for embeddings.

* Flask-based Web Interface.

* Dockerized Deployment.

* AWS EC2 + ECR Hosting.

* Secure API key management via environment variables.


## System Architecture

**Data Ingestion Pipeline**

* Medical PDF extracted.
* Text chunked using RecursiveCharacterTextSplitter.
* Embeddings generated using Sentence Transformers.
* Vectors stored in Pinecone.

**Query Execution Flow (RAG)**

* User submits a query.
* Relevant chunks retrieved from Pinecone.
* Retrieved context passed to LLM.
* Grounded response generated and returned.

## Dual LLM Strategy

**Groq(Llama-3.1)**

* Ultra-low latency (<1s).
* Real-time triage, fast Q&A.

**Gemini 2.5 Flash**

* Deep reasoning & Structure.
* Clinical explainations.

**But both models:**

* **Retrieve the same verified context.**
* **Produce 100% grounded responses.**


## Tech Stack

**Language :** Python.

**LLMs :** Groq (Llama-3.1-8B), Google Gemini 2.5 Flash.

**RAG Framework :** LangChain.

**Vector DB :** Pinecone.

**Embeddings :** HuggingFace(Sentence-Transformers/all-MiniLM-L6-v2).

**Frontend :** HTML, CSS, JavaScript.

**Backend :** Flask.

**Devops & Cloud :**




## Setup Instructions
**STEPS:**

Clone the repository
```
git clonehttps://github.com/ritheshvreddy/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-
```
### Local setup

**Step 1 :**
Create Virtual Environment

```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```
**Step 2 :**
Install the requirements

```
pip install -r requirements.txt
```
**Step 3 :**
Create a .env file in the root directory and add your Pinecone, Groq, Gemini credentials as follows:

```
PINECONE_API_KEY="***************"
PINECONE_ENVIRONMENT="***************"
GROQ_API_KEY="***************"
GEMINI_API_KEY="***************" 
```
**Step 4 :**
Run the following command to store embeddings to pinecon

```
python store_index.py
```
**Step 5 :** 
Run Locally 

**Groq – Llama 3.1**
```
python app.py
```
Access : http://localhost:5000

**Gemini 2.5 Flash :**

Change port to 5001 
```env
PORT=5001
```
Then run:
```
python app.py
```
Access : http://localhost:5001

### AWS Cloud Deployment (EC2 + ECR + IAM)

**Step 1 :** Login to AWS console.

**Step 2 :** Create IAM user for Deployment
* Go to AWS IAM.
* Create a new user.
* Attach policies:
   * AmazonEC2FullAccess
   * AmazonEC2ContainerRegistryFullAccess
* Generate Access Key & Secret Key

**Step 2 :** Create Amazon ECR Repository
* Go to AWS ECR (e.g.,medicalchatbot)
* Create repository
* region : us-east-1
* save the URI : <account-id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot

**Step 3 :** Launch EC2 Instance
* Ubuntu 24.04
* Instance type : t3.small recommended
* Open ports:
  * 5000 (GROQ)
  * 5001 (Gemini)
  * 22 (SSH)

**Step 4 :** Install Docker on EC2
```
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
newgrp docker
```
**Step 5 :** Configure EC2 as self-hosted runner
```
settings>actions>runner>new self hosted runner> choose os (Linux)> then run command one by one|
```
**Step 6 :** Setup github secrets
* AWS_ACCESS_KEY_ID
* AWS_DEFUALT_REGION
* AWS_SECRET_ACCESS_KEY
* ECR_REPO
* PINECONE_API_KEY
* GROQ_API_KEY
* GEMINI_API_KEY

**Step 7 :** Login to ECR(EC2)
```
aws ecr get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```
**Step 5 :** Build & Push Docker Images to AWS ECR
* GROQ Image
```
docker build -t medical-chatbot-groq .
docker tag medicalchatbot-groq \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-groq:latest
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-groq:latest
```
* Gemini Image
```
docker build -t medical-chatbot-gemini .
docker tag medicalchatbot-gemini \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-gemini:latest
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-gemini:latest
```
**Step 8 :** Run Containers on EC2
* GROQ (LLaMA-3.1) Container
```
docker run -d \
  --name medical-chatbot-groq \
  -p 5000:5000 \
  -e PINECONE_API_KEY=xxx \
  -e GROQ_API_KEY=xxx \
  -e GEMINI_API_KEY=xxx \
  <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-groq:latest
```
* GEMINI 2.5 FLASH Container
```
docker run -d \
  --name medical-chatbot-gemini \
  -p 5001:5001 \
  -e PORT=5001 \
  -e PINECONE_API_KEY=xxx \
  -e GROQ_API_KEY=xxx \
  -e GEMINI_API_KEY=xxx \
  <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/medicalchatbot-gemini:latest
```
**Public Endpoints**

GROQ (LLaMA-3.1) : http://<EC2_PUBLIC_IP>:5000

Gemini 2.5 Flash : http://<EC2_PUBLIC_IP>:5001  

Notes :
* Each LLM runs in a separate Docker container
* Containers remain active as long as the EC2 instance is running
* Stopping or terminating the EC2 instance will stop the application

## Future Enhancements
1.	Multi-Modal Support: Allow users to upload images of skin conditions or lab reports for analysis.
2.	Voice Interface: Integrate Speech-to-Text to allow elderly patients to interact via voice.
3.	Chat History: Implement session memory so the bot remembers previous context in the conversation.
4.  Replace Flask dev server with Gunicorn
5.  Authentication & user sessions
