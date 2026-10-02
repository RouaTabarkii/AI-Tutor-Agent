# AI Tutor Agent

## 1. Overview

An AI-powered tutor that assists students by answering questions, generating assessments, and providing explanations based on course materials.

## 2. Features

- Upload PDF documents.

- Extract and split document text into chunks.

- Generate embeddings and store them in a vector database.

- Retrieve relevant information using semantic search.

- Answer questions using an LLM.

- Interact through a Streamlit chat interface.

## 3. Technologies :
- Pyhton
- LangChain
- Qdrant
- Docker
- RAG
- Groq
- LLM
- Streamlit

## 4. Installation :
- git clone http://RouaTabarkii/AI-Tutor-Agent
- cd AI-Tutor-Agent
- pip install -r requirements.txt

## 5. Configuration :
- Create a virtual environment (.env)
- Create an API key from Groq api key
- Add it here : GROQ_API_KEY="your api key"

## 6. Run the application :
- streamlit run main.py

## 7. Project sturcture :
llm_lambda/
├── qdrant_data

├── uploaded_files

├── main.py

├── .gitignore

├── README.md

└── requirements.txt
