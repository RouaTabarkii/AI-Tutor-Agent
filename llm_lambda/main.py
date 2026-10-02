import os
import json
from unittest import loader
from groq import Groq
from langchain_groq import ChatGroq
import pypdf
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
import pathlib
from pathlib import Path




load_dotenv()


QDRANT_URL = "http://localhost:6333"


if "history" not in st.session_state:
    st.session_state.history = []


client = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model = "openai/gpt-oss-120b" 
)


#l from_messages y9bel ken liste fi wostha tuple w f kol tuple ylzm l item chnwa esmouw w chnwa role mte3ou


prompt = ChatPromptTemplate.from_messages([
        ('system','''You are a tutor ai that helps student for their homework
        answer the questions of the students in the language of the user
        and explain in a simple way cause you're talking to teenagers using the file if the user didn't send 
        a file then answer without the file logic
        '''),
        MessagesPlaceholder(variable_name="history"),
        ('human',"{questions}")

])


chain = prompt | client | StrOutputParser()

user_input = st.chat_input("\nAsk Something: ")

my_file = st.file_uploader("Upload your document", type=None, accept_multiple_files=False, help=None, max_upload_size=None, label_visibility="visible")

os.makedirs("./uploaded_files", exist_ok=True) 

if my_file is not None : 
        file_path = f"./uploaded_files/{my_file.name}"
        with open(file_path, "wb") as f :
            f.write(my_file.getbuffer())
        reader = PyPDFLoader(Path(file_path))   
        #reader accept only paths 
        my_doc = reader.load()

     

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        chunks = splitter.split_documents(my_doc)
        print(f"Created {len(chunks)} chunks.")


        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


        vector_store = QdrantVectorStore.from_documents(
            
            documents=chunks,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name="ai_tutor",
        )


        doc = vector_store.similarity_search(

            user_input, k=3
        )

        context = "\n\n".join([document.page_content for document in doc])



if user_input is not None : 
    if user_input.lower() == "/bye" : 
        st.chat_message("assistant").write("AI, Good bye !")


    else : 
        st.chat_message("user").write(user_input)
        response = chain.invoke({
            'questions' : user_input,
            'history' : st.session_state.history,
            'text' : context
            })

        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response))
        st.chat_message("assistant").write(response)


#StrOutputParser() t3awedh l .content
    # print("AI : {}".format(response) )




# def resume_pdf(doc):
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": f"Summarize this document:\n\n{doc}"}],
#     )
#     return resp.choices[0].message.content


# def generate_quiz(doc, num_questions=5):
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=[{
#             "role": "user",
#             "content": f"Create {num_questions} multiple-choice quiz questions "
#                        f"(with correct answers) from this document, return as JSON:\n\n{doc}"
#         }],
#     )
#     return resp.choices[0].message.content


# def generate_exam(doc, num_questions=10, difficulty="medium"):
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=[{
#             "role": "user",
#             "content": f"Create a {difficulty} exam with {num_questions} questions "
#                        f"and an answer key from this document:\n\n{doc}"
#         }],
#     )
#     return resp.choices[0].message.content


# def search_document(doc, query):
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=[{
#             "role": "user",
#             "content": f"Find and quote the parts of this document relevant to "
#                        f"'{query}':\n\n{doc}"
#         }],
#     )
#     return resp.choices[0].message.content


# AVAILABLE_FUNCTIONS = {
#     "resume_pdf": resume_pdf,
#     "generate_quiz": generate_quiz,
#     "generate_exam": generate_exam,
#     "search_document": search_document,
# }


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "resume_pdf",
#             "description": "Summarize the uploaded document into a short overview.",
#             "parameters": {"type": "object", "properties": {}, "required": []},
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "generate_quiz",
#             "description": "Create quiz questions (multiple choice) from the uploaded document.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "num_questions": {"type": "integer", "description": "How many questions to generate"}
#                 },
#                 "required": ["num_questions"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "generate_exam",
#             "description": "Create a formal exam with an answer key from the uploaded document.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "num_questions": {"type": "integer"},
#                     "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
#                 },
#                 "required": ["num_questions"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "search_document",
#             "description": "Search the document for a specific topic or keyword and return relevant excerpts.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string", "description": "What to search for"}
#                 },
#                 "required": ["query"],
#             },
#         },
#     },
# ]


# def run_agent(user_prompt, doc):
#     messages = [
#         {
#             "role": "user",
#             "content": f"The user uploaded a document and said: '{user_prompt}'. "
#                        f"Decide which tool fits best and call it.",
#         }
#     ]

#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         tools=tools,
#         tool_choice="auto",
#     )

#     response_message = response.choices[0].message
#     tool_calls = response_message.tool_calls

#     if not tool_calls:
#         return response_message.content

#     messages.append(response_message)

#     for call in tool_calls:
#         function_name = call.function.name
#         function_args = json.loads(call.function.arguments)
#         function_to_call = AVAILABLE_FUNCTIONS[function_name]

#         result = function_to_call(doc=doc, **function_args)

#         messages.append({
#             "role": "tool",
#             "tool_call_id": call.id,
#             "name": function_name,
#             "content": result,
#         })

#     final_response = client.chat.completions.create(
#         model=MODEL,
#         messages=messages,
#     )
#     return final_response.choices[0].message.content




# uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

# doc = ""
# if uploaded_file is not None:
#     reader = pypdf.PdfReader(uploaded_file)
#     for i in range(len(reader.pages)):
#         doc += reader.pages[i].extract_text()


# if user_prmpt and uploaded_file:
#     with st.spinner("Thinking..."):
#         answer = run_agent(user_prmpt, doc)
#     st.write(answer)

# elif user_prmpt and not uploaded_file:
#     st.warning("Please upload a PDF first.")