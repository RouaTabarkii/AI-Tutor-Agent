import os
import json

from groq import Groq
import pypdf

import streamlit as st

print(os.getcwd())

MODEL = "llama-3.3-70b-versatile"  # confirmed available on this account via the sidebar model list

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)



def resume_pdf(doc):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Summarize this document:\n\n{doc}"}],
    )
    return resp.choices[0].message.content


def generate_quiz(doc, num_questions=5):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Create {num_questions} multiple-choice quiz questions "
                       f"(with correct answers) from this document, return as JSON:\n\n{doc}"
        }],
    )
    return resp.choices[0].message.content


def generate_exam(doc, num_questions=10, difficulty="medium"):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Create a {difficulty} exam with {num_questions} questions "
                       f"and an answer key from this document:\n\n{doc}"
        }],
    )
    return resp.choices[0].message.content


def search_document(doc, query):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Find and quote the parts of this document relevant to "
                       f"'{query}':\n\n{doc}"
        }],
    )
    return resp.choices[0].message.content


AVAILABLE_FUNCTIONS = {
    "resume_pdf": resume_pdf,
    "generate_quiz": generate_quiz,
    "generate_exam": generate_exam,
    "search_document": search_document,
}


tools = [
    {
        "type": "function",
        "function": {
            "name": "resume_pdf",
            "description": "Summarize the uploaded document into a short overview.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": "Create quiz questions (multiple choice) from the uploaded document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_questions": {"type": "integer", "description": "How many questions to generate"}
                },
                "required": ["num_questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_exam",
            "description": "Create a formal exam with an answer key from the uploaded document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_questions": {"type": "integer"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["num_questions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Search the document for a specific topic or keyword and return relevant excerpts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"],
            },
        },
    },
]


def run_agent(user_prompt, doc):
    messages = [
        {
            "role": "user",
            "content": f"The user uploaded a document and said: '{user_prompt}'. "
                       f"Decide which tool fits best and call it.",
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if not tool_calls:
        return response_message.content

    messages.append(response_message)

    for call in tool_calls:
        function_name = call.function.name
        function_args = json.loads(call.function.arguments)
        function_to_call = AVAILABLE_FUNCTIONS[function_name]

        result = function_to_call(doc=doc, **function_args)

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "name": function_name,
            "content": result,
        })

    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return final_response.choices[0].message.content




uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

doc = ""
if uploaded_file is not None:
    reader = pypdf.PdfReader(uploaded_file)
    for i in range(len(reader.pages)):
        doc += reader.pages[i].extract_text()

user_prmpt = st.text_input("Enter your prompt:")

if user_prmpt and uploaded_file:
    with st.spinner("Thinking..."):
        answer = run_agent(user_prmpt, doc)
    st.write(answer)

elif user_prmpt and not uploaded_file:
    st.warning("Please upload a PDF first.")