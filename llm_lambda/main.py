import os

from groq import Groq
import pypdf 

import streamlit as st
import os
print(os.getcwd())



uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

if uploaded_file is not None: 

    reader = pypdf.PdfReader(uploaded_file)

    doc = ""

    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        doc += text
        # print(text)

client = Groq( 
    api_key=os.environ.get("GROQ_API_KEY"),
)

user_prmpt = st.text_input("Enter your prompt:")

if user_prmpt:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_prmpt+ "\n\n" + doc
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    st.write(chat_completion.choices[0].message.content)