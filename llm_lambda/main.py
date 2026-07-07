import os

from groq import Groq
import pypdf 

import os
print(os.getcwd())

reader = pypdf.PdfReader("../Lettre de Motivation.pdf")
print(len(reader.pages
          ))

client = Groq( 
    api_key=os.environ.get("GROQ_API_KEY"),
)

user_prmpt = input("Enter your prompt: ")
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": user_prmpt,
        }
    ],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

print(chat_completion.choices[0].message.content)