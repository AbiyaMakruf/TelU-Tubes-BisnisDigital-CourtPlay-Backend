import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID") 

client = genai.Client()

def text_generation(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text

def image_understanding(image_path, prompt):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    client = genai.Client()
    response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
        ),
        prompt
    ]
    )

    return response.text