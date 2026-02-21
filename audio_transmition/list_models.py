import os
import google.generativeai as genai

API_KEY = "AIzaSyCkwySDnCXLRMdjr9-2AI1YZOsbthja9nc"
genai.configure(api_key=API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
        print(m.name)
