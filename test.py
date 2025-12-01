import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyDp_8vaTegDyR7UOMq2CRNp_K98-sanOTc")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
