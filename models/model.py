import requests
from models.constants import ENDPOINTS, OPENAI_API_KEY, HF_API_KEY, GEMINI_API_KEY, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
import google.generativeai as genai
import os
from openai import AzureOpenAI
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model: 
    def __init__(self, model): 
        self.MODEL = model
        
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
        ]

        # self.gemini = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config, safety_settings=safety_settings)
        self.gemini = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config, safety_settings=safety_settings)

        self.OPENAI_CLIENT = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def query_mixtral(self, prompt): 
        MAX_TOKENS = 512
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        data = {
            "inputs": prompt,
              "parameters": {
                "max_new_tokens": MAX_TOKENS,
            } 
        }
        response = requests.post(ENDPOINTS["MIXTRAL"], headers=headers, json=data)
        # print(response.json())
        return response.json()[0]["generated_text"]

    def query_gpt(self, prompt): 
        deployment_name="gpt-4-turbo"
            
        response = self.OPENAI_CLIENT.chat.completions.create(
            model = deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        with open(f"models/gpt_dump.txt", "a") as f: 
            f.write(json.dumps(response.json()))

        return response.choices[0].message.content

        ### call using Will's OpenAI endpoint
        # headers = {
        #     "Authorization": f"Bearer {OPENAI_API_KEY}",
        #     "Content-Type": "application/json",
        # }
        # data = {
        #     "model": "gpt-4-turbo",
        #     "messages": [

        # }
        # response = requests.post(ENDPOINTS["GPT-4"], headers=headers, json=data)
        # return response.json()['choices'][0]['message']['content']

    def query_llama(self, prompt): 
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        data = {
            "inputs": prompt
        }
        response = requests.post(ENDPOINTS["LLAMA"], headers=headers, json=data)
        return response.json()[0]["generated_text"]

    def query_gemini(self, prompt=""):
        try: 
            response = self.gemini.generate_content(prompt)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        # time.sleep(4)
        return response.text
    
    def query(self, prompt):
        response = None
        if self.MODEL == "LLAMA": 
            response = self.query_llama(prompt)
        if self.MODEL == "GPT-4": 
            response = self.query_gpt(prompt)
        if self.MODEL == "GEMINI": 
            response = self.query_gemini(prompt)
        if self.MODEL == "MIXTRAL":
            response = self.query_mixtral(prompt)
        return response
    
