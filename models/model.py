import requests
from models.constants import ENDPOINTS, OPENAI_API_KEY, HF_API_KEY, GEMINI_API_KEY

PLACEHOLDER_PROPMT = "Hi! How's it going?"

class Model: 
    def __init__(self, model): 
        self.MODEL = model

    def query_gpt(self, system_prompt="", user_prompt=PLACEHOLDER_PROPMT): 
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
        }
        response = requests.post(ENDPOINTS["GPT-4"], headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

    def query_llama(self, prompt=PLACEHOLDER_PROPMT): 
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }            
        data = {
            "inputs": prompt
        }
        response = requests.post(ENDPOINTS["LLAMA"], headers=headers, json=data)
        return response.json()[0]["generated_text"]

    def query_gemini(self, prompt=PLACEHOLDER_PROPMT):
        data = {
            "contents": [{
                "parts":[{
                "text": prompt}]
            }]
        }
        response = requests.post(ENDPOINTS["GEMINI"], json=data)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    def query(self): 
        if self.MODEL == "LLAMA": 
            return self.query_llama()
        if self.MODEL == "GPT-4": 
            return self.query_gpt()
        if self.MODEL == "GEMINI": 
            return self.query_gemini()