import requests
from models.constants import ENDPOINTS, OPENAI_API_KEY, HF_API_KEY, GEMINI_API_KEY
import google.generativeai as genai

class Model: 
    def __init__(self, model): 
        self.MODEL = model

        # configure gemini
        genai.configure(api_key=GEMINI_API_KEY)
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        self.gemini = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config)

    def query_gpt(self, prompt): 
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "user", "content": prompt}]
        }
        response = requests.post(ENDPOINTS["GPT-4"], headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

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
        response = self.gemini.generate_content(prompt)
        return response.text
    
    def query(self, prompt):
        if self.MODEL == "LLAMA": 
            return self.query_llama(prompt)
        if self.MODEL == "GPT-4": 
            return self.query_gpt(prompt)
        if self.MODEL == "GEMINI": 
            return self.query_gemini(prompt)

    def generate_principles(self, data_point): 
        pass

    def check_entailment(self, principle1, principle2): 
        with open("./prompts/entailment_prompt.txt") as f:
            prompt = f.read().format(principle1=principle1, principle2=principle2)
        if self.model == "GPT-4": 
            return self.query_gpt(prompt)
        elif self.mdoel == "GEMINI":
            return self.query_gemini(prompt)