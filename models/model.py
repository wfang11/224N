import requests
from models.constants import ENDPOINTS, OPENAI_API_KEY, HF_API_KEY, GEMINI_API_KEY
import google.generativeai as genai

PLACEHOLDER_PROPMT = "Hi! How's it going?"

class Model: 
    def __init__(self, model): 
        self.MODEL = model

        genai.configure(api_key=GEMINI_API_KEY)
        # Set up the model
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
        
        self.gemini = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    def query_gpt(self, system_prompt="", user_prompt=PLACEHOLDER_PROPMT): 
        with open("models/prompts/gpt_system_prompt.txt") as f:         
            system_prompt = f.read()

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

        prompt_parts = [
            "Hello! Give me a fun fact about cows."
        ]
        
        response = self.gemini.generate_content(prompt_parts)
        return response.text
        # print(response.text)
        # print(ENDPOINTS["GEMINI"])
        # data = {
        #     "contents": [{
        #         "parts":[{
        #         "text": prompt}]
        #     }]
        # }
        # response = requests.post(ENDPOINTS["GEMINI"], json=data)
        # return response.json()["candidates"][0]["content"]["parts"][0]["text"]

#****** ADDED BY VIKRAM *********
    def check_entailment_gpt(self, principle1, principle2):
        with open("./prompts/entailment_prompt.txt") as f:
            system_prompt = f.read().format(principle1=principle1, principle2=principle2)
        user_prompt = "Evaluate the entailment."

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = requests.post(ENDPOINTS["GPT-4"], headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']

    def check_entailment_llama(self, principle1, principle2):
        with open("./prompts/entailment_prompt.txt") as f:
            prompt = f.read().format(principle1=principle1, principle2=principle2)

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}"
        }
        data = {
            "inputs": prompt
        }
        response = requests.post(ENDPOINTS["LLAMA"], headers=headers, json=data)
        return response.json()[0]["generated_text"]

    def check_entailment_gemini(self, principle1, principle2):
        with open("./prompts/entailment_prompt.txt") as f:
            prompt = f.read().format(principle1=principle1, principle2=principle2)

        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        response = requests.post(ENDPOINTS["GEMINI"], json=data)
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]

#****** [END] ADDED BY VIKRAM *********

    #### WILL TODO: make this into a generate_principle function
    def query(self, system_prompt=None, user_prompt=None):
        if self.MODEL == "LLAMA": 
            return self.query_llama()
        if self.MODEL == "GPT-4": 
            return self.query_gpt()
        if self.MODEL == "GEMINI": 
            return self.query_gemini()
        

    #### WILL TODO: make this into an evaluate function
    def evaluate(self, data): 
        pass