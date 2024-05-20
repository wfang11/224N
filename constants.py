from dotenv import load_dotenv
import os

ENDPOINTS = {
    "LLAMA": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
    "GEMINI": "", # free of cost
    "GPT-4": "https://api.openai.com/v1/chat/completions",
}

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')