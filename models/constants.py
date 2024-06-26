from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

ENDPOINTS = {
    "LLAMA": "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
    "GEMINI": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
    "GPT-4": "https://api.openai.com/v1/chat/completions",
}