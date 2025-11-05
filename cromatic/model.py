from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement in simple terms."}
    ]
)
print(response.choices[0].message.content)

## get the information from a page 

## def strike_tags 

## def stream_page (are some of these websites huge?) -> let's do just 10?

## def 