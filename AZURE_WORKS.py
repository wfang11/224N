import requests
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from models.model import Model

###### THIS ONE WORKS
gpt = Model("GPT-4")
print(gpt.query_gpt("hi"))

