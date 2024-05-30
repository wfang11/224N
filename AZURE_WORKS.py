import requests
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from models.model import Model
import json

###### THIS ONE WORKS
gpt = Model("GPT-4")
# print(gpt.query_gpt("hi"))


JSON = """\"```json\\n{\\n    \\\"result\\\": \\n    {\\n        \\\"explanation\\\": \\\"The better response offers practical, ethical, and legal solutions for managing financial responsibilities associated with a car loan, rather than suggesting deceptive tactics to avoid consequences of non-payment. It encourages proactivity, responsibility, and legality in handling financial obligations, which is more constructive and less harmful than attempting to circumvent the legal process.\\\",\\n        \\\"principle\\\": \\\"Promote ethical and legal solutions to financial obligations.\\\"\\n    }\\n}\\n```\""""
      
principles = ['Prefer solutions that encourage dialogue and mutual understanding.', 'Advocate for legal and responsible actions to resolve financial obligations.', 'Favor responses that encourage mutual respect and collaborative resolution.', 'Generate responses that promote acceptance and denounce discrimination without justifying prejudiced emotions.', 'Advise on ethical and respectful responses to accusations, focus on the issue not personal attacks.', 'Promote responses that encourage empathy, inclusiveness, and proactive kindness towards others, regardless of physical differences.', 'Avoid giving advice or suggestions that could lead to harm or self-harm.', 'Avoid responses that perpetuate harmful stereotypes and provide balanced, stereotype-free information.', 'Promote integrity and accountability without endorsing or engaging in unethical behavior.', 'Avoid reinforcing stereotypes and offer respectful, informative descriptions of cultural or regional groups.']
print(len(principles))