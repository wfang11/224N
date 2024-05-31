from raptor import BaseSummarizationModel
from models.model import Model

#Imports model class and then initializes a gemini model; prompt for summarization goes in prompt below.
class GeminiSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.model = Model("GPT-4")
    
#WILL: TO-DO: check if this prompt is good enough. We can make it better for better summarization. It has to be made by us; the default prompt did not work
    def summarize(self, context, max_tokens=500):
        prompt = f"You are given principles that an LLM should follow to enhance safety when generating responses. Condense these principles into a single, concise summary that captures their essence. Aim for brevity and abstraction without exceeding 35 words. Output just your summary principle. Here are the principles: {context}:"
        # prompt = f"You are given principles that an LLM should follow when generating responses to enhance safety. Write a summary of the following principles, that summarizes the given principles into one summary principle. Output just the principle: {context}:"
        summary = self.model.query_gpt(prompt)
        return summary
