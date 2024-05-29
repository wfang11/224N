from raptor import BaseSummarizationModel
from models.model import Model

#Imports model class and then initializes a gemini model; prompt for summarization goes in prompt below.
class GeminiSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.model = Model("GEMINI")
    
#WILL: TO-DO: check if this prompt is good enough. We can make it better for better summarization. It has to be made by us; the default prompt did not work
    def summarize(self, context, max_tokens=500):
        prompt = f"You are given principles that an LLM should follow when generating responses to enhance safety. Write a summary of the following principles, that summarizes the given principles into one overarching principle: {context}:"
        summary = self.model.query_gemini(prompt)
        return summary
