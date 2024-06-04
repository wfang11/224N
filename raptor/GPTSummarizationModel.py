from raptor import BaseSummarizationModel
from models.model import Model

#Imports model class and then initializes a GPT model; prompt for summarization goes in prompt below.
class GPTSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.model = Model("GPT-4")
        with open('summary_instruction.txt', 'r') as file:
            self.base_prompt = file.read()
    

    def summarize(self, context, max_tokens=500):
        complete_prompt = self.base_prompt.format(context=context)
        summary = self.model.query_gpt(complete_prompt)
        return summary
