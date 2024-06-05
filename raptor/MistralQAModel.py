from raptor import BaseQAModel
from models.model import Model

class MistralQAModel(BaseQAModel):
    def __init__(self):
        self.model = Model("MIXTRAL")

    def answer_question(self, context, question):
        """
        Generates an answer to a given question based on the provided context
        using the Gemini model configured in the Model class.

        Args:
            context (str): The text that provides context for the question.
            question (str): The question that needs an answer.

        Returns:
            str: The answer generated by the model.
        """
        prompt = f"You are given the following principles to adhere to as context: {context}\nKeeping these principles in mind, respond to the following prompt: {question}"
        answer = self.model.query(prompt)
        return answer