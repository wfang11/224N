from models.model import Model

class PrincipleGenerator: 
    def __init__(self, model): 
        self.model = Model(model)

    def process_datapoint(self, data_point):
        harmful_prompt = data_point['init_prompt']
        undesirable_response = data_point['init_response']
        better_response = data_point['revision_response']
        return harmful_prompt, undesirable_response, better_response

    def generate_principle(self, prompt):
        return self.model.query(prompt)

    def generate_principles(self, data_points):
        principles = []
        for data_point in data_points:
            print(data_point)
            harmful_prompt, undesirable_response, better_response = self.process_datapoint(data_point)
            user_prompt = f"{harmful_prompt} Undesirable Response: {undesirable_response} Better Response: {better_response}"
            principle = self.generate_principle(user_prompt)
            principles.append(principle)  # change this line to add_principle_to_principles(principle), which also calls entailment
        return principles