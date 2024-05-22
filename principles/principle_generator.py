from models.model import Model
import json

class PrincipleGenerator: 
    def __init__(self, model): 
        self.model = Model(model)

    def process_datapoint(self, data_point):
        harmful_prompt = data_point['init_prompt']
        undesirable_response = data_point['init_response']
        better_response = data_point['revision_response']
        return harmful_prompt, undesirable_response, better_response

    def generate_principle(self, harmful_prompt, undesirable_response, better_response):
        with open("models/prompts/generate_system.txt") as f:
            sys = f.read()
        with open("models/prompts/generate_user.txt") as f:
            user = f.read().format(harmful_prompt=harmful_prompt, undesirable_response=undesirable_response, better_response=better_response)
        prompt = sys + '\n' + user
        response = self.model.query(prompt)
        return json.loads(response)['result']['principle']

    ## TODO: make this better. The prompt is not working rn. 
    def generate_principles(self, data_points):
        i = 0
        principles = []
        for data_point in data_points:
            if i % 5 == 0: 
                print(f"Generating principle for data point {i}")
            harmful_prompt, undesirable_response, better_response = self.process_datapoint(data_point)
            principle = self.generate_principle(harmful_prompt, undesirable_response, better_response)
            principles.append(principle)  
            i += 1
        return principles
    
    ## TODO: add only if it is entailed. 
    def add_principle_to_principles(): 

        pass