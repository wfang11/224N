from models.model import Model
import json
from evaluation.entailment import EntailmentChecker

class PrincipleGenerator: 
    def __init__(self, model, entail_model="GEMINI"): 
        self.model = Model(model)
        self.entail_model = EntailmentChecker(entail_model)

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
        print(response)
        return json.loads(response)['result']['principle']
    
    ## TODO: should test this functionality and make sure that it works.
    def should_add_principle(self, principle, existing_principles):  
        concatenated_principles = '. '.join(existing_principles) + '.'
        entailing = self.entail_model.check_entailment_api(concatenated_principles, principle)
        ## ESPECIALLY this call here --> dependent on formatting and types. 
        if str(entailing) == "1": 
            return True
        return False

    def generate_principles(self, data_points):
        i = 0
        principles = []
        for data_point in data_points:
            if i % 5 == 0: 
                print(f"Generating principle for data point {i}")
            harmful_prompt, undesirable_response, better_response = self.process_datapoint(data_point)
            principle = self.generate_principle(harmful_prompt, undesirable_response, better_response)
            if self.should_add_principle(): 
                principles.append(principle)  
            i += 1
        return principles