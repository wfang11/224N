from models.model import Model
import json
from evaluation.entailment import EntailmentChecker

class PrincipleGenerator: 
    def __init__(self, model, entail_model="GEMINI"): 
        self.model = Model(model)
        self.entail_model = EntailmentChecker(entail_model)

    ### TODO: update this from BeaverTails' data format
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

    ### TODO: update this!!
    def generate_principles(self, data_points):
        i = 0
        principles = []
        for data_point in data_points:
            if i % 5 == 0: 
                print(f"Generating principle for data point {i}")
            harmful_prompt, undesirable_response, better_response = self.process_datapoint(data_point)

            #### also give ground truth principle?? idk --> need to have this for eval on utterance level 
            principle = self.generate_principle(harmful_prompt, undesirable_response, better_response)
            print("running entailment check for ..")
            print(f"principle: {principle}")
            print(f"principles: {principles}")

            # ### REMOVE THIS CALL! 
            # if len(principles) == 0 or self.should_add_principle(principle, principles):
            #     principles.append(principle)
                
            i += 1
        print("FINAL PRINCIPLES")
        print(principles)
        return principles
    
    # def should_add_principle(self, principle, existing_principles):  
    #     concatenated_principles = '. '.join(existing_principles) + '.'
    #     entailing = self.entail_model.check_entailment_api(concatenated_principles, principle)
    #     ## ESPECIALLY this call here --> dependent on formatting and types. 
    #     if str(entailing) == "1": 
    #         print("I SHOULD NOT ADD THIS TO PRINCIPLES")
    #         return False
    #     print("I AM ADDING THIS TO PRINCIPLES")
    #     print(entailing)
    #     return True

    ### TODO: create self-critique mechanism
    def self_critique(): 
        pass

    