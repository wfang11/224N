from data.hf_datasets import CAI, BeaverTails
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker
import datetime
import json

DEBUG = True # turn this flag on to only compare to just 1 CAI principle, NUM_EXAMPLE training examples

beavertail_dataset = BeaverTails("train")
data_points = beavertail_dataset.dataset 

### Filter data points to highest quality: safer response (classified: safe) is aligned with better response, other response is not safe
data_points = [x for x in data_points if x['is_response_0_safe'] != x['is_response_1_safe'] and x['safer_response_id'] == x['better_response_id']]

### Generate all principles!
def generate_principles(subset):
    principle_generator = PrincipleGenerator(model="GPT-4")
    summary_principles = principle_generator.generate_principles(subset)
    # log the generated principles
    with open(f"principles/generation_logs/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(summary_principles))
    return summary_principles


def __main__(): 

    START = 10
    END = 20
    subset = [y for x, y in enumerate(data_points) if START <= x < END] if DEBUG else data_points # weird hack to slice the dataset 

    # make principles --> validate how specific these are!
    principle_generator = PrincipleGenerator(model="GEMINI")

    principles = []
    ### TODO: migrate this functionality to the principle generation file
    for data_point in subset: 
        ## get harmful prompt
        prompt = data_point['prompt']
        better_response = data_point['response_0'] if data_point['safer_response_id'] == 0 else data_point['response_1']
        worse_response = data_point['response_1'] if data_point['safer_response_id'] == 0 else data_point['response_0']
        principle = principle_generator.generate_principle(prompt, worse_response, better_response)
        principles.append(principle)

    print("Principles at the end: ", principles)

    with open(f"principles/generation_logs/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(principles))

if __name__ == "__main__": 
    __main__()
