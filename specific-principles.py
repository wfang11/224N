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
    START = 20
    END = 200

    errored = [34, 46, 52, 82, 93, 95, 103, 109, 110, 126, 146, 159, 175, 176, 179, 193, 195]
    subset = [y for x, y in enumerate(data_points) if x in errored]
    # subset = [y for x, y in enumerate(data_points) if START <= x < END] if DEBUG else data_points 

    # make principles --> validate how specific these are!
    principle_generator = PrincipleGenerator(model="GEMINI")

    principles = []
    errored = []
    error_indices = []
    ### TODO: migrate this functionality to the principle generation file
    for idx, data_point in enumerate(subset): 
        if idx % 10 == 0: 
            print("\n\n\n\n\Working on data point: ", idx)
        ## get harmful prompt
        prompt = data_point['prompt']
        better_response = data_point['response_0'] if data_point['safer_response_id'] == 0 else data_point['response_1']
        worse_response = data_point['response_1'] if data_point['safer_response_id'] == 0 else data_point['response_0']
        try: 
            principle = principle_generator.generate_principle(prompt, worse_response, better_response)
            principles.append(principle)
        except: 
            print("Errored on data point ", idx)
            errored.append(data_point)
            error_indices.append(idx)

    error_indices = [x + START for x in error_indices]
    print("Principles at the end: ", principles)
    print("Errored on: ", error_indices)

    with open(f"principles/generation_logs/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(principles))
    with open(f"principles/errored/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(errored))

if __name__ == "__main__": 
    __main__()
