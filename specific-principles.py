from data.hf_datasets import CAI, BeaverTails
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker
import datetime
import json
import logging
import random

DEBUG = True # turn this flag on to only compare to just 1 CAI principle, NUM_EXAMPLE training examples

beavertail_dataset = BeaverTails("train")
data_points = beavertail_dataset.dataset 

### Filter data points to highest quality: safer response (classified: safe) is aligned with better response, other response is not safe
data_points = [x for x in data_points if x['is_response_0_safe'] != x['is_response_1_safe'] and x['safer_response_id'] == x['better_response_id']]

def __main__(): 
    START = 0
    END = 1000

    # prepare data subset
    subset = [y for _, y in enumerate(data_points)]
    random.seed(42)
    random.shuffle(subset)
    subset = [y for x, y in enumerate(subset) if START <= x < END] if DEBUG else data_points 

    # make principles --> validate how specific these are!
    principle_generator = PrincipleGenerator(model="GPT-4")

    principles = []
    for idx, data_point in enumerate(subset): 
        print("\n\n\n\nWorking on data point: ", idx + START)
        ## get harmful prompt
        prompt = data_point['prompt']
        better_response = data_point['response_0'] if data_point['safer_response_id'] == 0 else data_point['response_1']
        worse_response = data_point['response_1'] if data_point['safer_response_id'] == 0 else data_point['response_0']
        try: 
            principle = principle_generator.generate_principle(prompt, worse_response, better_response, idx + START)
            principles.append(principle)
            with open(f"principles/generation_logs/GPT_TEMP_BEAVERTAILS.txt", "a") as f: 
                f.write(principle + "\n")
        except: 
            print("Errored on data point ", idx + START)
            with open(f"principles/errored/GPT_TEMP_BEAVERTAILS", "a") as f: 
                f.write(str(idx + START) + "\n")
    print("all done!")

if __name__ == "__main__": 
    __main__()
