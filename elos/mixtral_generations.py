import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data.hf_datasets import BeaverTails
from models.model import Model
import random
from layer_comparison import Layer_Comparison
import json

## initialize models and data
beavertails = BeaverTails()
layer_comparison = Layer_Comparison()
mixtral = Model("MIXTRAL")
data = [x for x in beavertails.dataset]
KEYS = ['No Principles', 'All layers']
for i in range(4):
    KEYS.append([f'Layer {i}'])

random.seed(42)
random.shuffle(data)

## get only the response, not the prompt
def parse_response(response, prompt): 
    start_index = response.find(prompt) + len(prompt)
    return response[start_index:].strip()

errored = []
START = 0
END = 10
for i in range(START, END):
    try: 
        # obtain generations
        prompt = f"\nGenerate a response to the following: \n{data[i]['prompt']}"
        print(prompt)
        response = layer_comparison.compare_layers(prompt)
        results = {k:list(v) for k, v in response.items()}

        # get only response
        for k, v in results.items():
            results[k][0] = parse_response(results[k][0], prompt)

        # add prompt
        results["prompt"] = prompt

        with open(f"elos/generations/{i}.json", "a") as f: 
            f.write(json.dumps(results))
    except:
        print("ERRORED on data point ", i)
        with open(f"elos/generations/errored.txt", "a") as f: 
            f.write(f"{i}\n")