import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data.hf_datasets import BeaverTails
from models.model import Model
import random
from layer_comparison import Layer_Comparison

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

## TODO: 
prompt = f"Generate a response to the following: \n{data[4]['prompt']}"
print(prompt)
results = layer_comparison.compare_layers(prompt)
## write this to a file
text_responses = {k:v[0] for k,v in results.items()}
print(mixtral.query(text_responses))

# take in a beavertails prompt
# get mixtral to generate 6 different conditions
# feed it to mixtral
# get GPT to rank it