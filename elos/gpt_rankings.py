import json
import random

## initialize variables
KEYS = ['No Principles', 'All layers']
for i in range(4):
    KEYS.append([f'Layer {i}'])

i = 0
with open(f"elos/generations/{i}.json") as f:
    data = json.loads(f.read())
prompt = data["prompt"].strip()
text_responses = {k:v[0] for k, v in data.items() if k != "prompt"}

# Shuffle keys
keys_list = list(text_responses.keys())
random.shuffle(keys_list)


############### TODO #1: make the prompt ############################
# condition 0 -> text_responses[keys_list[0]]
# condition 1 -> text_responses[keys_list[1]]
# ...etc. 
############### TODO #1: make the prompt ############################


############### TODO #2: call gpt instead of this ############################
gpt_rankings = list(range(1, len(keys_list) + 1))  # Example GPT rankings
############### TODO #2: call gpt instead of this ############################


ranked_responses = {keys_list[i]: gpt_rankings[i] for i in range(len(keys_list))}
original_order_rankings = {key: ranked_responses[key] for key in text_responses.keys()}

print("OG: ", original_order_rankings)


############### TODO #3: write to a file ############################

############### TODO #3: write to a file ############################