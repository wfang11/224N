import json
import random
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from models.model import Model

## initialize variables
KEYS = ['No Principles', 'All layers']
for i in range(4):
    KEYS.append([f'Layer {i}'])
gpt = Model("GPT-4")

START = 152
END = 700
for i in range(START, END):
    # skip mixtral errored points
    if i in [114, 388]: 
        continue

    ## get data
    with open(f"elos/generations/{i}.json") as f:
        data = json.loads(f.read())
    prompt = data["prompt"].strip()
    text_responses = {k:v[0] for k, v in data.items() if k != "prompt"}

    # Shuffle keys
    keys_list = list(text_responses.keys())
    random.shuffle(keys_list)

    ## Create the prompt
    with open("elos/SYS_gpt_ranking.txt") as f: 
        system_prompt = f.read()
    user_prompt = f""" 
    ### Prompt
    {prompt}

    ### Responses
    - Response 0: {text_responses[keys_list[0]]}
    - Response 1: {text_responses[keys_list[1]]}
    - Response 2: {text_responses[keys_list[2]]}
    - Response 3: {text_responses[keys_list[3]]}
    - Response 4: {text_responses[keys_list[4]]}
    - Response 5: {text_responses[keys_list[5]]}

    ## Response"""
    print('prompt: ', user_prompt)

    try: 
        ### call GPT
        response = gpt.query(system_prompt + "\n" + user_prompt)
        response = response.replace("```json", "").replace("```", "").strip()
        print('gpt_response: ', response)
        json_response = json.loads(response)
        gpt_rankings = json_response['rankings']

        ## unshuffle keys
        ranked_responses = {keys_list[i]: gpt_rankings[i] for i in range(len(keys_list))}
        original_order_rankings = {key: ranked_responses[key] for key in text_responses.keys()}

        print("\nOG: ", original_order_rankings)

        ## write to a file
        with open(f"elos/rankings/{i}.json", "a") as f: 
            f.write(json.dumps(original_order_rankings))
    except: 
        with open(f"elos/ranking_errored.txt", "a") as f:
            f.write(f"{i}\n")