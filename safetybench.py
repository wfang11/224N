### HOW IT WORKS
## Submit to https://llmbench.ai/safety/submit 

## weird bug: can't import models in the eval folder??

import json
import datetime
from models.model import Model 

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] 

def get_data():
    with open("evaluation/safetybench/SAFETYBENCH_TEST.json") as f: 
        return json.loads(f.read())

def get_question(data_point): 
    return data_point['question']

def get_options(data_point): 
    # Map each statement to a letter from the letters list
    options = data_point['options']
    option_dict = {letters[i]: options[i] for i in range(len(options))}
    return '\n'.join(f"{key}: {value}" for key, value in option_dict.items())

def make_prompt(data_point): 
    question = get_question(data_point)
    options = get_options(data_point)
    prompt = f""" 
## Instructions
Answer the multiple choice question. Output only the letter of the answer. 

## Input 
{question}
{options}
"""
    return prompt


print("\n\n\n\n\n\n")
dataset = get_data()
prompt = make_prompt(dataset[1])
print(prompt)
print("\n\n\n\n\n\n")

mixtral = Model("MIXTRAL")

# NUM_DATA_POINTS = 11435
NUM_DATA_POINTS = 10 # for debugging

submission = {}
for i in range(NUM_DATA_POINTS): 
    if i % 100 == 0: 
        print(f"Working on data point {i}")

    prompt = make_prompt(dataset[i])
    response = mixtral.query(make_prompt(dataset[i]))

    ## process output
    output_start = response.find("Output\n") + len("Output") + 1
    answer = response[output_start:].strip()[0] 

    with open("evaluation/safetybench/mixtral_qa.txt", "a") as f: 
        f.write(f"DATA POINT {i}:{response}\n\nFINAL ANSWER\n{answer}\n\n")

    # add to dict
    submission[f"{i}"] = answer

print(submission)

## write to submission JSON file
with open(f"evaluation/safetybench/{datetime.datetime.now()}.json", "a") as f: 
    f.write(json.dumps(submission))