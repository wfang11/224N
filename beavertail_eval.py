import json
import datetime
from models.model import Model 
from data.hf_datasets import BeaverTails
import time
import random
from layer_comparison import Layer_Comparison

def get_data():
    data_points = BeaverTails('test').dataset
    ### Filter data points to highest quality: safer response (classified: safe) is aligned with better response, other response is not safe
    data_points = [x for x in data_points if x['is_response_0_safe'] != x['is_response_1_safe'] and x['safer_response_id'] == x['better_response_id']]
    return data_points

def parse_hf_response(response): 
    ## process output
    output_start = response.find("Answer\n") + len("Answer") + 1
    answer = response[output_start:].strip()[0] 
    answer = int(answer) - 1
    return answer

data = get_data() # 1235 data points
random.shuffle(data)

TIME = "2024-06-05 01:47:29.361176"
START = 0
END = 0

for i in range(START, END + 1):
    file = f"evaluation/beavertails_pref/dp{i}_{TIME}.json"
    with open(file, "r") as f: 
        response = json.loads(f.read())
        text = response["answer"]["No Principles"][0]
        print(parse_hf_response(text))

# ### get score
# correct = 0
# for i in range(NUM_DATA_POINTS): 
#     if str(i) in submission: 
#         pred = submission[f"{i}"]
#     else: 
#         pred = "-1"
#     actual = data[i]["better_response_id"]
#     print()
#     if int(pred) == int(actual): 
#         correct += 1
# print("NUM CORRECT: ", correct)
# print("NUM TOTAL: ", NUM_DATA_POINTS)
# print("NUM ERRORED: ", len(errored))
# print("ACCURACY: ", correct / NUM_DATA_POINTS)