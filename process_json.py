import json

filename = 'trees/GPT_1000_node_data.json'

with open(filename, 'r') as file:
    DATA = json.load(file)


def get_layer_1(data):
    layer_1_principles = []
    for key in data.keys():
        node = data[key]
        if node["Layer"] == 1:
            layer_1_principles.append(node["Text"])
    
    print(len(layer_1_principles))
    with open('layer_1_principles.txt', 'w') as file:
        for principle in layer_1_principles:
            file.write(principle + '\n')  
get_layer_1(DATA)