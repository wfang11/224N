from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from raptor.GeminiSummarizationModel import GeminiSummarizationModel
from raptor.SBertEmbeddingModel import SBertEmbeddingModel
from raptor.GeminiQAModel import GeminiQAModel
from raptor.utils import reverse_mapping
RAC = RetrievalAugmentationConfig(
    summarization_model=GeminiSummarizationModel(), embedding_model=SBertEmbeddingModel(), qa_model=GeminiQAModel(), tb_max_tokens=5)
import json


#Comment out the following lines if you are loading in straight from the trees
RA = RetrievalAugmentation(config=RAC)
file_path = 'principles/generation_logs/GPT_0-999.txt'

with open(file_path, 'r') as file:
    text = file.read()
RA.add_documents(text)
SAVE_PATH = "raptor/trees"
RA.save(SAVE_PATH)
#Comment till here 

#Comment this next line in if you just want to just load in RA object from saved tree in trees
# RA = RetrievalAugmentation(tree=SAVE_PATH, config=RAC)

#Loads in all nodes from the tree, sorts them and then prints their text
myDict = {}
children = {}

nodes_to_layer = reverse_mapping(RA.tree.layer_to_nodes)
for key, node in RA.tree.all_nodes.items():
    children[key] = node.children
    myDict[key] = node.text

node_info = {}
for i in range(len(myDict)):
    node_info[i] = {
        "Node": i,
        "Layer": nodes_to_layer[i],
        "Text": myDict[i],
        "Children": list(children[i])
    }

    print(f"Node {i}, Layer {nodes_to_layer[i]}: {myDict[i]}\n")
    print(f"Children of Node {i}: {children[i]}")

json_data = json.dumps(node_info, indent=4)

with open("GPT_1000_node_data.json", "w") as json_file:
    json_file.write(json_data)
