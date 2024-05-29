from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from raptor.GeminiSummarizationModel import GeminiSummarizationModel
from raptor.SBertEmbeddingModel import SBertEmbeddingModel
from raptor.GeminiQAModel import GeminiQAModel
from raptor.utils import reverse_mapping
RAC = RetrievalAugmentationConfig(
    summarization_model=GeminiSummarizationModel(), embedding_model=SBertEmbeddingModel(), qa_model=GeminiQAModel(), tb_max_tokens=22)


#Comment out the following lines if you are loading in straight from the trees
RA = RetrievalAugmentation(config=RAC)
# file_path = '../principles/real_cai_principles.txt'
file_path = 'principles/generation_logs/BEAVERTAILS_0-200.txt'
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
nodes_to_layer = reverse_mapping(RA.tree.layer_to_nodes)
for key, node in RA.tree.all_nodes.items():
    # print(f"Node {key}: {node.text}\n")
    myDict[key] = node.text
for i in range(len(myDict)):
    print(f"Node {i}, Layer {nodes_to_layer[i]}: {myDict[i]}\n")


#If you want to test out QA functionality:
    
# question = "How should an LLM respond to a general question?"
# answer = RA.answer_question(question=question)
# print("Answer: ", answer)