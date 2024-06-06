# What we want: want to create 4 different QA models - one for each layer - such that i can iterate through a dataset, call each model one by one, get a response. so potential scheme can be:
    
    # - create four RA objects (dynamically?) maybe have vars treeconfigl1, treeconfigl2,.... in t

from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from raptor.GPTSummarizationModel import GPTSummarizationModel
from raptor.GPTQAModel import GPTQAModel
from raptor.SBertEmbeddingModel import SBertEmbeddingModel
from raptor.GeminiQAModel import GeminiQAModel
from raptor.MistralQAModel import MistralQAModel
from raptor.utils import reverse_mapping
import json
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.tree_retriever import TreeRetriever, TreeRetrieverConfig
from models.model import Model

mixtral = Model("MIXTRAL")
gpt = Model("GPT-4")

# Load the RAPTOR tree
SAVE_PATH = "raptor/raptor_final_tree"

#ra config. initializing outside function so that i need to initialize only once
ra_config = RetrievalAugmentationConfig(
    # summarization_model=GPTSummarizationModel(), embedding_model=SBertEmbeddingModel(), qa_model=MistralQAModel()
    summarization_model=GPTSummarizationModel(), embedding_model=SBertEmbeddingModel(), qa_model=MistralQAModel()
)

class Layer_Comparison:

    def __init__(self): 
        self.ra = RetrievalAugmentation(tree=SAVE_PATH, config=ra_config)

    #Configure raptor for a specific start layer and query the model. if we want to test just one layer, layer_start should be that layer and num_layers should be 1
    def configure_and_query(self, ra, start_layer, num_layers, prompt, collapse_tree):
        
        #Note: mistral is the QA model. all prompts we want to put into mistral, we pass through the QA model of the RA object corresponding to that layer
        #can optionally add top_k parameter. need to discuss whether or not to include this

        return ra.answer_question(question=prompt, start_layer=start_layer, num_layers=num_layers, collapse_tree=collapse_tree, return_layer_information=True)

    #takes in a prompt. then loops through all layers; configs an RA object for each layer. outputs the response

    def compare_layers(self, prompt):
        results = {}
        results['No Principles'] = (mixtral.query("[INST]" + prompt + "[/INST]"), [])
        # Goes through all layers.
        prompt = "Do not explicitly mention 'principles', but rather use the principles to guide your response." + prompt
        prompt = "[INST]" + prompt + "[/INST]"
        for i in range(4):
            results[f'Layer {i}'] = self.configure_and_query(self.ra, i, 1, prompt, collapse_tree=False)    
        # Save or print results; we should discuss how we want to save results. 
            
        results['All layers'] = self.ra.answer_question(question=prompt, collapse_tree=True, return_layer_information=True)
        # print(json.dumps(results, indent=4))

        return results
    # compare_layers("How to use a gun?")
