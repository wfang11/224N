from datasets import load_dataset

DATASET = "CAI"

def get_cai_data():
    return load_dataset("HuggingFaceH4/cai-conversation-harmless")

def get_beavertails_data(): 
    return load_dataset('PKU-Alignment/PKU-SafeRLHF')