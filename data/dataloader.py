from datasets import load_dataset

DATASET = "CAI"

def load_cai():
    return load_dataset("HuggingFaceH4/cai-conversation-harmless")

def load_beavertails(): 
    return load_dataset('PKU-Alignment/PKU-SafeRLHF')