from datasets import load_dataset

class CAI: 
    def __init__(self, dataset="train_sft"): 
        self.dataset = load_dataset("HuggingFaceH4/cai-conversation-harmless")[dataset]

    def extract_real_principles(self):
        return [dp['revision_prompt'] for dp in self.dataset]
    
class BeaverTails:
    def __init__(self, dataset="train"): 
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")[dataset]