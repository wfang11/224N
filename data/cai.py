from datasets import load_dataset

class CAI: 
    def __init__(self, dataset="train_sft"): 
        self.dataset = load_dataset("HuggingFaceH4/cai-conversation-harmless")[dataset]
