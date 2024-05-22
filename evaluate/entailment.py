from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from models.model import Model

model_query = Model()

class EntailmentChecker:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
        self.bert = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
        self.bert.eval()

        self.model = Model(model)

    def check_entailment_bert(self, principle1, principle2):
        label_mapping = ['contradiction', 'entailment', 'neutral']
        tokenized_input = self.tokenizer(principle1, principle2, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.bert(**tokenized_input).logits
            result_label = label_mapping[outputs.argmax(dim=1).item()]
        return result_label

    def check_entailment_api(self, principle1, principle2):
        return self.model.check_entailment(principle1, principle2)

    def evaluate_entailment(self, principles1, principles2, mode='pairwise', method='GPT'):
        results = []
        if mode == 'pairwise':
            for princ1 in principles1:
                for princ2 in principles2:
                    if method == 'bert':
                        result_label = self.check_entailment_bert(princ1, princ2)
                    else:
                        result_label = self.check_entailment_api(princ1, princ2, method)
                    results.append({
                        'from_principle': princ1,
                        'to_principle': princ2,
                        'result': result_label
                    })

        elif mode == 'aggregate':
            concatenated_principles2 = '. '.join(principles2) + '.'
            for princ1 in principles1:
                if method == 'bert':
                    result_label = self.check_entailment_bert(princ1, concatenated_principles2)
                else:
                    result_label = self.check_entailment_api(princ1, concatenated_principles2, method)
                results.append({
                    'from_principle': princ1,
                    'to_aggregate_principles': concatenated_principles2,
                    'result': result_label
                })

        return results

