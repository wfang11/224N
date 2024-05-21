from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from model import Model

model_query = Model()

class EntailmentChecker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')
        self.model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
        self.model.eval()

    def check_entailment_bert(self, principle1, principle2):
        label_mapping = ['contradiction', 'entailment', 'neutral']
        tokenized_input = self.tokenizer(principle1, principle2, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**tokenized_input).logits
            result_label = label_mapping[outputs.argmax(dim=1).item()]
        
        return result_label

    def check_entailment_api(self, principle1, principle2, model_type):
        if model_type == "GPT":
            return model_query.check_entailment_gpt(principle1, principle2)
        elif model_type == "LLAMA":
            return model_query.check_entailment_llama(principle1, principle2)
        elif model_type == "GEMINI":
            return model_query.check_entailment_gemini(principle1, principle2)

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

