from data.cai import CAI
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker
import datetime

NUM_EXAMPLES = 2
DEBUG = True # turn this flag on to only compare to just 1 CAI principle
cai_dataset = CAI(dataset="train_sft")
data_points = cai_dataset.dataset
subset = [y for x, y in enumerate(data_points) if x < NUM_EXAMPLES] # weird hack to slice the dataset 
entailment_checker = EntailmentChecker("GEMINI")

def generate_principles():
    principle_generator = PrincipleGenerator(model="GEMINI")
    summary_principles = principle_generator.generate_principles(subset)
    # log the generated principles
    with open(f"principles/generation_logs/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(summary_principles))
    return summary_principles

def get_real_principles():
    # gotta extract real principles from dataset, we need to figure out how to do this!! they're not formatted well. 
    real_principles = cai_dataset.extract_real_principles()
    unique_principles = list(set(real_principles))

    ## run once!
    # with open("principles/real_cai_principles.txt", "w") as f: 
    #     f.write('\n'.join(unique_principles))

    if DEBUG:
        return unique_principles[:2]
    return unique_principles

## WILL TODO: add more prompts fpr checking entailment
def get_entailment_score(generated_principles, real_principles, mode='aggregate'):

    if mode == "pairwise": 
        # # checks if single generated princ entails single real princ
        pairwise_generated_to_real = entailment_checker.evaluate_entailment(generated_principles, real_principles, mode='pairwise')

        # # checks if single real princ entails generated real princ
        pairwise_real_to_generated = entailment_checker.evaluate_entailment(real_principles, generated_principles, mode='pairwise')
        return {"pairwise_generated_to_real": pairwise_generated_to_real, "pairwise_real_to_generated": pairwise_real_to_generated}

    elif mode == "aggregate": 
        # checks if all real principles entail single generated princ
        aggregate_real_to_generated = entailment_checker.evaluate_entailment(real_principles, generated_principles, mode='aggregate')

        # checks if all generated principles entail single real princ
        aggregate_generated_to_real = entailment_checker.evaluate_entailment(generated_principles, real_principles, mode='aggregate')
        results = {"aggregate_real_to_generated": aggregate_real_to_generated, "aggregate_generated_to_real": aggregate_generated_to_real}
        return results

def __main__(): 
    generated_principles = generate_principles()
    real_principles = get_real_principles()
    scores = get_entailment_score(generated_principles, real_principles, mode='aggregate')
    print(scores)

if __name__ == "__main__": 
    __main__()
