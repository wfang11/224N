from data.cai import CAI
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker
import datetime
import json

DEBUG = True # turn this flag on to only compare to just 1 CAI principle, NUM_EXAMPLE training examples

cai_dataset = CAI(dataset="train_sft")
data_points = cai_dataset.dataset
entailment_checker = EntailmentChecker("GEMINI")

NUM_CAI_PRINCIPLES = 1
NUM_EXAMPLES = 1
subset = [y for x, y in enumerate(data_points) if x < NUM_EXAMPLES] if DEBUG else data_points # weird hack to slice the dataset 

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

    ## run once! --> after written, no need
    # with open("principles/real_cai_principles.txt", "w") as f: 
    #     f.write('\n'.join(unique_principles))

    if DEBUG:
        return unique_principles[:NUM_CAI_PRINCIPLES]
    return unique_principles

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
    # make principles
    generated_principles = generate_principles() 

    # get cai principles
    real_principles = get_real_principles()

    # generate entailment scores
    scores = get_entailment_score(generated_principles, real_principles, mode='aggregate')
    with open(f"evaluation/entailment_score_logs/{datetime.datetime.now().json}", "a") as f:
        json.dumps(f)

    ### TODO: interpret this score and quantify it somehow? 
    print(scores)

    ### TODO: BLEU score? 

    ### TODO: manually log the principles and validate 
        # 1) principle generation
        # 2) entailment 

if __name__ == "__main__": 
    __main__()
