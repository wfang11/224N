from data.hf_datasets import CAI, BeaverTails
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker
import datetime
import json

DEBUG = True # turn this flag on to only compare to just 1 CAI principle, NUM_EXAMPLE training examples

# cai_dataset = CAI(dataset="train_sft")
# data_points = cai_dataset.dataset

beavertail_dataset = BeaverTails("330k_train")
data_points = beavertail_dataset.dataset 

NUM_EXAMPLES = 10


### Generate all principles!
def generate_principles(subset):
    principle_generator = PrincipleGenerator(model="GPT-4")
    summary_principles = principle_generator.generate_principles(subset)
    # log the generated principles
    with open(f"principles/generation_logs/{datetime.datetime.now()}.txt", "a") as f: 
        f.write('\n'.join(summary_principles))
    return summary_principles


def __main__(): 
    # filter data from beavertails --> where one is good and one is bad? 
    subset = [y for x, y in enumerate(data_points) if x < NUM_EXAMPLES] if DEBUG else data_points # weird hack to slice the dataset 
    

    # make principles --> validate how specific these are!
    principle_generator = PrincipleGenerator(model="GEMINI")

    print(subset)

    # print(generate_principles)


if __name__ == "__main__": 
    __main__()
