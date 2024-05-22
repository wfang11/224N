from data.cai import CAI
from models.model import Model
from principles.principle_generator import PrincipleGenerator
from evaluation.entailment import EntailmentChecker

def run_evaluation():
    cai_dataset = CAI(dataset="train_sft")
    data_points = cai_dataset.dataset
    data_points = [data_points[0]]
    principle_generator = PrincipleGenerator(model="GEMINI")

    # first, generate the principles using generate_principles from principle_generator.py
    summary_principles = principle_generator.generate_principles(data_points)

    # gotta extract real principles from dataset, we need to figure out how to do this!! they're not formatted well. 
    real_principles = cai_dataset.extract_real_principles()
    entailment_checker = EntailmentChecker("GEMINI")
    
    # checks if single generated princ entails single real princ
    pairwise_generated_to_real = entailment_checker.evaluate_entailment(summary_principles, real_principles, mode='pairwise')
 
    # checks if single real princ entails generated real princ
    pairwise_real_to_generated = entailment_checker.evaluate_entailment(real_principles, summary_principles, mode='pairwise')

    # checks if all real principles entail single generated princ
    aggregate_real_to_generated = entailment_checker.evaluate_entailment(real_principles, summary_principles, mode='aggregate')

    # checks if all generated principles entail single real princ
    aggregate_generated_to_real = entailment_checker.evaluate_entailment(summary_principles, real_principles, mode='aggregate')
    results = {"pairwise_generated_to_real": pairwise_generated_to_real, "pairwise_real_to_generated": pairwise_real_to_generated, "aggregate_real_to_generated": aggregate_real_to_generated, "aggregate_generated_to_real": aggregate_generated_to_real}
    return results

run_evaluation()