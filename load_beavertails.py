from datasets import load_dataset

# Load the dataset
dataset = load_dataset('PKU-Alignment/PKU-SafeRLHF')

# Print some information about the dataset
print(dataset)
print("Sample data:", dataset['train'][0])