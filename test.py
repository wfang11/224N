from data.cai import CAI

def main():
    cai_dataset = CAI(dataset="train_sft")
    real_principles = cai_dataset.extract_real_principles()
    unique_principles = set(real_principles)
    print("Total number of principles:", len(real_principles))
    print("Number of unique principles:", len(unique_principles))
    print("\nUnique Principles:")
    for principle in unique_principles:
        print(principle)

if __name__ == "__main__":
    main()
