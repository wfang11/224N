import pandas as pd
import re
import random
from pathlib import Path
#********************STEP 1 of principle generation manual validation********************
def extract_and_create_csv(file_path):
    data_list = []
    file_content = Path(file_path).read_text()
    data_points = re.split(r"RUNNING FOR DATA POINT \d+:\n", file_content)[1:]  # Split and remove the first empty element
    print(data_points[0])
    for data in data_points:
        harmful_prompt = re.search(r"### Harmful Prompt\n(.+)", data).group(1)
        print("Harmful prompt: \n", harmful_prompt)
        undesirable_response = re.search(r"### Undesirable Response\n(.+)", data).group(1)
        print("Undesirable response: \n", undesirable_response)

        better_response = re.search(r"### Better Response\n(.+)", data).group(1)
        print("Better response: \n", better_response)

        ai_principle = re.search(r"principle([^\.]*\.)", data)
        if ai_principle:
            ai_principle=ai_principle.group(1)
        else:
            continue

        print("Principle: \n", ai_principle[10:])
        data_point = {
            "Harmful Prompt": harmful_prompt,
            "Undesirable Response": undesirable_response,
            "Better Response": better_response,
            "AI Generated Principle": ai_principle[10:]
        }
        data_list.append(data_point)
    df = pd.DataFrame(data_list)
    df.to_csv("man_val_step_1.csv", index=False)
# file_path = "models/GPT_0-999.txt"
# extract_and_create_csv(file_path)

#********************STEP 2 of principle generation manual validation********************
def create_sample_for_manual_filling(input_csv_path, sample_size):
    # Load the original data
    df = pd.read_csv(input_csv_path)
    
    # Sample the data
    df_sampled = df.sample(n=sample_size, random_state=1)
    df_sampled['Human Generated Principle'] = ""  # Add an empty column for manual filling
    
    # Save to new CSV
    output_csv_path = "sample_for_manual_filling.csv"
    df_sampled.to_csv(output_csv_path, index=False)
    print(f"Sampled data saved to '{output_csv_path}'. Ready for manual principle generation.")

# input_csv_path = 'man_val_step_1.csv'
# sample_size = 300
# create_sample_for_manual_filling(input_csv_path=input_csv_path, sample_size=sample_size)

#********************STEP 3 of principle generation manual validation********************
def prepare_comparison_csv(input_filled_csv_path):
    # Load the manually filled data
    df = pd.read_csv(input_filled_csv_path)

    # Initialize new columns
    df['Principle 1'] = None
    df['Principle 2'] = None
    df['AI Generated'] = ""  
    df['Better'] = "" 

    # Randomly assign AI and human-generated principles to Principle 1 and Principle 2
    for idx, row in df.iterrows():
        responses = [row['AI Generated Principle'], row['Human Generated Principle']]
        random.shuffle(responses)
        df.at[idx, 'Principle 1'] = responses[0]
        df.at[idx, 'Principle 2'] = responses[1]
    output_csv_path = "comparison_ready.csv"
    df.to_csv(output_csv_path, index=False)

# input_filled_csv_path = "sample_for_manual_filling.csv"
# prepare_comparison_csv(input_filled_csv_path)



#********************STEP 1 of principle scoring manual validation********************
def create_validation_csv_with_scoring(input_csv_path, sample_size):
    # Load the manually filled data
    df = pd.read_csv(input_csv_path)
    
    # Sample the data
    df_sampled = df.sample(n=sample_size, random_state=2)
    
    # Add empty columns for scoring
    df_sampled['Relevance'] = ""
    df_sampled['Accuracy'] = ""
    df_sampled['Clarity'] = ""
    df_sampled['Specificity'] = ""
    
    # Save to new CSV
    output_csv_path = "validation_with_scoring.csv"
    df_sampled.to_csv(output_csv_path, index=False)
    print(f"Validation data with scoring columns saved to '{output_csv_path}'.")

# input_csv_path = 'man_val_step_1.csv'
# sample_size = 300
# create_validation_csv_with_scoring(input_csv_path=input_csv_path, sample_size=sample_size)
    
    