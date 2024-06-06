FILEPATH = "principles/generation_logs/GPT_TEMP_BEAVERTAILS.txt"
def reformat_principles(filepath):
    principles_list = []
    with open(filepath, 'r') as file:
        for line in file:
            principles_list.append(line.strip())
    counter = 0
    for i in range(len(principles_list) -1):
        if (len(principles_list[i]) + len(principles_list[i]) < 212):
            counter += 1
            print(i)
    print(f"{counter}/{len(principles_list)} are too short")
reformat_principles(FILEPATH)