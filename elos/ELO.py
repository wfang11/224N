import numpy as np
import matplotlib.pyplot as plt
from multielo import MultiElo
import random
import json
KEYS = ['No Principles', 'All layers']
for i in range(4):
    KEYS.append([f'Layer {i}'])
print(KEYS)
elo_ratings = np.array([1000, 1000, 1000, 1000, 1000, 1000])
elo_history = [np.array([1000]), np.array([1000]), np.array([1000]), np.array([1000]), np.array([1000]), np.array([1000])]

elo = MultiElo()

with open("elos/ranking_errored.txt") as f:
    errored = f.readlines()
errored = [int(x.strip()) for x in errored] 

########### TODO: Replace with file ###############
## the rankings may be out of order, read from dict keys
START = 0
END = 500
rankings = []
print(KEYS)
for i in range(START, END + 1):
    if i in errored:
        continue
    ### RANDOM
    # ranks = [1, 1, 2, 2, 3, 3]
    # random.shuffle(ranks)
    with open(f"elos/rankings/{i}.json") as f:
        data_point = json.loads(f.read())
    ranks = [
        data_point["No Principles"],
        data_point["All layers"],
        data_point["Layer 0"],
        data_point["Layer 1"],
        data_point["Layer 2"],
        data_point["Layer 3"],
    ]
    # if all(x == 1 for x in ranks): ## goof
    #     continue
    rankings.append(ranks)


###################################################

for ranks in rankings:

    # Update ratings based on user input
    new_ratings = elo.get_new_ratings(elo_ratings, result_order=ranks)
    
    # Update the rating history
    for idx, rating in enumerate(new_ratings):
        elo_history[idx] = np.append(elo_history[idx], rating)
    
    elo_ratings = new_ratings
    print("Current Elo Ratings:", elo_ratings)

for idx, recipe in enumerate(KEYS):
    plt.plot(elo_history[idx], label=recipe)

conditions = ["No Principles", "All layers", "Layer 0", "Layer 1", "Layer 2", "Layer 3"]
total_counts = {key: 0 for key in conditions}
win_counts = {key: 0 for key in conditions}
loss_counts = {key: 0 for key in conditions}
total_ranks = {key: 0 for key in conditions}

for ranking in rankings:
    min_rank = min(ranking)
    max_rank = max(ranking)
    
    for condition, rank in zip(conditions, ranking):
        total_counts[condition] += 1
        total_ranks[condition] += rank
        
        if rank == min_rank:
            win_counts[condition] += 1
        if rank == max_rank:
            loss_counts[condition] += 1

## get number of ties
count_all_ones = sum(all(rank == 1 for rank in x) for x in rankings)
print("All tied: ", count_all_ones)  # This will print the number of lists where all elements are 1

# chuck all-tied ones 
rankings = [x for x in rankings if not all(rank == 1 for rank in x)]

### Need to validate this
win_rates = {key: win_counts[key] / total_counts[key] for key in conditions}
loss_rates = {key: loss_counts[key] / total_counts[key] for key in conditions}
average_performance = {key: total_ranks[key] / total_counts[key] for key in conditions}

print("Win Rates:", win_rates)
print("Loss Rates:", loss_rates)
print("Average Rank:", average_performance)
### Need to validate this

plt.xlabel("Number of Iterations")
plt.ylabel("Elo Rating")
plt.title("Elo Rating Changes")
plt.legend()
plt.show()