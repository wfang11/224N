import numpy as np
import matplotlib.pyplot as plt
from multielo import MultiElo
recipes = ['Model 1', 'Model 2', 'Model 3']
elo_ratings = np.array([1500, 1500, 1500, 1500, 1500, 1500])
elo_history = [np.array([1500]), np.array([1500]), np.array([1500]), np.array([1500]), np.array([1500]), np.array([1500])]

elo = MultiElo()

### TODO! Read in ranks from a file
rankings = [[]]
for ranks in rankings:

    # Update ratings based on user input
    new_ratings = elo.get_new_ratings(elo_ratings, ranks)
    
    # Update the rating history
    for idx, rating in enumerate(new_ratings):
        elo_history[idx] = np.append(elo_history[idx], rating)
    
    elo_ratings = new_ratings
    print("Current Elo Ratings:", elo_ratings)

    
for idx, recipe in enumerate(recipes):
    plt.plot(elo_history[idx], label=recipe)

plt.xlabel("Number of Iterations")
plt.ylabel("Elo Rating")
plt.title("Elo Rating Changes")
plt.legend()
plt.show()
