import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

# Print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss='warp')
# Train model
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
    # Number of users and movies in training data
    n_users, n_items = data['train'].shape

    # Generate recommendations for each user we input
    for user_id in user_ids:
        # Movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # Movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        # Rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        # Print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)

#ask the user to input the user id
user_id1 = int(input("Enter the user id 1: "))
user_id2 = int(input("Enter the user id 2: "))
user_id3 = int(input("Enter the user id 3: "))
#call the function
sample_recommendation(model, data, [user_id1, user_id2, user_id3])

#aak if the user would like to check another user id in a for loop
while True:
    aak = input("Would you like to check another user id? (y/n): ")
    if aak == "n":
        break
    user_id1 = int(input("Enter the user id 1: "))
    user_id2 = int(input("Enter the user id 2: "))
    user_id3 = int(input("Enter the user id 3: "))
    sample_recommendation(model, data, [user_id1, user_id2, user_id3])
    





