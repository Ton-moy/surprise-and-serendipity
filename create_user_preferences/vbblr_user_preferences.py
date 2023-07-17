# This script creates user preferences for the Variance Bounded Bayesian Linear Regression model
import os
import json
import glob
import numpy as np
from model.models import VarianceBoundedBayesianModel
from bayesian_surprise import kl_divergence

# Get the directory paths for users, topic distributions, and user preferences for the vbBLR model
user_dir = '../Users'
topic_distr_dir = '../topic_distrs'
user_pref_dir = '../vbblr_user_preferences'

user_json_paths = glob.glob(os.path.join(user_dir, '*.json'))

for user_json_path in user_json_paths:

    # Get User review data
    with open(user_json_path, 'r') as f:
        data = json.load(f)
        user = data['user_id']
        reviews = data['reviews']

    # Get all necessary topic distributions for user reviews
    topic_distr_dict = {}
    for rev in reviews:
        book_id = rev['book_id']
        topic_distr_paths = glob.glob(os.path.join(topic_distr_dir, f'{book_id}.txt'))

        if len(topic_distr_paths) == 0:
            continue
        else:
            distr = np.loadtxt(topic_distr_paths[0])
            topic_distr_dict[book_id] = distr

    #Initializing BLR Model
    n_topics = 100
    default_variance = 1.0
    noise_precision = 1.0
    tau = 0.4
    model = VarianceBoundedBayesianModel(n_topics, default_variance, noise_precision, tau)

    # Get initial means (user preferences) and covariances
    p_means, p_cov = model.get_params()
    out_data = {
        'user_id': user,
        'reviews': [],
        'count': len(reviews),
    }


    for rev in reviews:
        book_id = rev['book_id']
        topic_distr = topic_distr_dict[book_id]
        orig_rating = float(rev['rating'])

        # Scaling rating from 1-5 => -2-2
        rating = orig_rating - 3

        # Get predicted rating, updated means (user preferences) and covariances
        prediction = model.predict(topic_distr)
        means, s, cov = model.update(topic_distr, rating)

        # Store user's info in a dictionary
        temp = {}
        temp['means'] = list(means)
        temp['topic_distr'] = list(topic_distr)
        temp['rating'] = rating
        temp['prediction'] = prediction
        temp['book_id'] = book_id

        # Get KL divergence value (Bayesian surprise)
        divergence = kl_divergence(cov, p_cov, means, p_means)

        p_cov = cov
        p_means = means  

        temp['kl_divergence'] = divergence

        out_data['reviews'].append(temp)

    # Write out the user's info in the 'vbblr_user_preferences' directory
    with open(os.path.join(user_pref_dir, f'{user}.json'), 'w') as f:
        json.dump(out_data, f)










