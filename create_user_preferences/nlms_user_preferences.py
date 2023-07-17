# This script creates user preferences for the Normalized Least Mean Square model
import numpy as np
import os
import json
import glob
from model.models import NLMS

# Get the directory paths for users, topic distributions, and user preferences for the NLMS model
user_dir = '../Users'
topic_distr_dir = '../topic_distrs'
user_pref_dir = '../nlms_user_preferences'

user_json_paths = glob.glob(os.path.join(user_dir, '*.json'))

for user_json_path in user_json_paths:

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

    #Initializing NLMS Model
    n_topics = 100
    step_size = 0.1
    eps = 0.001
    k = 3
    model = NLMS(n_topics, step_size, eps)

    p_means = model.get_params()
    out_data = {
        'user_id': user,
        'reviews': [],
        'count': len(reviews),
    }

    mean_list = []
    topic_distr_list = []
    book_id_list = []
    rating_list = []
    prediction_list = []
    
    k_step = 0 
    for rev in reviews:

        book_id = rev['book_id']

        topic_distr = topic_distr_dict[book_id]
        topic_distr_list.append(topic_distr)

        orig_rating = float(rev['rating'])

        #Scaling rating from 1-5 => -2-2
        rating = orig_rating - 3
        
        rating_list.append(rating)
        book_id_list.append(book_id)

        # Get predicted rating and updated means (user preferences)
        prediction = model.predict(topic_distr)
        means = model.update(topic_distr, rating)

        prediction_list.append(prediction)
        mean_list.append(means)
        k_step += 1
        
        if k_step == k:
            mean_diff = (abs(np.linalg.norm(means - p_means)))
            temp = {}
            temp['means'] = list(mean_list[k_step-k])
            temp['topic_distr'] = list(topic_distr_list[k_step-k])
            temp['rating'] = rating_list[k_step-k]
            temp['prediction'] = prediction_list[k_step-k]
            temp['book_id'] = book_id_list[k_step-k]
            temp['mean_diff'] = mean_diff # surprise value
            
            out_data['reviews'].append(temp)
        
        if k_step > k and k_step < len(reviews):
            mean_diff = (abs(np.linalg.norm(mean_list[k_step-1] - mean_list[k_step-k-1])))
            temp = {}
            temp['means'] = list(mean_list[k_step-k])
            temp['topic_distr'] = list(topic_distr_list[k_step-k])
            temp['rating'] = rating_list[k_step-k]
            temp['prediction'] = prediction_list[k_step-k]
            temp['book_id'] = book_id_list[k_step-k]
            temp['mean_diff'] = mean_diff # surprise value

            out_data['reviews'].append(temp)
            
        if k_step == len(reviews):
            for i in range(k):
                mean_diff = (abs(np.linalg.norm(mean_list[k_step-1] - mean_list[k_step-k+i-1])))
                temp = {}
                temp['means'] = list(mean_list[k_step-k+i])
                temp['topic_distr'] = list(topic_distr_list[k_step-k+i])
                temp['rating'] = rating_list[k_step-k+i]
                temp['prediction'] = prediction_list[k_step-k+i]
                temp['book_id'] = book_id_list[k_step-k+i]
                temp['mean_diff'] = mean_diff # surprise value

                out_data['reviews'].append(temp)

    
    # Write out the user's info in the 'nlms_user_preferences' directory
    with open(os.path.join(user_pref_dir, f'{user}.json'), 'w') as f:
        json.dump(out_data, f)






