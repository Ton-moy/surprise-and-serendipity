# This script models surprise for the Basic model
import numpy as np
import os
import json
import glob

# Get the directory paths for users, topic distributions, and user preferences for the Basic model
user_dir = '../Users'
topic_distr_dir = '../topic_distrs'
user_pref_dir = '../basic_model_user_preferences'

user_json_paths = glob.glob(os.path.join(user_dir, '*.json'))

for user_json_path in user_json_paths:

    # Getting User review data
    with open(user_json_path, 'r') as f:
        data = json.load(f)
        user = data['user_id']
        reviews = data['reviews']

    # Getting all necessary topic distributions for user reviews
    topic_distr_dict = {}
    for rev in reviews:
        book_id = rev['book_id']
        topic_distr_paths = glob.glob(os.path.join(topic_distr_dir, f'{book_id}.txt'))

        if len(topic_distr_paths) == 0:
            continue
        else:
            distr = np.loadtxt(topic_distr_paths[0])
            topic_distr_dict[book_id] = distr
            
    out_data = {
        'user_id': user,
        'reviews': [],
        'total_sur_books' : [],
    }
    
    all_topic_distr = []
    n_topics = 100
    
    Book_no = -1

    max_weight = np.zeros(n_topics)
    for rev in reviews:

        book_id = rev['book_id']

        topic_distr = topic_distr_dict[book_id]
        all_topic_distr.append(topic_distr) 
        orig_rating = float(rev['rating'])
        
        #Scaling rating from 1-5 => -2-2
        rating = orig_rating - 3
        Book_no = Book_no+1
        
        if max(topic_distr - max_weight) >= 0.12:            
            temp = {}
            temp['book_no'] = Book_no
            temp['book_id'] = book_id
            temp['topic_distr'] = list(topic_distr)
            temp['difference'] = max(topic_distr - max_weight)
            temp['max_weight'] = list(np.maximum(max_weight, topic_distr))
            temp['rating'] = rating
            temp['surprising'] = '1'
            out_data['reviews'].append(temp)
            
        else:
            temp = {}
            temp['book_no'] = Book_no
            temp['book_id'] = book_id
            temp['topic_distr'] = list(topic_distr)
            temp['difference'] = max(topic_distr - max_weight)
            temp['max_weight'] = list(np.maximum(max_weight, topic_distr))
            temp['rating'] = rating
            temp['surprising'] = '0'
            out_data['reviews'].append(temp)
                
        max_weight = np.maximum(max_weight, topic_distr) 
     
    # Write out the user's info in the 'basic_model_user_preferences' directory 
    with open(os.path.join(user_pref_dir, f'{user}.json'), 'w') as f:
        json.dump(out_data, f)

        