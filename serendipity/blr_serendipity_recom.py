# This script calculates the performance of BLR model for serendipitous item recommendation
import numpy as np
import os
import json
import glob

"""
In this code, we considered reference users as user_one and all other users as user_two.
At a time, we evaluate one user. So just changing reference user's path, we evaluated all four reference users.
"""

user_one_path = ('./blr_user_preferences/baf13c4f873e8076c5a285370bc8b681.json')
#user_one_path = ('./blr_user_preferences/c0d62a1d66649a61ffa302f8baf16d05.json')
#user_one_path = ('./blr_user_preferences/ff03096dafc8911bfe595ee1b69fa43a.json')
#user_one_path = ('./blr_user_preferences/1108a957e7eb21cc10ed1dcd5c5d1b97.json')

preference_paths = glob.glob('./blr_user_preferences/*.json')

# Take serendipitous book ID for each reference user
ser_id_baf = ['13335037', '6639100', '18190302', '20546874', '22387714', '20561643', '64229']
ser_id_cod  = ['2813153', '16174631', '893172', '21535713', '920607', '11235712', '12913325', '29357293', '23363874', '13650477', '24359966']
ser_id_ff03 = ['18190302', '21793182', '20546874', '4674549', '9680533', '24359966', '15716836', '7681608', '23001588', '11763128', '23363874', '8239985']
ser_id_1108 = ['6853', '10378606', '10108463', '8612987', '10779721', '1170202', '7840833', '12870772', '764347', '42899', '16115612']

ser_id = [ser_id_baf, ser_id_cod, ser_id_ff03, ser_id_1108]

start_index = 15 # Skipping first 15 books for each users 
preference_dist_threshold = 4.0 #tau_d
kl_divergence_threshold = 0.08 #tau_s
N = 5
TP = 0
FP = 0
FN = 0
TN = 0
info_for_similar_user = []

with open(user_one_path, 'r') as f:
    user_one_data = json.load(f)
user_one_reviews = user_one_data['reviews']

user_one_id = user_one_path.split('/')[-1][:-5]

if user_one_id == 'baf13c4f873e8076c5a285370bc8b681':
    book_IDs = ser_id[0]
if user_one_id == 'c0d62a1d66649a61ffa302f8baf16d05':
    book_IDs = ser_id[1]
if user_one_id == 'ff03096dafc8911bfe595ee1b69fa43a':
    book_IDs = ser_id[2]
if user_one_id == '1108a957e7eb21cc10ed1dcd5c5d1b97':
    book_IDs = ser_id[3]

# Each book read by user u1
for i_one, rev_one in enumerate(user_one_reviews[start_index:]):
    
    mean_diff = []
    info = []
    book_ID = rev_one['book_id']
    
    # Go through all the users for each book read by user u1
    for user_two_path in preference_paths:
        user_two_id = user_two_path.split('/')[-1][:-5]
        
        # Check user u1 and user u2 are similar or not
        if user_one_id == user_two_id:
            continue
        
        with open(user_two_path, 'r') as f:
            user_two_data = json.load(f)
        user_two_reviews = user_two_data['reviews']
            
        # Go through all the books read by user u2
        for i_two, rev_two in enumerate(user_two_reviews[start_index:]):
                
            # Checking whether u1 and u2 read the same book or not
            if rev_one['book_id'] == rev_two['book_id']:
                
                # Getting pervious book's mean for user u1 and u2
                means_one = np.array(user_one_reviews[start_index + i_one - 1]['means'])
                means_two = np.array(user_two_reviews[start_index + i_two - 1]['means'])   
                    
                means_dist = np.linalg.norm(means_one - means_two)
                mean_diff.append(means_dist)
                
                # Collecting information of the matched book for the both users, u1 and u2
                temp = {}
                kl_div_one = rev_one['kl_divergence']
                kl_div_two = rev_two['kl_divergence']
                    
                temp = {
                        'user_one_id': user_one_id,
                        'user_one_rating': rev_one['rating'],
                        'user_two_id': user_two_id,
                        'user_two_rating': rev_two['rating'],
                        'kl_div_one': kl_div_one,
                        'kl_div_two': kl_div_two,
                        'book_id_user1': rev_one['book_id'],
                        'book_id_user2': rev_two['book_id'],
                        'mean_difference': means_dist,
                        }
                info.append(temp)
                continue

    if len(mean_diff) == 0:
        continue 
        
    # Take top 5 similar user based on mean difference
    top_similar_user = sorted(info, key = lambda x:x['mean_difference'])[:N]
    
    # Consider only those users from the top 5 whose mean difference less than mean difference threshold
    similar_user_within_threshold = []
    for user in top_similar_user:
    	if user['mean_difference'] < preference_dist_threshold:
    	    similar_user_within_threshold.append(user)
    	 
    if len(similar_user_within_threshold) == 0:
    	continue   
    similar_user_kl_sorted = sorted(similar_user_within_threshold, key = lambda x:x['kl_div_two'], reverse = True)
    
    if similar_user_kl_sorted[0]['kl_div_two'] > kl_divergence_threshold and similar_user_kl_sorted[0]['user_two_rating'] > 0: 
        if (similar_user_kl_sorted[0]['book_id_user1']) in book_IDs:
            TP += 1
        else:
            FP += 1
            
    else:
        if similar_user_kl_sorted[0]['book_id_user1'] in book_IDs:
            FN += 1
        else:
            TN += 1

print(f'****** BLR Serendipitous Item Recommendation *****KL_({kl_divergence_threshold})***********mean_diff_({preference_dist_threshold})*****Top_{N}**{user_one_id}******')

if (TP + FP) == 0 or (TP + FN) == 0 or TP == 0:
    print('True Positive: ', TP, '   False Positive: ', FP, '   False Negative: ', FN, '   True Negative: ', TN)
else:
    p = TP/(TP + FP)
    r = TP/(TP + FN)
    f1 = (2 * p * r)/(p + r)
    print('Precision: ', p*100, '   Recall: ', r*100, '   F1_Score: ', f1*100)
