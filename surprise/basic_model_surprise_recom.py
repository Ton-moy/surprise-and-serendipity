# This script calculates the performance of Basic model for serendipitous item recommendation
import json
import glob
import numpy as np
import os

user_one_path = ('./basic_model_user_preferences/baf13c4f873e8076c5a285370bc8b681.json')
#user_one_path = ('./basic_model_user_preferences/c0d62a1d66649a61ffa302f8baf16d05.json')
#user_one_path = ('./basic_model_user_preferences/ff03096dafc8911bfe595ee1b69fa43a.json')
#user_one_path = ('./basic_model_user_preferences/1108a957e7eb21cc10ed1dcd5c5d1b97.json')

preference_paths = glob.glob('./basic_model_user_preferences/*.json')

# Take surprising book ID for each reference user
sur_id_baf = ['13335037', '6639100', '18190302', '20546874', '40436', '2445580', '22387714', '20561643', '64229', '12333644']
sur_id_cod  = ['2813153', '11763128', '16174631', '893172', '21535713', '920607', '11235712', '12913325', '29357293', '31114367', '23363874', '13650477', '24359966']
sur_id_ff03 = ['18190302', '21793182', '25721392', '20546874', '4674549', '9680533', '21421609', '24359966', '15716836', '7681608', '23001588', '11763128', '23363874', '8239985']
sur_id_1108 = ['6853', '10378606', '10108463', '8612987', '9680533', '10779721', '10818853', '6314763', '1170202', '7840833', '6751356', '12870772', '764347', '13627570', '42899', '16115612', '17667688']

sur_id = [sur_id_baf, sur_id_cod, sur_id_ff03, sur_id_1108]
#user_one_path = preference_paths

topic_distr_diff_threshold = 3.5 #tau_d
max_topic_distr_diff_threshold = 0.12
start_index = 15
N = 10
TP = 0
FP = 0
FN = 0
TN = 0
info_for_similar_user = []
count = 0
all = 0
#cnt = 0


topic_distr_sum_one = 0
topic_distr_avg_one = 0
td_avg_one = []

with open(user_one_path, 'r') as f:
    user_one_data = json.load(f)
user_one_reviews = user_one_data['reviews']

for i_one, rev_one in enumerate(user_one_reviews):
    topic_distr_sum_one = topic_distr_sum_one + (rev_one['rating'] * np.array(rev_one['topic_distr']))
    topic_distr_avg_one = topic_distr_sum_one / (i_one + 1)
    td_avg_one.append(topic_distr_avg_one)

user_one_id = user_one_path.split('/')[-1][:-5]

if user_one_id == 'baf13c4f873e8076c5a285370bc8b681':
    book_IDs = sur_id[0]
if user_one_id == 'c0d62a1d66649a61ffa302f8baf16d05':
    book_IDs = sur_id[1]
if user_one_id == 'ff03096dafc8911bfe595ee1b69fa43a':
    book_IDs = sur_id[2]
if user_one_id == '1108a957e7eb21cc10ed1dcd5c5d1b97':
    book_IDs = sur_id[3]


#each book read by user u1
for i_one, rev_one in enumerate(user_one_reviews[start_index:]):
    
    topic_distr_diff = []
    info = []
    
    if i_one > 0:
        book_ID = rev_one['book_id']
    #print('User one book_id: ', book_ID)
        
        all = all + 1
     
        # go through all the users for each book read bu user u1
        for user_two_path in preference_paths:
        
            user_two_id = user_two_path.split('/')[-1][:-5]
        
            #check user u1 and user u2 are similar or not
            if user_one_id == user_two_id:
                continue
        
            with open(user_two_path, 'r') as f:
                user_two_data = json.load(f)
            user_two_reviews = user_two_data['reviews']
            
            #go through all the books read by user u2
            topic_distr_sum_two = 0
            topic_distr_avg_two = 0
            td_avg_two = []
            for i_two, rev_two in enumerate(user_two_reviews):
                topic_distr_sum_two = topic_distr_sum_two + (rev_two['rating'] * np.array(rev_two['topic_distr']))
                topic_distr_avg_two = topic_distr_sum_two / (i_two + 1)
                td_avg_two.append(topic_distr_avg_two)
                if i_two < start_index:
                    continue
             
                #check u1 and u2 read the same book or not
                if user_one_reviews[i_one]['book_id'] == user_two_reviews[i_two]['book_id']:
                    topic_distr_dist = np.linalg.norm(np.array(scattered_data2[i_one - 1]) - np.array(td_avg_two[i_two - 1]))
                    topic_distr_diff.append(topic_distr_dist)
                    temp = {}
                    temp = {
                            'user_one_id': user_one_id,
                            'user_one_rating': rev_one['rating'],
                            'user_two_id': user_two_id,
                            'user_two_rating': rev_two['rating'],
                            'topic_distr_one_max': rev_one['difference'],
                            'topic_distr_two_max': rev_two['difference'],
                            'book_id': rev_one['book_id'],
                            'book_id2': rev_two['book_id'],
                            'topic_distr_difference': topic_distr_dist,
                            }
                    info.append(temp)
                    continue      

    if len(topic_distr_diff) == 0:
        continue
    #take top N similar user based on mean difference
    top_similar_user = sorted(info, key = lambda x:x['topic_distr_difference'])[:N]
    
    #consider only those users from the top 10 whose topic distr difference less than topic distr difference threshold
    similar_user_within_threshold = []
    for user in top_similar_user:
    	if user['topic_distr_difference'] < topic_distr_diff_threshold:
    	    similar_user_within_threshold.append(user)
    	 
    if len(similar_user_within_threshold) == 0:
    	continue   
    similar_user_td_sorted = sorted(similar_user_within_threshold, key = lambda x:x['topic_distr_two_max'], reverse = True)
    
    if similar_user_td_sorted[0]['topic_distr_two_max'] >= max_topic_distr_diff_threshold:
     
        if (similar_user_td_sorted[0]['book_id']) in book_IDs:
            TP += 1
        else:
            fp += 1

    else:
        if similar_user_td_sorted[0]['book_id'] in book_IDs:
            FN += 1
        else:
            TN += 1	
               
print(f'****** Basic Model Surprising Item Recommendation *****max_topic_distr_diff_threshold_(0.12)*****top_10*****topic_distr_diff_threshold_(3.5)***{user_one_id}********************************')

if (TP + FP) == 0 or (TP + FN) == 0 or TP == 0:
    print('True Positive: ', TP, '   False Positive: ', FP, '   False Negative: ', FN, '   True Negative: ', TN)
else:
    p = TP/(TP + FP)
    r = TP/(TP + FN)
    f1 = (2 * p * r)/(p + r)
    print('Precision: ', p*100, '   Recall: ', r*100, '   F1_Score: ', f1*100)

