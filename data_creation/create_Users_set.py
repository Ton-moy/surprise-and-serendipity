import gzip
import json
import os
import time
import glob
from sortedcontainers import SortedList
import calendar
from dateutil import parser

# specify the path of raw dataset file
data_file = './goodreads_reviews_dedup.json.gz'

book_dir = './book'

user_dict = {}
user_counts = {}
get_timestamp = lambda rev: rev['timestamp']

# open the gzip-compressed data file
with gzip.open(data_file) as f:
    for line in f:
        d = json.loads(line)
        book = d['book_id']
        user = d['user_id']
        rating = d['rating']

        book_paths = glob.glob(os.path.join(book_dir, f'{book}.txt'))
        
        # skip reviews without a read date
        if d['read_at'] == '':
            continue
        
        # skip books that don't exist in the Books set for each user 
        add_freq = 1
        if len(book_paths) == 0:
            add_freq = 0
            continue

        dct = {
            'book_id': book,
            'user_id': user,
            'rating': rating,
            'timestring': d['read_at'],
            'timestamp': calendar.timegm(parser.parse(d['read_at']).timetuple())
        }       

        #add user and book info for each user in chronological order
        if user not in user_dict:
            user_dict[user] = SortedList([dct], key=get_timestamp)
            user_counts[user] = add_freq
        else:
            user_dict[user].add(dct)
            user_counts[user] += add_freq

# save each user's data in a separate JSON file in the Users directory          
for user in user_dict:
    if user_counts[user] > 100:
        with open(f'./Users/{user}.json', 'w') as f:
            data = {
                'user_id': user,
                'reviews': list(user_dict[user])
                }
            f.write(json.dumps(data))


