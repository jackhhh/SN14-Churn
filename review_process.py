import json

with open('../yelp_dataset/yelp_academic_dataset_review_restaurants.json', 'w', encoding='utf-8') as f_out, \
    open('../yelp_dataset/yelp_academic_dataset_business_restaurants_list.txt', 'r', encoding='utf-8') as f_restaurants_list, \
    open('../yelp_dataset/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f_read:
    restaurants_list = f_restaurants_list.read().split('\n')
    for line in f_read:
        obj = json.loads(line)
        if obj['business_id'] in restaurants_list:
            f_out.write(line)

