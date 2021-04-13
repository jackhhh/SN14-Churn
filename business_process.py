import json

with open('../yelp_dataset/yelp_academic_dataset_business_restaurants.json', 'w', encoding='utf-8') as f_out, \
    open('../yelp_dataset/yelp_academic_dataset_business_restaurants_list.txt', 'w', encoding='utf-8') as f_restaurants_list, \
    open('../yelp_dataset/yelp_academic_dataset_business.json', 'r', encoding='utf-8') as f_read:
    for line in f_read:
        obj = json.loads(line)
        if obj['categories']:
            if 'Restaurants' in obj['categories'] or 'Food' in obj['categories']:
                f_out.write(line)
                f_restaurants_list.write(obj['business_id'] + '\n')

