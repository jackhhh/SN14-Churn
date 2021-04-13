import json

with open('../yelp_dataset/yelp_academic_dataset_user_active.json', 'w', encoding='utf-8') as f_out, \
    open('../yelp_dataset/yelp_academic_dataset_user_active_list.txt', 'w', encoding='utf-8') as f_active_list, \
    open('../yelp_dataset/yelp_academic_dataset_user.json', 'r', encoding='utf-8') as f_read:
    for line in f_read:
        obj = json.loads(line)
        if obj['review_count'] >= 50:
            f_out.write(line)
            f_active_list.write(obj['user_id'] + '\n')

