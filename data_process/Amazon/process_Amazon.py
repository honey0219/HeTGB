import numpy as np
import re

titles_list = []

with open('amazon_ratings_texts.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        title = line.strip()
        if title:
            titles_list.append(title)

data = np.load('amazon_ratings.npz')
node_labels = data['node_labels']
edges = data['edges']
train_masks = data['train_masks']
val_masks = data['val_masks']
test_masks = data['test_masks']

train_masks = train_masks[0]
val_masks = val_masks[0]
test_masks = test_masks[0]


data_dict = {}
with open('amazon-meta.txt', 'r') as file:
    lines = file.readlines()

    current_asin = None
    product_info = {}
    iter_lines = iter(lines)
    reading_reviews = False
    reading_category = False

    for line in iter_lines:
        line = line.strip()

        if not line:
            reading_reviews = False
            continue

        if line.startswith('ASIN:'):
            if current_asin is not None and 'title' in product_info:
                title = product_info['title']
                if title in data_dict:
                    title = f"{title} ({current_asin})"
                data_dict[title] = product_info

            current_asin = line.split(':')[1].strip()
            product_info = {}
            reading_reviews = False
            product_info['ASIN'] = current_asin

        elif line.startswith('title:'):
            product_info['title'] = line.split(':', 1)[1].strip()

        elif line.startswith('group:'):
            product_info['group'] = line.split(':', 1)[1].strip()

        elif line.startswith('salesrank:'):
            product_info['salesrank'] = int(line.split(':', 1)[1].strip())

        elif line.startswith('similar:'):
            product_info['similar'] = line.split(':', 1)[1].strip().split()

        elif line.startswith('categories:'):
            product_info['categories'] = []
            reading_category = True

        elif line.startswith('reviews:'):
            reading_category = False
            reading_reviews = True
            product_info['reviews'] = []
            review_metadata = line.split(':', 1)[1].strip()
            product_info['reviews_meta'] = review_metadata

        elif reading_reviews:
            product_info['reviews'].append(line)
        elif reading_category:
            product_info['categories'].append(line)
    if current_asin is not None and 'title' in product_info:
        title = product_info['title']
        if title in data_dict:
            title = f"{title} ({current_asin})"
        data_dict[title] = product_info


node_texts = []
for title in titles_list:
    if title in data_dict:
        info = data_dict[title]

        text1 = f"Name/title of the product: {info['title']}; "

        text1 += f"Amazon Salesrank: {info['salesrank']}; "

        text2 = f"Total number of co-purchased products is {info['similar'][0]};"

        reviews_meta_without_rating = re.sub(r'avg rating: \d+(\.\d+)?', '', info['reviews_meta']).strip()
        avg_rating_match = re.search(r'avg rating: (\d+(\.\d+)?)', info['reviews_meta'])
        avg_rating_value = float(avg_rating_match.group(1)) if avg_rating_match else None
        text4 = f"Number of reviews: {reviews_meta_without_rating}; "

        reviews_to_process = info['reviews'][:5] if len(info['reviews']) > 5 else info['reviews']
        cleaned_reviews = [re.search(r'rating: \d+(\.\d+)?(.*)', review).group(0).strip().replace(' ', '') for review in reviews_to_process]
        cleaned_reviews_new = [re.sub(r'rating:(\d+)votes:(\d+)helpful:(\d+)', r'rating: \1 votes: \2 helpful: \3', review) for review in cleaned_reviews]
        text5 = f"The first five reviews: {' | '.join(cleaned_reviews_new)}"

        all_text = text1 + text2 + text4 + text5
        all_text = all_text.replace('\t', ' ')
        node_texts.append(all_text)
    else:
        print(f"Error in title: {title}")

label_texts = ["Rating 5.0", "Rating 4.5", "Rating 4.0", "Rating 3.5", "Rating lower than 3.5"]
edges = edges.T
label_texts = np.array(label_texts)
node_texts = np.array(node_texts)
np.savez(
    'dataset.npz',
    edges=edges,
    node_labels=node_labels,
    node_texts=node_texts,
    label_texts=label_texts,
    train_masks=train_masks[0],
    val_masks=val_masks[0],
    test_masks=test_masks[0]
)