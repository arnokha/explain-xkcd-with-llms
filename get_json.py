import pandas as pd
import os
import json
import re

def extract_comic_index(url):
    # Extract the comic index from the URL using regex
    match = re.search(r'/(\d+)/?$', url.strip('/'))
    if match:
        return int(match.group(1))
    else:
        return None

def convert_csv_to_json(csv_file_path, json_file_path, index_filter=None):
    df = pd.read_csv(csv_file_path)
    
    # Extract comic_index from URL
    df['comic_index'] = df['URL'].apply(extract_comic_index)
    
    # Drop rows where comic_index could not be extracted
    df = df.dropna(subset=['comic_index'])
    df['comic_index'] = df['comic_index'].astype(int)
    
    if index_filter is not None:
        df = df[df['comic_index'].isin(index_filter)]
    
    df = df.sort_values('comic_index')
    df.to_json(json_file_path, orient='records', force_ascii=False)
    print(f'Converted {csv_file_path} to {json_file_path}')

# Paths to your CSV files
csv_json_pairs = [
    ('inputs/xkcd_comic.csv', 'outputs/json/xkcd_comic.json'),
    ('inputs/xkcd_explanations.csv', 'outputs/json/xkcd_explanations.json'),
    ('outputs/responses/openai/responses.csv', 'outputs/json/openai_responses.json'),
    ('outputs/responses/anthropic/responses.csv', 'outputs/json/anthropic_responses.json'),
    ('outputs/responses/google/responses.csv', 'outputs/json/google_responses.json')
]

# Create output directory if it doesn't exist
os.makedirs('outputs/json', exist_ok=True)

# Read responses to get the list of comic indices
responses_files = [
    'outputs/responses/openai/responses.csv',
    'outputs/responses/anthropic/responses.csv',
    'outputs/responses/google/responses.csv'
]

# Collect comic indices from the response DataFrames
comic_indices = set()
for response_file in responses_files:
    df_response = pd.read_csv(response_file)
    comic_indices.update(df_response['comic_index'].unique())

# Convert each CSV file to JSON
for csv_file, json_file in csv_json_pairs:
    if 'responses.csv' in csv_file:
        # For response files, ensure comic_index is int and sort
        df = pd.read_csv(csv_file)
        df['comic_index'] = df['comic_index'].astype(int)
        df = df.sort_values('comic_index')
        df.to_json(json_file, orient='records', force_ascii=False)
        print(f'Converted {csv_file} to {json_file}')
    else:
        # For other files, extract comic_index from URL, filter, and sort
        convert_csv_to_json(csv_file, json_file, index_filter=comic_indices)
