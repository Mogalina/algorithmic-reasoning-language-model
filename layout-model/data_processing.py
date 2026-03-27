import pandas as pd
import numpy as np 

def filter_csv_pandas(input_filepath):
    df = pd.read_csv(input_filepath)
    
    columns_to_keep = ['company', 'difficulty', 'preparation_days', 'title']
    filtered_df = df[columns_to_keep]
    

    return filtered_df

processed = filter_csv_pandas('../embeddings/dataset/dataset.csv')

processed['days_until'] = processed['preparation_days']
processed = processed.drop(columns=['preparation_days'])

count = processed.groupby(['company', 'days_until', 'difficulty'])['title'].count().reset_index()

data_on_columns = count.pivot(
    index=['company', 'days_until'], 
    columns='difficulty', 
    values='title'
).reset_index()

data_on_columns = data_on_columns.rename(columns={
    'EASY': 'easycount',
    'MEDIUM': 'mediumcount',
    'HARD': 'hardcount'
})

cols_to_fill = ['easycount', 'mediumcount', 'hardcount']
data_on_columns[cols_to_fill] = data_on_columns[cols_to_fill].fillna(0).astype(int)
data_on_columns.columns.name = None
# print(data_on_columns)

data_on_columns.to_csv("results.csv", index=False)

# google_final = data_on_columns[data_on_columns['company'] == 'Google']
# print(google_final)

# google_set  = count.loc['Google']

# print(google_set)

new_days = [7, 14, 25, 45, 70, 110, 190, 250]
all_companies = data_on_columns['company'].unique()
existing_days = data_on_columns['days_until'].unique()
all_days_combined = list(existing_days) + new_days

def predict_missing_days_svd(df, value_cal, k_components=3):
    pass 

