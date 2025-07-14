#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def load_and_split_data(data_path='train-test-data.xlsx'):
    print("Loading data...")
    xl = pd.ExcelFile(data_path)
    sheet_names = xl.sheet_names
    print(f"Available sheets: {sheet_names}")
    
    if 'train' in sheet_names and 'test' in sheet_names:
        print("Found separate 'train' and 'test' sheets. Loading directly...")
        train_df = pd.read_excel(data_path, sheet_name='train')
        test_df = pd.read_excel(data_path, sheet_name='test')
        
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Total samples: {len(train_df) + len(test_df)}")
        print(f"Columns: {train_df.columns.tolist()}")
        
        return train_df, test_df

def analyze_dataset(df, split_name=""):
    print(f"\n=== {split_name} Dataset Analysis ===")
    print(f"Total samples: {len(df)}")
    disease_counts = df['output_disease'].str.split(', ').str.len()
    print(f"Disease count distribution:")
    print(disease_counts.value_counts().sort_index())
    all_diseases = []
    for diseases in df['output_disease']:
        all_diseases.extend([d.strip() for d in diseases.split(',')])
    
    unique_diseases = list(set(all_diseases))
    print(f"Total unique diseases: {len(unique_diseases)}")
    print(f"Most common diseases:")
    disease_freq = pd.Series(all_diseases).value_counts().head(10)
    print(disease_freq)
    
    return unique_diseases

def save_processed_data(train_df, test_df, output_dir='processed_data'):
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    train_json = []
    test_json = []
    
    for _, row in train_df.iterrows():
        train_json.append({
            'case_id': int(row['case_id']),
            'input_finding': row['input_finding'],
            'output_disease': row['output_disease']
        })
    
    for _, row in test_df.iterrows():
        test_json.append({
            'case_id': int(row['case_id']),
            'input_finding': row['input_finding'],
            'output_disease': row['output_disease']
        })
    
    with open(f'{output_dir}/train.json', 'w') as f:
        json.dump(train_json, f, indent=2)
    
    with open(f'{output_dir}/test.json', 'w') as f:
        json.dump(test_json, f, indent=2)
    
    print(f"Data saved to {output_dir}/")

def main():
    train_df, test_df = load_and_split_data()
    train_diseases = analyze_dataset(train_df, "Training")
    test_diseases = analyze_dataset(test_df, "Test")
    save_processed_data(train_df, test_df)
    
    all_diseases = list(set(train_diseases + test_diseases))
    with open('processed_data/disease_vocabulary.json', 'w') as f:
        json.dump(sorted(all_diseases), f, indent=2)
    
    print(f"\nData preprocessing complete!")
    print(f"Disease vocabulary saved ({len(all_diseases)} unique diseases)")

if __name__ == "__main__":
    main() 