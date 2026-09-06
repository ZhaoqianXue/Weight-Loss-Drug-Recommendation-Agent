import pandas as pd
import json
import sys
import os

input_path = 'data_standardized/standardized_reviews_all.csv'

approval_count = 0
off_label_count = 0
approval_condition_set = set()
off_label_condition_set = set()

try:
    df = pd.read_csv(input_path)
    for index, row in df.iterrows():
        try:
            relations_str = row.get('standardized_relations')
            if isinstance(relations_str, str):
                relations = json.loads(relations_str)
                if isinstance(relations, list):
                    for rel in relations:
                        if isinstance(rel, dict) and rel.get('relation') == 'treats' and rel.get('end', {}).get('label') == 'Disease':
                            approval_status = rel.get('properties', {}).get('approval')
                            condition_name = rel.get('end', {}).get('properties', {}).get('name')
                            if approval_status == 'yes':
                                approval_count += 1
                                if condition_name:
                                    approval_condition_set.add(condition_name)
                            elif approval_status == 'no':
                                off_label_count += 1
                                if condition_name:
                                    off_label_condition_set.add(condition_name)
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    print(f'Approval: {approval_count}')
    print(f'Off Label: {off_label_count}')
    approval_condition = sorted(list(approval_condition_set))
    off_label_condition = sorted(list(off_label_condition_set))
    print(f'\napproval_condition = {approval_condition}')
    print(f'\noff_label_condition = {off_label_condition}')

except FileNotFoundError:
    print(f"Error: {input_path} not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
