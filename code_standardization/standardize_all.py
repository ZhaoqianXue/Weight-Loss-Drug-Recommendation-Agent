import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_standardization import standardization
import pandas as pd

if __name__ == "__main__":
    input_path = "data_extracted/extracted_reviews_all.csv"
    output_path = "data_standardized/standardized_reviews_all.csv"
    review_df = pd.read_csv(input_path)
    review_df['Date'] = pd.to_datetime(review_df['Date']).dt.strftime('%Y-%m-%d')
    ae_texts, ae_embeddings = standardization.prepare_ae_data(standardization.merged_df_embedded)
    standardized_df = standardization.standardize_side_effects(review_df, ae_texts, ae_embeddings)
    standardized_df = standardized_df.drop(columns=[col for col in ['structured_info', 'relations'] if col in standardized_df.columns])
    standardization.save_results(standardized_df, output_path)
    print(f"Standardization completed and saved to {output_path}") 