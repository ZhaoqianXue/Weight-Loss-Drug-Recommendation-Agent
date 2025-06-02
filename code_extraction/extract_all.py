import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_extraction import schema_extraction
import pandas as pd

if __name__ == "__main__":
    file_path = "data_webmd/webmd_all_reviews.csv"
    df = pd.read_csv(file_path)
    results_df = schema_extraction.process_reviews_structured(df, num_reviews=len(df))
    results_df.to_csv('data_extracted/extracted_reviews_all.csv', index=False)
    print("Extraction completed and saved to data_extracted/extracted_reviews_all.csv") 