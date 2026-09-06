#!/usr/bin/env python3
"""
Side Effects Comparison Script

This script creates a detailed comparison showing the transformation of
side effects from original terms to standardized terms.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import json
from collections import defaultdict
import os

def extract_side_effects_detailed():
    """
    Extract detailed side effects comparison from the standardized data.
    """
    # Load original and standardized data
    original_df = pd.read_csv('data_extracted/extracted_reviews_top10.csv')
    standardized_df = pd.read_csv('data_standardized/standardized_reviews_umls.csv')

    comparison_data = []

    for idx, (orig_row, std_row) in enumerate(zip(original_df.iterrows(), standardized_df.iterrows())):
        orig_row = orig_row[1]  # Get the actual row data
        std_row = std_row[1]

        try:
            # Parse original structured info
            orig_data = json.loads(orig_row['structured_info'])
            std_data = json.loads(std_row['structured_info_standardized'])

            orig_side_effects = orig_data.get('side_effects', [])
            std_side_effects = std_data.get('side_effects', [])

            # Match and compare side effects
            for orig_se, std_se in zip(orig_side_effects, std_side_effects):
                comparison_data.append({
                    'Review_ID': idx,
                    'User': orig_row['User'],
                    'Original_Term': orig_se.get('name', ''),
                    'Severity': orig_se.get('severity', 'Not specified'),
                    'Standardized_Term': std_se.get('standard_name', 'N/A'),
                    'Is_Standardized': std_se.get('standardized', False),
                    'Standardization_Status': 'Successfully Mapped' if std_se.get('standardized', False) else 'Not Mapped'
                })

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error processing review {idx}: {e}")
            continue

    return pd.DataFrame(comparison_data)

def main():
    """
    Main function to generate the side effects comparison.
    """
    print("🔄 Generating the detailed side-effect comparison...")

    # Extract detailed comparison
    comparison_df = extract_side_effects_detailed()

    # Save detailed comparison
    detailed_file = 'data_standardized/detailed_side_effects_comparison.csv'
    comparison_df.to_csv(detailed_file, index=False, encoding='utf-8')

    print(f"📄 Detailed comparison saved to: {detailed_file}")

    # Print summary by review
    print("\n" + "="*80)
    print("📋 Side-effect standardization by review")
    print("="*80)

    for review_id in comparison_df['Review_ID'].unique():
        review_data = comparison_df[comparison_df['Review_ID'] == review_id]
        user = review_data.iloc[0]['User']
        total_side_effects = len(review_data)
        standardized_count = len(review_data[review_data['Is_Standardized'] == True])

        print(f"\n👤 Review {review_id} (User: {user}):")
        print(f"  📊 Total side effects: {total_side_effects}")
        print(f"  ✅ Successfully standardized: {standardized_count}")
        print(f"  📈 Standardization rate: {standardized_count/total_side_effects*100:.1f}%")
        print(f"  📝 Side-effect details:")

        for _, row in review_data.iterrows():
            status_icon = "✅" if row['Is_Standardized'] else "❌"
            if row['Is_Standardized']:
                print(f"    {status_icon} '{row['Original_Term']}' → '{row['Standardized_Term']}'")
            else:
                print(f"    {status_icon} '{row['Original_Term']}' (Unmapped)")

    # Print overall mapping summary
    print("\n" + "="*80)
    print("📈 Overall mapping results")
    print("="*80)

    total_side_effects = len(comparison_df)
    successfully_mapped = len(comparison_df[comparison_df['Is_Standardized'] == True])

    print(f"\n📊 Overall statistics:")
    print(f"  • Total side-effect records: {total_side_effects}")
    print(f"  • Successful mappings: {successfully_mapped}")
    print(f"  • Mapping success rate: {successfully_mapped/total_side_effects*100:.2f}%")

    # Show mapping examples
    print(f"\n✅ Successful mapping examples:")
    mapped_df = comparison_df[comparison_df['Is_Standardized'] == True]
    unique_mappings = mapped_df[['Original_Term', 'Standardized_Term']].drop_duplicates()

    for i, (_, row) in enumerate(unique_mappings.iterrows(), 1):
        print(f"  {i:2d}. '{row['Original_Term']}' → '{row['Standardized_Term']}'")

    # Show unmapped terms
    print(f"\n❌ Unmapped terms:")
    unmapped_df = comparison_df[comparison_df['Is_Standardized'] == False]
    unique_unmapped = unmapped_df['Original_Term'].unique()

    for i, term in enumerate(unique_unmapped, 1):
        print(f"  {i:2d}. '{term}'")

    # Severity analysis
    print(f"\n🔢 Severity analysis:")
    severity_counts = comparison_df['Severity'].value_counts()
    for severity, count in severity_counts.items():
        print(f"  • {severity}: {count}  occurrences")

    print("\n" + "="*80)
    print("✅ Side-effect comparison complete!")
    print("="*80)

if __name__ == '__main__':
    main()
