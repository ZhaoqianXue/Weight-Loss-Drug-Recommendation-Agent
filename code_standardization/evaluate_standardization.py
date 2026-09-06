#!/usr/bin/env python3
"""
Standardization Evaluation Script

This script evaluates the UMLS-based standardization results by comparing
side effects before and after standardization.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set
import os

# Use a portable font for English chart labels
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# File paths
ORIGINAL_FILE = 'data_extracted/extracted_reviews_top10.csv'
STANDARDIZED_FILE = 'data_standardized/standardized_reviews_umls.csv'
REPORT_FILE = 'data_standardized/standardization_report_umls.json'
OUTPUT_DIR = 'data_standardized'

def extract_side_effects_from_structured_info(structured_info_str: str, is_standardized: bool = False) -> List[Dict]:
    """
    Extract side effects from structured_info string.

    Args:
        structured_info_str: JSON string containing structured info
        is_standardized: Whether this is the standardized version

    Returns:
        List of side effect dictionaries
    """
    if not isinstance(structured_info_str, str):
        return []

    try:
        data = json.loads(structured_info_str)
        side_effects = data.get('side_effects', [])

        results = []
        for se in side_effects:
            se_dict = {
                'original_name': se.get('name', '').lower().strip(),
                'severity': se.get('severity'),
            }

            if is_standardized:
                se_dict['standard_name'] = se.get('standard_name')
                se_dict['standardized'] = se.get('standardized', False)

            results.append(se_dict)

        return results

    except (json.JSONDecodeError, TypeError):
        return []

def analyze_standardization_results() -> Dict:
    """
    Analyze the standardization results and generate comprehensive evaluation.

    Returns:
        Dictionary containing analysis results
    """
    # Load data
    print("📊 Loading data files...")
    original_df = pd.read_csv(ORIGINAL_FILE)
    standardized_df = pd.read_csv(STANDARDIZED_FILE)

    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    # Extract side effects from both versions
    print("🔍 Extracting side effects...")
    original_side_effects = []
    standardized_side_effects = []

    for idx, row in original_df.iterrows():
        original_ses = extract_side_effects_from_structured_info(row['structured_info'])
        for se in original_ses:
            se['review_id'] = idx
        original_side_effects.extend(original_ses)

    for idx, row in standardized_df.iterrows():
        standardized_ses = extract_side_effects_from_structured_info(
            row['structured_info_standardized'], is_standardized=True
        )
        for se in standardized_ses:
            se['review_id'] = idx
        standardized_side_effects.extend(standardized_ses)

    # Create mapping for comparison
    mapping_dict = report_data.get('mappings', {})

    # Analysis
    analysis = {
        'total_side_effects': len(original_side_effects),
        'unique_original_terms': len(set(se['original_name'] for se in original_side_effects)),
        'successful_mappings': len(mapping_dict),
        'mapping_rate': f"{len(mapping_dict) / len(set(se['original_name'] for se in original_side_effects)) * 100:.2f}%",
        'detailed_mappings': [],
        'unmapped_terms': report_data.get('unmapped_terms', []),
        'term_frequency': Counter(se['original_name'] for se in original_side_effects),
        'standardized_term_frequency': defaultdict(int),
        'severity_distribution': defaultdict(int),
        'review_level_stats': defaultdict(int)
    }

    # Detailed mapping analysis
    for std_se in standardized_side_effects:
        original_name = std_se['original_name']
        standard_name = std_se.get('standard_name')
        is_standardized = std_se.get('standardized', False)

        if is_standardized and standard_name:
            analysis['standardized_term_frequency'][standard_name] += 1
            analysis['detailed_mappings'].append({
                'original': original_name,
                'standardized': standard_name,
                'review_id': std_se['review_id'],
                'severity': std_se.get('severity')
            })

        # Severity analysis
        severity = std_se.get('severity') or 'Not specified'
        analysis['severity_distribution'][severity] += 1

    # Review-level statistics
    for idx in range(len(standardized_df)):
        review_ses = [se for se in standardized_side_effects if se['review_id'] == idx]
        standardized_count = sum(1 for se in review_ses if se.get('standardized', False))
        total_count = len(review_ses)

        if total_count > 0:
            analysis['review_level_stats'][f'Review_{idx}'] = {
                'total_side_effects': total_count,
                'standardized_side_effects': standardized_count,
                'standardization_rate': f"{standardized_count / total_count * 100:.1f}%"
            }

    return analysis

def create_detailed_comparison_table(analysis: Dict) -> pd.DataFrame:
    """
    Create a detailed comparison table of original vs standardized terms.

    Args:
        analysis: Analysis results dictionary

    Returns:
        DataFrame with comparison results
    """
    comparison_data = []

    # Add successfully mapped terms
    for mapping in analysis['detailed_mappings']:
        comparison_data.append({
            'Original Term': mapping['original'],
            'Standardized Term': mapping['standardized'],
            'Frequency': analysis['term_frequency'][mapping['original']],
            'Status': 'Mapped',
            'Severity': mapping.get('severity', 'Not specified')
        })

    # Add unmapped terms
    for term in analysis['unmapped_terms']:
        comparison_data.append({
            'Original Term': term,
            'Standardized Term': 'N/A',
            'Frequency': analysis['term_frequency'][term],
            'Status': 'Unmapped',
            'Severity': 'Various'
        })

    df = pd.DataFrame(comparison_data)
    return df.drop_duplicates(subset=['Original Term']).sort_values('Frequency', ascending=False)

def generate_evaluation_report(analysis: Dict):
    """
    Generate a comprehensive evaluation report.

    Args:
        analysis: Analysis results dictionary
    """
    print("\n" + "="*80)
    print("📋 UMLS Standardization Evaluation Report")
    print("="*80)

    print(f"\n📊 Overall statistics:")
    print(f"  • Total side-effect records: {analysis['total_side_effects']}")
    print(f"  • Unique original terms: {analysis['unique_original_terms']}")
    print(f"  • Successfully mapped terms: {analysis['successful_mappings']}")
    print(f"  • Mapping success rate: {analysis['mapping_rate']}")

    print(f"\n✅ Successfully mapped terms ({len(analysis['detailed_mappings'])} mappings):")
    successful_mappings = {}
    for mapping in analysis['detailed_mappings']:
        key = (mapping['original'], mapping['standardized'])
        if key not in successful_mappings:
            successful_mappings[key] = analysis['term_frequency'][mapping['original']]

    sorted_mappings = sorted(successful_mappings.items(), key=lambda x: x[1], reverse=True)
    for i, ((original, standardized), freq) in enumerate(sorted_mappings[:10], 1):
        print(f"  {i:2d}. '{original}' → '{standardized}' (frequency: {freq} occurrences)")

    if len(sorted_mappings) > 10:
        print(f"      ... remaining: {len(sorted_mappings) - 10} mappings")

    print(f"\n❌ Unmapped terms ({len(analysis['unmapped_terms'])} terms):")
    unmapped_with_freq = [(term, analysis['term_frequency'][term]) for term in analysis['unmapped_terms']]
    unmapped_with_freq.sort(key=lambda x: x[1], reverse=True)

    for i, (term, freq) in enumerate(unmapped_with_freq, 1):
        print(f"  {i:2d}. '{term}' (frequency: {freq} occurrences)")

    print(f"\n📈 Standardized term frequency distribution:")
    top_standardized = sorted(analysis['standardized_term_frequency'].items(),
                            key=lambda x: x[1], reverse=True)[:10]
    for i, (term, freq) in enumerate(top_standardized, 1):
        print(f"  {i:2d}. '{term}': {freq} occurrences")

    print(f"\n🔢 Severity distribution:")
    for severity, count in analysis['severity_distribution'].items():
        print(f"  • {severity}: {count} occurrences")

    print(f"\n📄 Standardization by review:")
    for review_id, stats in list(analysis['review_level_stats'].items())[:5]:
        print(f"  • {review_id}: {stats['standardized_side_effects']}/{stats['total_side_effects']} "
              f"({stats['standardization_rate']})")

    if len(analysis['review_level_stats']) > 5:
        print(f"  ... remaining: {len(analysis['review_level_stats']) - 5} reviews")

def create_visualization(analysis: Dict):
    """
    Create visualizations for the standardization results.

    Args:
        analysis: Analysis results dictionary
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UMLS Standardization Analysis', fontsize=16, fontweight='bold')

    # 1. Mapping success rate pie chart
    mapped_count = analysis['successful_mappings']
    unmapped_count = len(analysis['unmapped_terms'])

    axes[0, 0].pie([mapped_count, unmapped_count],
                   labels=['Successfully mapped', 'Unmapped'],
                   autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Term mapping success rate')

    # 2. Top original terms frequency
    top_original = sorted(analysis['term_frequency'].items(), key=lambda x: x[1], reverse=True)[:10]
    terms, freqs = zip(*top_original)

    axes[0, 1].barh(range(len(terms)), freqs, color='#3498db')
    axes[0, 1].set_yticks(range(len(terms)))
    axes[0, 1].set_yticklabels(terms, fontsize=8)
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_title('Original term frequency (Top 10)')

    # 3. Top standardized terms frequency
    if analysis['standardized_term_frequency']:
        top_standardized = sorted(analysis['standardized_term_frequency'].items(),
                                key=lambda x: x[1], reverse=True)[:10]
        std_terms, std_freqs = zip(*top_standardized)

        axes[1, 0].barh(range(len(std_terms)), std_freqs, color='#9b59b6')
        axes[1, 0].set_yticks(range(len(std_terms)))
        axes[1, 0].set_yticklabels(std_terms, fontsize=8)
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_title('Standardized term frequency (Top 10)')

    # 4. Severity distribution
    severities = list(analysis['severity_distribution'].keys())
    sev_counts = list(analysis['severity_distribution'].values())

    axes[1, 1].bar(range(len(severities)), sev_counts, color='#f39c12')
    axes[1, 1].set_xticks(range(len(severities)))
    axes[1, 1].set_xticklabels(severities, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Severity distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'standardization_analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Chart saved to: {OUTPUT_DIR}/standardization_analysis.png")

def main():
    """
    Main function to run the standardization evaluation.
    """
    print("🚀 Starting UMLS standardization evaluation...")

    # Perform analysis
    analysis = analyze_standardization_results()

    # Generate comparison table
    comparison_df = create_detailed_comparison_table(analysis)
    comparison_file = os.path.join(OUTPUT_DIR, 'standardization_comparison.csv')
    comparison_df.to_csv(comparison_file, index=False, encoding='utf-8')
    print(f"📄 Detailed comparison saved to: {comparison_file}")

    # Generate evaluation report
    generate_evaluation_report(analysis)

    # Create visualizations
    create_visualization(analysis)

    # Save detailed analysis
    analysis_file = os.path.join(OUTPUT_DIR, 'detailed_analysis.json')
    # Convert Counter objects to dict for JSON serialization
    analysis_serializable = {
        k: dict(v) if isinstance(v, (Counter, defaultdict)) else v
        for k, v in analysis.items()
    }

    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_serializable, f, indent=4, ensure_ascii=False)
    print(f"🔍 Detailed analysis saved to: {analysis_file}")

    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
