import pandas as pd
import json
import os
import requests
from tqdm import tqdm

# Initialize tqdm for pandas integration
tqdm.pandas()

# --- Configuration ---
# TODO: Please enter your UMLS API key here.
# You can obtain a key by creating a UMLS Terminology Services (UTS) account:
# https://uts.nlm.nih.gov/uts/signup-login
UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "") # IMPORTANT: This needs to be filled in.
UMLS_API_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

# --- File Paths ---
EXTRACTED_REVIEWS_FILE = 'data_extracted/extracted_reviews_top10.csv'
ADVERSE_EVENTS_FILE = 'data_embedded/ae.csv'
OUTPUT_DIR = 'data_standardized'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'standardized_reviews.csv')
CUI_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'cui_mappings.json')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- UMLS API Interaction ---

def get_cui_from_api(term, api_key):
    """
    Searches for a term using the UMLS API and returns the CUI of the best match.
    """
    if not api_key:
        raise ValueError("UMLS_API_KEY is not set. Please provide your API key.")

    search_url = f"{UMLS_API_BASE_URL}/search/current"
    params = {
        "string": term,
        "apiKey": api_key,
        "searchType": "normalizedString",
        "pageNumber": 1,
        "pageSize": 1
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        results = data.get("result", {}).get("results", [])
        if results and results[0].get("ui") != "NONE":
            return results[0].get("ui")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to UMLS API for term '{term}': {e}")
    return None

# --- Caching Mechanism ---

def load_cui_cache():
    """Loads the CUI mapping cache from a file."""
    if os.path.exists(CUI_MAPPING_FILE):
        with open(CUI_MAPPING_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cui_cache(cache):
    """Saves the CUI mapping cache to a file."""
    with open(CUI_MAPPING_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

# --- Main Standardization Logic ---

def get_terms_to_cui_map(terms, api_key, cache):
    """
    Takes a list of terms, queries the UMLS API for their CUIs, and returns a mapping.
    Uses a cache to avoid redundant API calls.
    """
    term_to_cui = {}
    for term in tqdm(terms, desc="Mapping terms to CUIs"):
        if term in cache:
            term_to_cui[term] = cache[term]
        else:
            cui = get_cui_from_api(term, api_key)
            term_to_cui[term] = cui
            cache[term] = cui  # Update cache
    return term_to_cui

def main():
    """
    Main function to perform the side effect standardization.
    """
    print("Starting side effect standardization process...")

    # Check for API Key
    if not UMLS_API_KEY:
        print("\n" + "="*50)
        print("!!! ACTION REQUIRED !!!")
        print("Please enter your UMLS API key in the `UMLS_API_KEY` variable")
        print("in the script `code_standardization/standardization_new.py`.")
        print("You can get a key from: https://uts.nlm.nih.gov/uts/signup-login")
        print("="*50 + "\n")
        return

    # Load data
    print("Loading data...")
    try:
        reviews_df = pd.read_csv(EXTRACTED_REVIEWS_FILE)
        standard_aes_df = pd.read_csv(ADVERSE_EVENTS_FILE)
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}. Please ensure the files exist.")
        return

    # Extract terms
    print("Extracting terms from data...")
    standard_ae_list = standard_aes_df['AE'].dropna().unique().tolist()

    non_standard_side_effects = set()
    for info in tqdm(reviews_df['structured_info'].dropna(), desc="Extracting side effects from reviews"):
        try:
            data = json.loads(info)
            for se in data.get('side_effects', []):
                if se.get('name'):
                    non_standard_side_effects.add(se['name'].lower())
        except (json.JSONDecodeError, TypeError):
            # Ignore malformed JSON
            continue

    non_standard_se_list = list(non_standard_side_effects)

    print(f"Found {len(standard_ae_list)} unique standard adverse events.")
    print(f"Found {len(non_standard_se_list)} unique non-standard side effects.")

    # Load cache and map terms to CUI
    print("Mapping terms to UMLS Concept Unique Identifiers (CUIs)...")
    cui_cache = load_cui_cache()

    # Map standard AEs
    standard_ae_to_cui = get_terms_to_cui_map(standard_ae_list, UMLS_API_KEY, cui_cache)

    # Map non-standard SEs
    non_standard_se_to_cui = get_terms_to_cui_map(non_standard_se_list, UMLS_API_KEY, cui_cache)

    # Save the updated cache
    save_cui_cache(cui_cache)
    print(f"CUI mappings cached to {CUI_MAPPING_FILE}")

    # Create CUI to Standard AE mapping (for reverse lookup)
    cui_to_standard_ae = {cui: name for name, cui in standard_ae_to_cui.items() if cui}

    # Create the final mapping from non-standard to standard term
    se_standardization_map = {
        non_standard_name: cui_to_standard_ae.get(non_standard_cui)
        for non_standard_name, non_standard_cui in non_standard_se_to_cui.items()
        if non_standard_cui in cui_to_standard_ae
    }

    print("Applying standardization mapping to the reviews data...")
    # Function to apply the mapping to the 'structured_info' column
    def standardize_side_effects(info_str):
        if not isinstance(info_str, str):
            return info_str
        try:
            data = json.loads(info_str)
            if 'side_effects' in data:
                for se in data['side_effects']:
                    se_name = se.get('name', '').lower()
                    standard_name = se_standardization_map.get(se_name)
                    se['standard_name'] = standard_name
            return json.dumps(data)
        except (json.JSONDecodeError, TypeError):
            return info_str

    reviews_df['structured_info_standardized'] = reviews_df['structured_info'].progress_apply(standardize_side_effects)

    # Save the result
    reviews_df.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "="*50)
    print("Standardization complete!")
    print(f"The standardized data has been saved to: {OUTPUT_FILE}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
