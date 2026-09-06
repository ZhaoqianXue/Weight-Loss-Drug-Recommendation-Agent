import pandas as pd
import json
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
import re
from fuzzywuzzy import process, fuzz
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pickle
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- NLTK setup ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Downloading NLTK 'punkt' model...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK 'wordnet' model...")
    nltk.download('wordnet', quiet=True)
# Ensure punkt_tab is downloaded if needed by word_tokenize
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK 'punkt_tab' model...")
    nltk.download('punkt_tab', quiet=True)

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize tqdm for pandas integration
tqdm.pandas()

# --- Configuration ---
# You can obtain a key by creating a UMLS Terminology Services (UTS) account:
# https://uts.nlm.nih.gov/uts/signup-login
UMLS_API_KEY = os.environ.get("UMLS_API_KEY", "")  # IMPORTANT: This needs to be filled in.
UMLS_API_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIMILARITY_THRESHOLD = 0  # Cosine similarity threshold for embedding-based matching

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Fuzzy Matching Thresholds ---
# New threshold to validate that a CUI's atoms are semantically related to the source term.
UMLS_VALIDATION_THRESHOLD = 0

# --- File Paths (Version 6) ---
EXTRACTED_REVIEWS_FILE = 'data_extracted/extracted_reviews_all.csv'
ADVERSE_EVENTS_FILE = 'data_embedded/ae.csv'
OUTPUT_DIR = 'data_standardized'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'standardized_reviews_umls_v6.csv')
CUI_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'cui_mappings_umls_v6.json')
CUI_ATOM_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'cui_atom_mappings_v6.json')
STANDARDIZATION_REPORT_FILE = os.path.join(OUTPUT_DIR, 'standardization_report_umls_v6.json')
EMBEDDINGS_CACHE_FILE = os.path.join(OUTPUT_DIR, 'embeddings_cache_v6.pkl')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Embedding-based Matching Functions ---

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Get embedding for a given text using OpenAI's embedding model."""
    try:
        text = text.replace("\n", " ").strip()
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for text '{text}': {e}")
        return []

def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100) -> Dict[str, List[float]]:
    """Get embeddings for a batch of texts efficiently."""
    embeddings_dict = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
        batch = texts[i:i + batch_size]
        try:
            # Clean the batch texts
            cleaned_batch = [text.replace("\n", " ").strip() for text in batch]
            response = openai_client.embeddings.create(input=cleaned_batch, model=model)

            for j, embedding_data in enumerate(response.data):
                original_text = batch[j]
                embeddings_dict[original_text] = embedding_data.embedding

        except Exception as e:
            logger.error(f"Error getting embeddings for batch starting at index {i}: {e}")
            # Fallback: get individual embeddings
            for text in batch:
                embedding = get_embedding(text, model)
                if embedding:
                    embeddings_dict[text] = embedding

        # Rate limiting
        time.sleep(0.1)

    return embeddings_dict

def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0

    # Convert to numpy arrays and reshape for sklearn
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)

    similarity = cosine_similarity(emb1, emb2)[0][0]
    return float(similarity)

def load_embeddings_cache() -> Dict[str, List[float]]:
    """Load embeddings cache from file."""
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load embeddings cache: {e}")
    return {}

def save_embeddings_cache(embeddings_cache: Dict[str, List[float]]):
    """Save embeddings cache to file."""
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(embeddings_cache, f)
        logger.info(f"Embeddings cache saved to {EMBEDDINGS_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Could not save embeddings cache: {e}")

def find_best_embedding_match(
    term: str,
    standard_terms: List[str],
    embeddings_cache: Dict[str, List[float]]
) -> Tuple[Optional[str], float]:
    """Find the best matching standard term using embedding similarity."""
    if term not in embeddings_cache:
        return None, 0.0

    term_embedding = embeddings_cache[term]
    best_match = None
    best_similarity = 0.0

    for standard_term in standard_terms:
        if standard_term not in embeddings_cache:
            continue

        similarity = calculate_cosine_similarity(term_embedding, embeddings_cache[standard_term])

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = standard_term

    return best_match, best_similarity

# --- Term Pre-processing ---

def lemmatize_term(term: str) -> str:
    """Lemmatizes a given term by tokenizing and processing each word."""
    if not isinstance(term, str) or not term:
        return ""
    # Use word_tokenize for better handling of complex strings and punctuation
    tokens = word_tokenize(term.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)

def preprocess_and_decompose(term: str) -> List[str]:
    """
    Cleans and decomposes a complex medical term into simpler, searchable parts.
    Returns a list of potential terms to search, ordered from most likely to least.
    """
    term = term.lower().strip()

    # Handle parentheses: "BPPV (Benign...)" -> ["benign...", "bppv"]
    parentheses_match = re.match(r'(.+)\s\((.+)\)', term)
    if parentheses_match:
        full_name = parentheses_match.group(2).strip()
        acronym = parentheses_match.group(1).strip()
        return [full_name, acronym]

    # Decompose composite terms by splitting on conjunctions/prepositions
    decomposed_term = term.replace(' with ', '||').replace(' and ', '||').replace(',', '||').replace('/', '||')
    parts = [p.strip() for p in decomposed_term.split('||') if p.strip()]

    if len(parts) > 1:
        # If decomposed, search parts from longest to shortest, then the original term
        sorted_parts = sorted(parts, key=len, reverse=True)
        return sorted_parts + [term]

    return [term]

# --- UMLS API Interaction ---

class UMLSClient:
    """Client for interacting with UMLS Terminology Services API"""

    def __init__(self, api_key: str, base_url: str = UMLS_API_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url

        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.last_request_time = 0
        self.min_request_interval = 0.05  # Be respectful to the API

    def _rate_limit(self):
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def get_cuis_for_term(self, term: str, max_results: int = 10) -> List[str]:
        """Search for a term and return a list of candidate CUIs."""
        if not self.api_key:
            raise ValueError("UMLS_API_KEY is not set.")

        self._rate_limit()
        cleaned_term = term.strip().lower()
        if not cleaned_term:
            return []

        search_url = f"{self.base_url}/search/current"
        params = {"string": cleaned_term, "apiKey": self.api_key, "searchType": "words", "pageSize": max_results}

        try:
            response = self.session.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("result", {}).get("results", [])
            return [res['ui'] for res in results if res.get("ui") and res["ui"] != "NONE"]
        except requests.exceptions.RequestException as e:
            logger.error(f"UMLS API request failed for term '{term}': {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse UMLS API response for term '{term}': {e}")
        return []

    def get_atoms_for_cui(self, cui: str) -> List[str]:
        """Fetch all names (atoms) for a given CUI."""
        if not self.api_key:
            raise ValueError("UMLS_API_KEY is not set.")

        self._rate_limit()
        url = f"{self.base_url}/content/current/CUI/{cui}/atoms"
        params = {"apiKey": self.api_key, "sabs": "MSH,SNOMEDCT_US,MDR,RXNORM,CHV,NCI"} # Focus on relevant sources

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("result", [])
            return list(set([atom['name'].lower() for atom in results if atom.get('name')]))
        except requests.exceptions.RequestException as e:
            logger.error(f"UMLS API request failed for CUI '{cui}': {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse UMLS API response for CUI '{cui}': {e}")
        return []

# --- Caching Mechanism ---

def load_json_cache(file_path: str) -> Dict:
    """Loads a generic JSON cache file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Cache file error at {file_path}: {e}. Starting with an empty cache.")
    return {}

def save_json_cache(cache: Dict, file_path: str, description: str):
    """Saves a generic cache to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4, ensure_ascii=False)
        logger.info(f"{description} cached to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save cache file at {file_path}: {e}")

# --- Data Extraction and Processing ---

def extract_side_effects_from_reviews(reviews_df: pd.DataFrame) -> Set[str]:
    """Extract all unique side effects from the reviews."""
    side_effects = set()
    for info in tqdm(reviews_df['structured_info'].dropna(), desc="Extracting side effects"):
        try:
            data = json.loads(info)
            for se in data.get('side_effects', []):
                if se.get('name'):
                    side_effects.add(se['name'].strip()) # Keep original case for fuzzy matching
        except (json.JSONDecodeError, TypeError):
            continue
    return side_effects

def load_standard_adverse_events(ae_df: pd.DataFrame) -> List[str]:
    """Load standard adverse events from the FDA data."""
    return ae_df['AE'].dropna().str.strip().unique().tolist()

# --- Main Standardization Logic ---

def create_cui_to_standard_term_map(
    standard_terms: List[str], umls_client: UMLSClient, cache: Dict[str, List[str]]
) -> Dict[str, str]:
    """Create a map from a CUI to its preferred standard term."""
    cui_to_standard = {}
    for term in tqdm(standard_terms, desc="Mapping standard AEs to CUIs"):
        term_lower = term.lower()
        if term_lower not in cache:
            cache[term_lower] = umls_client.get_cuis_for_term(term_lower)

        for cui in cache.get(term_lower, []):
            # Prefer longer, more specific terms for a CUI mapping
            if cui not in cui_to_standard or len(term) > len(cui_to_standard[cui]):
                cui_to_standard[cui] = term
    return cui_to_standard

def create_standardization_map(
    non_standard_terms: List[str],
    standard_terms: List[str],
    standard_terms_lemmatized: List[str],
    lemmatized_to_original_map: Dict[str, str],
    cui_to_standard_map: Dict[str, str],
    umls_client: UMLSClient,
    cui_cache: Dict[str, List[str]],
    atom_cache: Dict[str, List[str]],
    embeddings_cache: Dict[str, List[float]] = None
) -> Dict[str, Dict[str, str]]:
    """
    Create the final mapping from non-standard to standard terms using a three-level strategy.
    """
    final_map = {}
    standard_terms_lemmatized_set = set(standard_terms_lemmatized)
    scorer = fuzz.WRatio

    # Track which terms remain unmapped after first two strategies
    unmapped_terms = []

    for term in tqdm(non_standard_terms, desc="Standardizing terms (Strategies 1-2)"):
        # Lemmatize the non-standard term for improved direct and fuzzy matching.
        term_lemmatized = lemmatize_term(term)

        # Strategy 1: Direct match on lemmatized terms (handles plurals, etc.)
        if term_lemmatized in standard_terms_lemmatized_set:
            standard_name = lemmatized_to_original_map[term_lemmatized]
            final_map[term] = {"standard_name": standard_name, "method": "Direct Match (Lemmatized)"}
            continue

        # Strategy 2: Validated UMLS CUI-based mapping
        # This version iterates through decomposed parts and uses the first high-confidence CUI match found.
        umls_match_found = False
        decomposed_parts = preprocess_and_decompose(term)

        for part in decomposed_parts:
            # Skip very short parts that are likely to be noise
            if len(part) < 3:
                continue

            part_lower = part.lower()
            if part_lower not in cui_cache:
                cui_cache[part_lower] = umls_client.get_cuis_for_term(part_lower)

            candidate_cuis = cui_cache.get(part_lower, [])
            if not candidate_cuis:
                continue

            best_cui_for_part = None
            highest_score_for_part = 0

            # Find the best CUI for this specific part
            for cui in candidate_cuis:
                if cui not in atom_cache:
                    atom_cache[cui] = umls_client.get_atoms_for_cui(cui)

                cui_atoms = atom_cache.get(cui, [])
                if not cui_atoms:
                    continue

                # Validate how well the CUI's known names match our original term part
                _match, score = process.extractOne(part, cui_atoms, scorer=scorer)

                if score > highest_score_for_part:
                    highest_score_for_part = score
                    best_cui_for_part = cui

            # If we found a strong semantic match for this part, map it and break
            if best_cui_for_part and highest_score_for_part >= UMLS_VALIDATION_THRESHOLD:
                if best_cui_for_part in cui_to_standard_map:
                    final_map[term] = {
                        "standard_name": cui_to_standard_map[best_cui_for_part],
                        "method": "UMLS Match (Validated)"
                    }
                    umls_match_found = True
                    break  # Exit the loop over parts as we've found a confident match

        if umls_match_found:
            continue

        # If no match found in first two strategies, add to unmapped list for embedding strategy
        unmapped_terms.append(term)

    # Strategy 3: Embedding-based matching for remaining unmapped terms
    if unmapped_terms and embeddings_cache:
        logger.info(f"Applying embedding-based matching to {len(unmapped_terms)} remaining unmapped terms...")

        for term in tqdm(unmapped_terms, desc="Embedding-based matching"):
            best_match, similarity = find_best_embedding_match(term, standard_terms, embeddings_cache)

            if best_match and similarity >= EMBEDDING_SIMILARITY_THRESHOLD:
                final_map[term] = {
                    "standard_name": best_match,
                    "method": "Embedding Match (OpenAI)",
                    "similarity_score": round(similarity, 4)
                }

    return final_map

def apply_standardization(reviews_df: pd.DataFrame, final_map: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """Apply standardization mapping to the reviews data."""
    def standardize_side_effects(info_str):
        if not isinstance(info_str, str): return info_str
        try:
            data = json.loads(info_str)
            if 'side_effects' in data:
                for se in data['side_effects']:
                    se_name = se.get('name', '').strip()
                    mapping_result = final_map.get(se_name)
                    if mapping_result:
                        se['standard_name'] = mapping_result['standard_name']
                        se['standardization_method'] = mapping_result['method']
                        se['standardized'] = True
                        if 'similarity_score' in mapping_result:
                            se['similarity_score'] = mapping_result['similarity_score']
                    else:
                        se['standard_name'] = None
                        se['standardization_method'] = 'Not Mapped'
                        se['standardized'] = False
            return json.dumps(data, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return info_str

    logger.info("Applying standardization to all reviews...")
    reviews_df_copy = reviews_df.copy()
    reviews_df_copy['structured_info_standardized'] = reviews_df_copy['structured_info'].progress_apply(standardize_side_effects)
    return reviews_df_copy

def generate_standardization_report(
    non_standard_terms: Set[str],
    final_map: Dict[str, Dict[str, str]]
) -> Dict:
    """
    Generates a detailed report of the standardization process.
    """
    report = {
        "total_unique_terms": len(non_standard_terms),
        "mapped_terms": len(final_map),
        "unmapped_terms": 0,
        "mapping_methods": {
            "Direct Match (Lemmatized)": 0,
            "UMLS Match (Validated)": 0,
            "Embedding Match (OpenAI)": 0
        },
        "mappings": {},
        "unmapped_list": []
    }

    # Populate the report
    for term, mapping_info in final_map.items():
        method = mapping_info['method']
        if method in report["mapping_methods"]:
            report["mapping_methods"][method] += 1

        # Store the full mapping info, not just the name
        report["mappings"][term] = mapping_info

    unmapped_set = non_standard_terms - set(final_map.keys())
    report["unmapped_terms"] = len(unmapped_set)
    report["unmapped_list"] = sorted(list(unmapped_set))

    return report

def main():
    """Main function to perform the enhanced side effect standardization."""
    print("="*60)
    print("Enhanced Side Effect Standardization (V6) - Three Strategy Version")
    print("="*60)

    # Check for required libraries
    try:
        import openai
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        print(f"\n!!! MISSING DEPENDENCY: {e}")
        print("Please install required packages:")
        print("pip install openai scikit-learn")
        return

    if not UMLS_API_KEY:
        print("\n!!! ACTION REQUIRED: Please enter your UMLS API key in the script.\n")
        return

    if not OPENAI_API_KEY:
        print("\n!!! ACTION REQUIRED: Please enter your OpenAI API key in the script.\n")
        return

    try:
        umls_client = UMLSClient(UMLS_API_KEY)

        logger.info("Loading data files...")
        reviews_df = pd.read_csv(EXTRACTED_REVIEWS_FILE)
        ae_df = pd.read_csv(ADVERSE_EVENTS_FILE)

        logger.info("Extracting terms from data...")
        non_standard_terms = extract_side_effects_from_reviews(reviews_df)
        standard_terms = load_standard_adverse_events(ae_df)

        logger.info(f"Found {len(non_standard_terms)} unique non-standard side effects.")
        logger.info(f"Found {len(standard_terms)} unique standard adverse events.")

        logger.info("Lemmatizing standard terms for improved matching...")
        # Create a mapping from lemmatized term back to its original form
        # Handle potential collisions by preferring longer, more descriptive original terms
        lemmatized_to_original_map = {}
        for term in standard_terms:
            lem_term = lemmatize_term(term)
            if lem_term not in lemmatized_to_original_map or len(term) > len(lemmatized_to_original_map[lem_term]):
                lemmatized_to_original_map[lem_term] = term
        standard_terms_lemmatized = list(lemmatized_to_original_map.keys())

        cui_cache = load_json_cache(CUI_MAPPING_FILE)
        atom_cache = load_json_cache(CUI_ATOM_MAPPING_FILE)

        cui_to_standard_map = create_cui_to_standard_term_map(standard_terms, umls_client, cui_cache)

        # Load embeddings cache and generate embeddings for terms not in cache
        logger.info("Loading embeddings cache...")
        embeddings_cache = load_embeddings_cache()

        # Get all unique terms (both non-standard and standard) that need embeddings
        all_terms_for_embedding = list(non_standard_terms) + standard_terms
        terms_needing_embeddings = [term for term in all_terms_for_embedding if term not in embeddings_cache]

        if terms_needing_embeddings:
            logger.info(f"Generating embeddings for {len(terms_needing_embeddings)} new terms...")
            new_embeddings = get_embeddings_batch(terms_needing_embeddings)
            embeddings_cache.update(new_embeddings)
            save_embeddings_cache(embeddings_cache)
        else:
            logger.info("All terms already have embeddings cached.")

        final_map = create_standardization_map(
            list(non_standard_terms),
            standard_terms,
            standard_terms_lemmatized,
            lemmatized_to_original_map,
            cui_to_standard_map,
            umls_client,
            cui_cache,
            atom_cache,
            embeddings_cache
        )

        save_json_cache(cui_cache, CUI_MAPPING_FILE, "CUI mappings")
        save_json_cache(atom_cache, CUI_ATOM_MAPPING_FILE, "CUI Atom mappings")

        standardized_reviews_df = apply_standardization(reviews_df, final_map)

        logger.info("Saving results...")
        standardized_reviews_df.to_csv(OUTPUT_FILE, index=False)

        # Generate and save the final standardization report
        logger.info("Generating final standardization report...")
        report = generate_standardization_report(non_standard_terms, final_map)

        # Save the report
        try:
            with open(STANDARDIZATION_REPORT_FILE, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Standardization report saved to {STANDARDIZATION_REPORT_FILE}")

            # Print a summary of the report
            print("\n--- Standardization Report Summary ---")
            print(f"Total Unique Terms: {report['total_unique_terms']}")
            print(f"Mapped Terms: {report['mapped_terms']}")
            print(f"Unmapped Terms: {report['unmapped_terms']}")
            print("\nMapping Methods Used:")

            for method, count in sorted(report['mapping_methods'].items()):
                print(f"  • {method}: {count}")
            print("------------------------------------")

        except Exception as e:
            logger.error(f"Could not save or print the report. Error: {e}")

        print("\n" + "="*60)
        print("STANDARDIZATION COMPLETE!")
        print("="*60)
        print(f"✅ Standardized data saved to: {OUTPUT_FILE}")
        print(f"✅ CUI mappings saved to: {CUI_MAPPING_FILE}")
        print(f"✅ CUI Atom mappings saved to: {CUI_ATOM_MAPPING_FILE}")
        print(f"✅ Embeddings cache saved to: {EMBEDDINGS_CACHE_FILE}")
        print(f"✅ Full report saved to: {STANDARDIZATION_REPORT_FILE}")
        print("="*60)

    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        import traceback
        traceback.print_exc()
        print("An error occurred. Check logs for details.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        import traceback
        traceback.print_exc()
        print("An error occurred. Check logs for details.")
