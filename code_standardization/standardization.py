# code_standardization/standardization.py

import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers.util import cos_sim
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer

################################################################################################################
# 0. File path and global constants
################################################################################################################
INPUT_REVIEW_PATH = "data_extracted/extracted_reviews_top10.csv"
INPUT_EMBEDDED_AE_PATH = "data_embedded/embedded_ae.csv"
OUTPUT_STANDARDIZED_PATH = "data_standardized/standardized_reviews_top10.csv"
THRESHOLD = 0.0  # Default similarity threshold for AE matching
review_df = pd.read_csv(INPUT_REVIEW_PATH)
merged_df_embedded = pd.read_csv(INPUT_EMBEDDED_AE_PATH)

################################################################################################################
# 1. Configuration and Model Initialization
################################################################################################################
# Initialize the sentence transformer model for semantic similarity
MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
# MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
# OpenAI configuration
OPENAI_API_KEY = "" # Add your API key here
openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_LLM_MODEL = "gpt-4.1-nano"
OPENAI_LLM_TEMPERATURE = 0.2
# GPT prompt template for adverse event matching
GPT_MATCHING_PROMPT = (
    "You are a clinical NLP assistant helping map patient-reported side effects to standardized adverse events (AEs).\n\n"
    "A user has reported the side effect: '{side_effect}'\n\n"
    "Below are several standardized AEs. Your task is to choose the one that most accurately matches the reported term.\n\n"
    "Only choose an AE if it clearly matches the meaning of the reported term.\n"
    "Do NOT choose an AE based on partial overlap or general similarity. For example, 'sickness' and 'morning sickness' are NOT the same.\n\n"
    "{ae_list}\n\n"
    "If none of the options is an unambiguous match, reply with an empty string.\n\n"
    "Best matching AE:"
)

# GPT prompt template for FDA approval classification
GPT_FDA_APPROVAL_PROMPT = (
    "You are a clinical regulatory expert analyzing FDA drug approvals.\n\n"
    "Given the medication '{medication}' and the disease/condition '{disease}', determine if this is an FDA-approved indication or off-label use.\n\n"
    "FDA Prescribing Information:\n"
    "{fda_info}\n\n"
    "Instructions:\n"
    "- If the disease/condition is explicitly mentioned as an approved indication for this medication in the FDA information, respond with 'approval'\n"
    "- If the disease/condition is NOT mentioned as an approved indication for this medication, respond with 'off_label'\n"
    "- Consider synonyms and related medical terms (e.g., 'diabetes' matches 'type 2 diabetes mellitus')\n"
    "- Be precise - only classify as 'approval' if there is clear evidence in the FDA information\n\n"
    "Classification (approval or off_label):"
)

# FDA prescribing information for known medications - Complete INDICATIONS AND USAGE
FDA_PRESCRIBING_INFO = {
    "wegovy": """
    WEGOVY is indicated as an adjunct to a reduced calorie diet and increased physical activity for chronic weight management in:
    - adults with an initial body mass index (BMI) of [see Dosage and Administration (2.1)]:
        - 30 kg/m^2 or greater (obesity) or
        - 27 kg/m^2 or greater (overweight) in the presence of at least one weight-related comorbid condition (e.g., hypertension, type 2 diabetes mellitus, or dyslipidemia)
    - pediatric patients aged 12 years and older with an initial BMI at the 95th percentile or greater standardized for age and sex (obesity) [see Dosage and Administration (2.1)]
    Limitation of Use
    - WEGOVY contains semaglutide and should not be coadministered with other semaglutide-containing products or with any other GLP-1 receptor agonist.
    - The safety and effectiveness of WEGOVY in combination with other products intended for weight loss, including prescription drugs, over-the-counter drugs, and herbal preparations, have not been established.
    - WEGOVY has not been studied in patients with a history of pancreatitis [see Warnings and Precautions (5.2)].
    """,
    "ozempic": """
    OZEMPIC is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus [see Clinical Studies (14.1)]. 
    Limitations of Use
    - OZEMPIC is not recommended as a first-line therapy for patients who have inadequate glycemic control on diet and exercise because of the uncertain relevance of rodent C-cell tumor findings to humans [see Warnings and Precautions (5.1)]. 
    - OZEMPIC has not been studied in patients with a history of pancreatitis. Consider other antidiabetic therapies in patients with a history of pancreatitis [see Warnings and Precautions (5.2)]. 
    - OZEMPIC is not a substitute for insulin. 
    - OZEMPIC is not indicated for use in patients with type 1 diabetes mellitus or for the treatment of patients with diabetic ketoacidosis, as it would not be effective in these settings.
    """,
    "rybelsus": """
    RYBELSUS is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus.
    Limitations of Use
    - RYBELSUS has not been studied in patients with a history of pancreatitis. Consider other antidiabetic therapies in patients with a history of pancreatitis [see Warnings and Precautions (5.2)].
    - RYBELSUS is not indicated for use in patients with type 1 diabetes mellitus.
    """,
    "zepbound": """
    ZEPBOUND™ is indicated as an adjunct to a reduced-calorie diet and increased physical activity for chronic weight management in adults with an initial body mass index (BMI) of:
    - 30 kg/m^2 or greater (obesity) or
    - 27 kg/m^2 or greater (overweight) in the presence of at least one weight-related comorbid condition (e.g., hypertension, dyslipidemia, type 2 diabetes mellitus, obstructive sleep apnea, or cardiovascular disease).
    Limitations of Use
    - ZEPBOUND contains tirzepatide. Coadministration with other tirzepatide-containing products or with any glucagon like peptide-1 (GLP-1) receptor agonist is not recommended.
    - The safety and efficacy of ZEPBOUND in combination with other products intended for weight management, including prescription drugs, over-the-counter drugs, and herbal preparations, have not been established.
    - ZEPBOUND has not been studied in patients with a history of pancreatitis [see Warnings and Precautions (5.5)].
    """,
    "mounjaro": """
    MOUNJARO™ is indicated as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus. 
    Limitations of Use 
    * MOUNJARO has not been studied in patients with a history of pancreatitis [see Warnings and Precautions (5.2)]. 
    * MOUNJARO is not indicated for use in patients with type 1 diabetes mellitus.
    """,
    "victoza": """
    VICTOZA is indicated:
    - as an adjunct to diet and exercise to improve glycemic control in adults with type 2 diabetes mellitus,
    - to reduce the risk of major adverse cardiovascular events (cardiovascular death, non-fatal myocardial infarction, or non-fatal stroke) in adults with type 2 diabetes mellitus and established cardiovascular disease [see Clinical Studies (14.2)].
    Limitations of Use:
    - VICTOZA is not a substitute for insulin. VICTOZA should not be used in patients with type 1 diabetes mellitus or for the treatment of diabetic ketoacidosis, as it would not be effective in these settings.
    - The concurrent use of VICTOZA and prandial insulin has not been studied.
    """,
    "saxenda": """
    SAXENDA is indicated as an adjunct to a reduced-calorie diet and increased physical activity for chronic weight management in: 
    * Adult patients with an initial body mass index (BMI) of [see Dosage and Administration (2.1)]:
        * 30 kg/m^2 or greater (obese), or 
        * 27 kg/m^2 or greater (overweight) in the presence of at least one weight-related comorbid condition (e.g., hypertension, type 2 diabetes mellitus, or dyslipidemia) 
    * Pediatric patients aged 12 years and older with:
        * body weight above 60 kg and
        * an initial BMI corresponding to 30 kg/m^2 or greater for adults (obese) by international cut-offs (Cole Criteria, Table 2) [see Dosage and Administration (2.1)] 
    Limitations of Use
    * SAXENDA contains liraglutide and should not be coadministered with other liraglutide-containing products or with any other GLP-1 receptor agonist.
    * The safety and effectiveness of SAXENDA in pediatric patients with type 2 diabetes have not been established.
    * The safety and effectiveness of SAXENDA in combination with other products intended for weight loss, including prescription drugs, over-the-counter drugs, and herbal preparations, have not been established.
    """
}

################################################################################################################
# 2. Data Preprocessing Functions
################################################################################################################
def prepare_ae_data(merged_df_embedded: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Prepare adverse event texts and embeddings for similarity matching.
    Args:
        merged_df_embedded (pd.DataFrame): Dataframe containing AE texts and embeddings
    Returns:
        Tuple[List[str], np.ndarray]: List of AE texts and their corresponding embeddings
    """
    ae_texts = merged_df_embedded['AE'].tolist()
    ae_embeddings = merged_df_embedded[[f'emb_{i}' for i in range(768)]].dropna().values
    print(f"Prepared {len(ae_texts)} AE texts with embeddings shape: {ae_embeddings.shape}")
    return ae_texts, ae_embeddings

################################################################################################################
# 3. Side Effect Extraction and Processing Functions
################################################################################################################
def extract_side_effects(structured_info_str: str) -> List[str]:
    """
    Extract all side effect names from structured information string (handle both string and list cases).
    Args:
        structured_info_str (str): JSON string containing structured information
    Returns:
        List[str]: List of side effect names
    """
    try:
        info = json.loads(structured_info_str)
        side_effects = info.get("side_effects", [])
        extracted_effects = []
        for se in side_effects:
            name = se.get("name")
            if isinstance(name, str):
                extracted_effects.append(name.strip())
            elif isinstance(name, list):
                extracted_effects.extend([n.strip() for n in name if isinstance(n, str)])
        return extracted_effects
    except Exception as e:
        print(f"Error extracting side effects: {e}")
        return []

def find_top_matches(effect: str, model: SentenceTransformer, ae_texts: List[str], 
                    ae_embeddings: np.ndarray, threshold: float = THRESHOLD) -> List[str]:
    """
    Return the top 10 most similar standardized AEs for the given side effect, filtered by a similarity threshold.
    Args:
        effect (str): Side effect to match
        model (SentenceTransformer): Sentence transformer model for encoding
        ae_texts (List[str]): List of standardized AE texts
        ae_embeddings (np.ndarray): Pre-computed embeddings for AE texts
        threshold (float): Minimum similarity threshold for matches (default THRESHOLD)
    Returns:
        List[str]: Top 10 most similar AE texts with similarity >= threshold
    """
    try:
        # Encode the side effect query
        query_emb = model.encode(effect, convert_to_numpy=True, normalize_embeddings=True)
        # Calculate similarities with all AE embeddings
        similarities = np.dot(ae_embeddings, query_emb)
        # Get top 10 matches (no threshold yet)
        top_indices = similarities.argsort()[::-1][:10]
        # Filter by threshold
        top_matches = [ae_texts[i] for i in top_indices if similarities[i] >= threshold]
        return top_matches
    except Exception as e:
        print(f"Error finding matches for '{effect}': {e}")
        return []

################################################################################################################
# 4. GPT-based Matching and Resolution Functions
################################################################################################################
def gpt_resolve_match(side_effect: str, ae_candidates: List[str]) -> Optional[str]:
    """
    Use GPT to resolve the best matching adverse event from candidates.
    Args:
        side_effect (str): Original side effect reported by patient
        ae_candidates (List[str]): List of candidate standardized AEs
    Returns:
        Optional[str]: Best matching AE or None if no clear match
    """
    if not ae_candidates:
        return None
    # Format the candidate list for the prompt
    formatted_ae_list = "\n".join(f"- {ae}" for ae in ae_candidates)
    formatted_prompt = GPT_MATCHING_PROMPT.format(
        side_effect=side_effect,
        ae_list=formatted_ae_list
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=OPENAI_LLM_TEMPERATURE
        )
        result = response.choices[0].message.content.strip()
        # Validate that the result is actually one of the candidates
        return result if result and result in ae_candidates else None
    except Exception as e:
        print(f"Error in GPT resolution for '{side_effect}': {e}")
        return None

def classify_fda_approval(medication: str, disease: str) -> str:
    """
    Use GPT to classify whether a medication-disease combination is FDA approved or off-label.
    Args:
        medication (str): Name of the medication
        disease (str): Name of the disease/condition
    Returns:
        str: 'approval' if FDA approved, 'off_label' if not approved for this indication
    """
    # Normalize medication name for lookup
    med_key = medication.lower().strip()
    
    # Get FDA information for this medication
    fda_info = FDA_PRESCRIBING_INFO.get(med_key, "No FDA prescribing information available for this medication.")
    
    # Format the prompt
    formatted_prompt = GPT_FDA_APPROVAL_PROMPT.format(
        medication=medication,
        disease=disease,
        fda_info=fda_info
    )
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=OPENAI_LLM_TEMPERATURE
        )
        result = response.choices[0].message.content.strip().lower()
        # Ensure we return either 'approval' or 'off_label'
        if 'approval' in result:
            return 'approval'
        else:
            return 'off_label'
    except Exception as e:
        print(f"Error in FDA classification for '{medication}' and '{disease}': {e}")
        return 'off_label'  # Default to off_label if there's an error

################################################################################################################
# 5. Data Update and Standardization Functions
################################################################################################################
def update_structured_info(structured_info_str: str, replacements: Dict[str, str]) -> str:
    """
    Update structured information with standardized side effect names (handle both string and list cases).
    Args:
        structured_info_str (str): Original structured information as JSON string
        replacements (Dict[str, str]): Mapping of original to standardized names
    Returns:
        str: Updated structured information as JSON string
    """
    try:
        info = json.loads(structured_info_str)
        for se in info.get("side_effects", []):
            name = se.get("name")
            if isinstance(name, str):
                if name in replacements:
                    se["name"] = replacements[name]
            elif isinstance(name, list):
                se["name"] = [replacements.get(n, n) for n in name]
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        print(f"Error updating structured info: {e}")
        return structured_info_str

def update_relations_with_fda_classification(relations_str: str, replacements: Dict[str, str]) -> str:
    """
    Update relations with standardized side effect names and FDA approval classification.
    Args:
        relations_str (str): Original relations as JSON string
        replacements (Dict[str, str]): Mapping of original to standardized names
    Returns:
        str: Updated relations as JSON string with FDA classification
    """
    if not relations_str:
        return relations_str
        
    try:
        relations = json.loads(relations_str)
        for rel in relations:
            # Update side effect names
            if rel.get("relation") == "causes" and rel.get("end", {}).get("label") == "SideEffect":
                name = rel.get("end", {}).get("properties", {}).get("name")
                if name in replacements:
                    rel["end"]["properties"]["name"] = replacements[name]
            
            # Add FDA approval classification for "treats" relations with Disease
            elif rel.get("relation") == "treats" and rel.get("end", {}).get("label") == "Disease":
                medication_name = rel.get("start", {}).get("properties", {}).get("name", "")
                disease_name = rel.get("end", {}).get("properties", {}).get("name", "")
                
                if medication_name and disease_name:
                    # Classify FDA approval status
                    approval_status = classify_fda_approval(medication_name, disease_name)
                    
                    # Update the properties
                    if "properties" not in rel:
                        rel["properties"] = {}
                    
                    if approval_status == "approval":
                        rel["properties"]["approval"] = "yes"
                        rel["properties"]["off_label"] = "no"
                    else:
                        rel["properties"]["approval"] = "no"
                        rel["properties"]["off_label"] = "yes"
        
        return json.dumps(relations, ensure_ascii=False)
    except Exception as e:
        print(f"Error updating relations with FDA classification: {e}")
        return relations_str

def process_row(row: pd.Series, model: SentenceTransformer, ae_texts: List[str], 
               ae_embeddings: np.ndarray) -> dict:
    """
    Process a single review row and return both standardized structured_info and standardized_relations.
    """
    original_info = row["structured_info"]
    effects = extract_side_effects(original_info)
    replacements = {}
    for effect in effects:
        candidates = find_top_matches(effect, model, ae_texts, ae_embeddings)
        best_match = gpt_resolve_match(effect, candidates)
        if best_match:
            replacements[effect] = best_match.lower()
    
    # Standardize structured_info
    standardized_info = update_structured_info(original_info, replacements)
    
    # Standardize relations and add FDA classification
    relations_str = row.get("relations", None)
    standardized_relations = update_relations_with_fda_classification(relations_str, replacements)
    
    return {
        "standardized_info": standardized_info,
        "standardized_relations": standardized_relations
    }

################################################################################################################
# 6. Main Processing Pipeline
################################################################################################################
def standardize_side_effects(review_df: pd.DataFrame, ae_texts: List[str], 
                           ae_embeddings: np.ndarray) -> pd.DataFrame:
    """
    Standardize side effects and add a new column standardized_relations.
    """
    print(f"Starting standardization process for {len(review_df)} reviews...")
    tqdm.pandas(desc="Standardizing side effects and FDA classification")
    results = review_df.progress_apply(
        lambda row: process_row(row, model, ae_texts, ae_embeddings), axis=1
    )
    review_df["standardized_info"] = results.apply(lambda x: x["standardized_info"])
    review_df["standardized_relations"] = results.apply(lambda x: x["standardized_relations"])
    print("Standardization process completed successfully")
    return review_df

def save_results(review_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the standardized results to CSV file.
    Args:
        review_df (pd.DataFrame): Dataframe with standardized information
        output_path (str): Output file path
    """
    try:
        review_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

################################################################################################################
# 7. Execute the standardization process
################################################################################################################
if __name__ == "__main__":
    # Prepare adverse event data
    ae_texts, ae_embeddings = prepare_ae_data(merged_df_embedded)
    # Standardize side effects
    standardized_df = standardize_side_effects(review_df, ae_texts, ae_embeddings)
    # Display sample results
    print(f"\nStandardization results:")
    print(f"Successfully processed {len(standardized_df)} reviews")
    # Show sample of standardized data
    if len(standardized_df) > 0:
        print("\nSample standardized information:")
        for idx, row in standardized_df.head(5).iterrows():
            if 'standardized_info' in row:
                print(f"\nReview {idx + 1}:")
                try:
                    original_info = json.loads(row['structured_info'])
                    standardized_info = json.loads(row['standardized_info'])
                    original_effects = [se.get('name') for se in original_info.get('side_effects', [])]
                    standardized_effects = [se.get('name') for se in standardized_info.get('side_effects', [])]
                    print(f"Original side effects: {original_effects}")
                    print(f"Standardized side effects: {standardized_effects}")
                    
                    # Show FDA classification results
                    if row.get('standardized_relations'):
                        relations = json.loads(row['standardized_relations'])
                        for rel in relations:
                            if rel.get("relation") == "treats" and rel.get("end", {}).get("label") == "Disease":
                                med_name = rel.get("start", {}).get("properties", {}).get("name", "")
                                disease_name = rel.get("end", {}).get("properties", {}).get("name", "")
                                approval = rel.get("properties", {}).get("approval", "")
                                off_label = rel.get("properties", {}).get("off_label", "")
                                print(f"FDA Classification - {med_name} treats {disease_name}: approval={approval}, off_label={off_label}")
                except Exception as e:
                    print(f"Error displaying sample data: {e}")
    # Save results
    save_results(standardized_df, OUTPUT_STANDARDIZED_PATH)
    print(f"\nStandardization completed successfully.")