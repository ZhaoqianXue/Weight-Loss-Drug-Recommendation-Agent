# code_extraction/schema_extraction.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Union, Any
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

################################################################################################################
# 0. File path and global constants
################################################################################################################
INPUT_CSV_PATH = "data_webmd/webmd_all_reviews.csv"
OUTPUT_CSV_PATH = "data_extracted/extracted_reviews_top10.csv"
df = pd.read_csv(INPUT_CSV_PATH)
# Initialize the LLM with structured output
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.2,
    api_key="" # Add your API key here
)

################################################################################################################
# 1. Define Pydantic Models for Structured Output
################################################################################################################
class DrugInfo(BaseModel):
    """Drug information extraction model"""
    name: str = Field(..., description="Drug brand name from CSV Brand Name column")
    dosage: Optional[str] = Field(None, description="Extracted dosage with units (e.g., '5mg', '10 mg')")
    dosage_form: Optional[str] = Field(None, description="Dosage form based on drug type")
    duration: Optional[str] = Field(None, description="Medication duration from CSV Medication Duration column")
    continued_use: Optional[str] = Field(None, description="Whether patient continues use (yes/no)")
    alternative_drug_considered: Optional[str] = Field(None, description="Whether alternative drugs were considered (yes/no)")
    
    @field_validator('dosage_form', mode='before')
    @classmethod
    def set_dosage_form(cls, v, values):
        drug_name = values.data.get('name')
        if drug_name:
            if drug_name.lower() in ['mounjaro', 'ozempic', 'wegovy', 'zepbound', 'victoza', 'saxenda']:
                return 'subcutaneous injection'
            elif drug_name.lower() == 'rybelsus':
                return 'oral'
        return None

class ConditionInfo(BaseModel):
    """Disease/condition information model"""
    name: Optional[str] = Field(None, description="Disease or condition name from CSV or extracted from text")
    severity: Optional[str] = Field(None, description="Severity level - only if explicitly mentioned")

class SideEffectInfo(BaseModel):
    """Side effect information model"""
    name: Union[str, List[str]] = Field(..., description="Side effect name(s)")
    severity: Optional[str] = Field(None, description="Severity of side effect. Categorize as: Mild, Moderate, or Severe, only if explicitly mentioned.")
    associated_drug: Optional[str] = Field(None, description="Drug associated with this side effect")

class NodeProperties(BaseModel):
    """Properties for graph nodes"""
    name: str = Field(..., description="Name of the entity")

class GraphNode(BaseModel):
    """Graph node model"""
    label: str = Field(..., description="Node label type")
    properties: NodeProperties = Field(..., description="Node properties")

class RelationProperties(BaseModel):
    """Properties for graph relations"""
    approval: Optional[str] = Field(None, description="Approval status")
    off_label: Optional[str] = Field(None, description="Off-label usage")
    severity: Optional[str] = Field(None, description="Severity for side effects")
    dosage: Optional[str] = Field(None, description="Dosage for side effects")

class GraphRelation(BaseModel):
    """Graph relation model"""
    start: GraphNode = Field(..., description="Starting node")
    end: GraphNode = Field(..., description="Ending node")
    relation: str = Field(..., description="Relation type (treats/causes)")
    properties: RelationProperties = Field(..., description="Relation properties")

class StructuredInfo(BaseModel):
    """Main structured information container"""
    drug: DrugInfo = Field(..., description="Drug information with brand name from CSV")
    condition: Optional[ConditionInfo] = Field(None, description="Condition information from CSV or extracted from text")
    side_effects: List[SideEffectInfo] = Field(default_factory=list, description="List of side effects")

class ExtractedData(BaseModel):
    """Complete extraction output model"""
    structured_info: StructuredInfo = Field(..., description="Structured information extracted from text")
    relations: List[GraphRelation] = Field(default_factory=list, description="Graph relations for knowledge graph")

################################################################################################################
# 2. Set up LangChain with Structured Output
################################################################################################################
# Create structured output chain
structured_llm = llm.with_structured_output(ExtractedData)
# Define the prompt template - modified to exclude drug name extraction
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a medical NLP assistant tasked with extracting structured knowledge from patient reviews for building a knowledge graph. Extract information according to these strict guidelines:

### Important Note:
The primary drug name "{drug_name}", condition "{condition_from_csv}", and medication duration "{medication_duration}" have been identified from the dataset. However, the review text may mention other medications and conditions as well.

### Strict Guidelines:
1. **Medication Information** (primary drug is pre-provided):
   - Use the provided primary drug name: {drug_name}
   - dosage: Extract dosage with units only if mentioned in text (e.g., '5mg', '10 mg')
   - dosage_form: Will be automatically set based on drug name
   - duration: Will be automatically set from CSV Medication Duration - DO NOT extract from text
   - continued_use: Extract whether patient continues use (yes/no) if mentioned
   - alternative_drug_considered: Set to "yes" only if:
     a) Another drug is mentioned as alternative/switch, OR
     b) Explicit comparison statements exist

2. **Disease/Condition Information** (condition handling logic):
   - The CSV condition is: "{condition_from_csv}"
   - Apply the following logic:
     a) If CSV condition is "Other": Always use "Other"
     b) If CSV condition is NOT "Other": Always use the CSV condition "{condition_from_csv}" regardless of what's in the text
   - Only extract severity if explicitly mentioned in the review text
   - Never infer condition from side effects or other indirect clues

3. **Side Effects** (CRITICAL - Drug Attribution):
   - Only extract when clearly described in the review text
   - If severity isn't mentioned, set to null
   - **IMPORTANT**: For each side effect, determine which specific drug caused it
   - Set "associated_drug" field based on the following rules:
     a) If the text explicitly states which drug caused the side effect (e.g., "Drug A caused nausea", "I got headaches from Drug B"), use that specific drug name
     b) If the text mentions multiple drugs but doesn't specify which caused the side effect, use the primary drug name "{drug_name}"
     c) If only the primary drug "{drug_name}" is mentioned, use "{drug_name}"
   - Pay careful attention to phrases like: "switched from X to Y", "X caused Z", "Y didn't have side effects", etc.
   - **Severity**: If severity is mentioned, categorize it strictly as one of the following: "Mild", "Moderate", or "Severe". If severity is mentioned but does not fit these categories, set to null.

4. **Null Handling**:
   - All missing/unmentioned fields MUST be null except where specified
   - Empty arrays for side_effects if none mentioned
   - For condition: follow the logic in point 2 above
   - For associated_drug: never leave null if side effect is mentioned - default to primary drug "{drug_name}"

5. **Relations**:
   - Include medication-disease relations if condition is present (following condition logic above)
   - Include medication-side effect relations for each side effect using the specific associated_drug
   - Use "treats" for medication-disease relations
   - Use "causes" for medication-side effect relations

### Processing Rules:
1. Focus on identifying which specific drug causes each side effect
2. Use the primary drug "{drug_name}" as default for side effects when attribution is unclear
3. Follow the condition logic strictly based on whether CSV condition is "Other" or not
4. Extract only explicit information - no inference or assumptions
5. Be conservative - if unsure about drug attribution, default to primary drug "{drug_name}"
6. Pay special attention to comparative statements and drug switching scenarios"""),
    ("human", """Analyze this patient review text and extract structured information (drug name is pre-provided as {drug_name}, condition from CSV is {condition_from_csv}, medication duration from CSV is {medication_duration}):

Review Text: {text}

Remember: 
- Use "{drug_name}" as the drug name
- For condition: If CSV condition is "Other", extract from text or keep "Other". If CSV condition is NOT "Other", use "{condition_from_csv}"
- Duration will be automatically set from CSV - do not extract from text
- Focus on extracting other information from the review text and determining drug-side effect attributions.""")
])

# Create the extraction chain
extraction_chain = extraction_prompt | structured_llm

################################################################################################################
# 3. Safe extraction function with error handling
################################################################################################################
def safe_extract_structured(text: str, drug_name: str, condition_from_csv: str = None, medication_duration: str = None) -> Optional[dict]:
    """
    Extract structured information from text using LangChain structured output
    Args:
        text (str): The review text to process
        drug_name (str): The brand name from CSV
        condition_from_csv (str): The condition from CSV
        medication_duration (str): The medication duration from CSV
    Returns:
        Optional[dict]: Extracted structured data or None if extraction fails
    """
    try:
        # Invoke the structured extraction chain with pre-provided drug name and condition
        result = extraction_chain.invoke({
            "text": text, 
            "drug_name": drug_name,
            "condition_from_csv": condition_from_csv or "Other",
            "medication_duration": medication_duration
        })
        # Convert Pydantic model to dictionary for compatibility
        extracted_dict = result.dict()       
        # Post-process to ensure strict adherence to requirements
        processed_result = post_process_extraction(extracted_dict, drug_name, condition_from_csv, medication_duration)
        return processed_result
    except Exception as e:
        print(f"Extraction failed for {drug_name}: {str(e)[:100]}...")
        return None

def post_process_extraction(extracted_dict: dict, drug_name: str, condition_from_csv: str = None, medication_duration: str = None) -> dict:
    """
    Post-process extracted data to ensure strict adherence to requirements
    Args:
        extracted_dict (dict): Raw extracted dictionary
        drug_name (str): The brand name from CSV
        condition_from_csv (str): The condition from CSV
        medication_duration (str): The medication duration from CSV      
    Returns:
        dict: Processed dictionary following strict guidelines
    """
    # Ensure drug name is correctly set from CSV
    structured_info = extracted_dict.get('structured_info', {})
    if structured_info.get('drug'):
        structured_info['drug']['name'] = drug_name
        # Set duration from CSV Medication Duration
        if medication_duration and str(medication_duration).strip() and str(medication_duration).strip().lower() != 'nan':
            structured_info['drug']['duration'] = str(medication_duration).strip()
        else:
            structured_info['drug']['duration'] = None

    # Apply condition logic based on CSV condition value
    if condition_from_csv and str(condition_from_csv).strip().lower() != 'other':
        # If CSV condition is NOT "Other", always use CSV condition
        if structured_info.get('condition'):
            structured_info['condition']['name'] = condition_from_csv
        else:
            structured_info['condition'] = {'name': condition_from_csv, 'severity': None}
    else:
        # If CSV condition is "Other" or empty, always use "Other"
        structured_info['condition'] = {'name': 'Other', 'severity': None}
    
    # Process side effects and ensure associated_drug is properly set
    if structured_info.get('side_effects'):
        valid_severities = {"mild", "moderate", "severe"}
        for side_effect in structured_info['side_effects']:
            # If associated_drug is not specified or is null, default to primary drug from CSV
            if not side_effect.get('associated_drug'):
                side_effect['associated_drug'] = drug_name
            # Validate and normalize severity
            severity = side_effect.get('severity')
            if severity and isinstance(severity, str):
                cleaned_severity = severity.strip().lower()
                if cleaned_severity not in valid_severities:
                    side_effect['severity'] = None # Set to None if not one of the valid categories
                else:
                    # Capitalize first letter for consistency
                    side_effect['severity'] = cleaned_severity.capitalize()
            else:
                side_effect['severity'] = None # Ensure it's None if not a string or empty

    # Clean up relations based on available information
    relations = []
    # Add medication-disease relations only if condition is present
    if (structured_info.get('drug') and structured_info.get('condition') and 
        structured_info.get('condition', {}).get('name')):
        relations.append({
            "start": {
                "label": "Medication",
                "properties": {"name": drug_name}
            },
            "end": {
                "label": "Disease",
                "properties": {"name": structured_info['condition']['name']}
            },
            "relation": "treats",
            "properties": {
                "approval": None,
                "off_label": None
            }
        })
    # Add medication-side effect relations using the specific associated_drug
    if structured_info.get('side_effects'):
        for side_effect in structured_info['side_effects']:
            side_effect_names = side_effect.get('name', [])
            if isinstance(side_effect_names, str):
                side_effect_names = [side_effect_names]
            # Use the associated_drug (which now defaults to primary drug if not specified)
            associated_drug = side_effect.get('associated_drug', drug_name)
            for side_effect_name in side_effect_names:
                relations.append({
                    "start": {
                        "label": "Medication",
                        "properties": {"name": associated_drug}
                    },
                    "end": {
                        "label": "SideEffect",
                        "properties": {"name": side_effect_name}
                    },
                    "relation": "causes",
                    "properties": {
                        "severity": side_effect.get('severity'),
                        "dosage": structured_info.get('drug', {}).get('dosage')
                    }
                })
    extracted_dict['relations'] = relations
    return extracted_dict

################################################################################################################
# 4. Process reviews with structured extraction
################################################################################################################
def process_reviews_structured(dataframe: pd.DataFrame, num_reviews: int = 10) -> pd.DataFrame:
    """
    Process reviews using structured extraction with brand names from CSV
    Args:
        dataframe (pd.DataFrame): Input dataframe with reviews
        num_reviews (int): Number of reviews to process
    Returns:
        pd.DataFrame: Processed dataframe with extracted information
    """
    results = []
    df_subset = dataframe.head(num_reviews)
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), 
                      desc=f"Processing first {num_reviews} reviews with structured extraction"):
        # Use Brand Name from CSV directly
        brand_name = row.get('Brand Name', '')
        condition_from_csv = row.get('Condition', '')
        medication_duration = row.get('Medication Duration', '')
        if not brand_name:
            print(f"Warning: No brand name found for row {row.name}")
            results.append(row.to_dict())
            continue
        extracted = safe_extract_structured(
            text=row['Textual Review'], 
            drug_name=brand_name,
            condition_from_csv=condition_from_csv,
            medication_duration=medication_duration
        )
        if extracted:
            # Combine original row data with extracted structured information
            result_row = {**row.to_dict()}
            # Convert structured_info and relations to JSON strings
            result_row['structured_info'] = json.dumps(extracted['structured_info'], ensure_ascii=False)
            result_row['relations'] = json.dumps(extracted['relations'], ensure_ascii=False)
            results.append(result_row)
        else:
            # Keep original row even if extraction failed
            results.append(row.to_dict())
    return pd.DataFrame(results)

################################################################################################################
# 5. Execute the structured extraction process
################################################################################################################
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(INPUT_CSV_PATH)
    # Display available brand names in the dataset
    print("Available brand names in dataset:")
    brand_names = df['Brand Name'].value_counts()
    print(brand_names)
    print(f"\nTotal unique brand names: {len(brand_names)}")
    # Process the first 10 reviews using structured extraction
    results_df = process_reviews_structured(df, num_reviews=10)
    # Display results
    print(f"\nStructured extraction results (first 10 reviews):")
    print(f"Successfully processed {len(results_df)} reviews")
    # Show sample of extracted data
    if len(results_df) > 0:
        print("\nSample extracted structured information:")
        for idx, row in results_df.head(3).iterrows():
            if 'structured_info' in row:
                print(f"\nReview {idx + 1}:")
                print(f"Brand Name from CSV: {row.get('Brand Name')}")
                # Parse JSON strings to display content
                structured_info = json.loads(row['structured_info']) if row['structured_info'] else {}
                relations = json.loads(row['relations']) if row['relations'] else []
                print(f"Drug in extracted data: {structured_info.get('drug', {}).get('name')}")
                print(f"Medication Duration from CSV: {row.get('Medication Duration')}")
                print(f"Duration in extracted data: {structured_info.get('drug', {}).get('duration')}")
                print(f"Condition from CSV: {row.get('Condition')}")
                print(f"Condition extracted from text: {structured_info.get('condition')}")
                print(f"Side Effects: {len(structured_info.get('side_effects', []))} found")
                print(f"Relations: {len(relations)} created")
                # Show side effects if any
                side_effects = structured_info.get('side_effects', [])
                if side_effects:
                    print("Side effects found:")
                    for se in side_effects:
                        print(f"  - {se.get('name')} (severity: {se.get('severity')})")
    # Optionally save results
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nExtraction completed using Brand Name from CSV instead of LLM extraction.")