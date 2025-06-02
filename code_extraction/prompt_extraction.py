# code_extraction/prompt_extraction.py
from openai import OpenAI
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import openai

file_path = "data_webmd/webmd_all_reviews.csv"
df = pd.read_csv(file_path)

################################################################################################################
# 1. Extract structured information from reviews
################################################################################################################

client = OpenAI(api_key="")  # Add your API key here)
def safe_extract(text):
    try:
        prompt = f"""
You are a medical NLP assistant tasked with extracting structured knowledge from patient reviews for building a knowledge graph. From the given text, identify and extract the following information in JSON format:

Required entities and attributes:
- drug: name, dosage, dosage_form, continued_use (yes/no), alternative_drug_considered (yes/no)
- disease: name, severity (ONLY if explicitly mentioned)
- side_effects: name, severity, frequency, duration

### Strict Guidelines:
1. **Medication Identification**:
   - Return the drug name only if the text contains Ozempic, Mounjaro, Wegovy, Rybelsus, Zepbound, Victoza or Saxenda; return null if no match is found.
   - If multiple medications are mentioned, capture all in separate entries
   - For side effects: Only attribute them to medications when explicitly stated (e.g., "X caused Y")
   - dosage: Extracted dosage with units (e.g., '5mg', '10 mg')
   - dosage_form: For any text matching Mounjaro, Ozempic, Wegovy, Zepbound, Victoza, or Saxenda, always return 'subcutaneous injection' as the dosage form. For Rybelsus, always return 'oral'.
                  If no drug name is explicitly mentioned, return null.
   - "alternative_drug_considered": Set to "yes" only if:
     a) Another drug is mentioned as alternative/switch, OR
     b) Explicit comparison statements exist

2. **Disease Information**:
   - Only extract disease/condition if EXPLICITLY mentioned
   - If no disease/condition is mentioned, set "disease" to null
   - Never infer disease from side effects or other indirect clues

3. **Side Effects**:
   - Only extract when clearly described
   - Must maintain medication-side effect attribution if specified
   - If severity/frequency/duration aren't mentioned, set to null

4. **Null Handling**:
   - All missing/unmentioned fields MUST be null
   - Empty arrays for side_effects if none mentioned
   - Omit entire disease object if no information

5. **Output Format**:
   - Strictly follow the JSON schema below
   - No explanatory text
   - No partial/inferred data

JSON Output Format:
{{
  "structured_info": {{
    "drug": {{
      "name": "",  // as array if multiple
      "dosage": "",
      "dosage_form": "",
      "continued_use": "",
      "alternative_drug_considered": ""
    }},
    "condition": null,  // ENTIRE OBJECT null if no disease info
    "side_effects": [
      {{
        "name": "", // as array if multiple
        "severity": "",
        "frequency": "",
        "duration": "",
        "associated_drug": ""
      }}
    ],
  }},
  "relations": [
    // ONLY include if disease is mentioned
    {{
      "start": {{
        "label": "Medication",
        "properties": {{
          "name": ""
        }}
      }},
      "end": {{
        "label": "Disease",
        "properties": {{
          "name": ""
        }}
      }},
      "relation": "treats",
      "properties": {{
        "approval": null,
        "off_label": null
      }}
    }},
    // For each side effect
    {{
      "start": {{
        "label": "Medication",
        "properties": {{
          "name": ""
        }}
      }},
      "end": {{
        "label": "SideEffect",
        "properties": {{
          "name": ""
        }}
      }},
      "relation": "causes",
      "properties": {{
        "severity": null,
        "duration": null,
        "dosage": null
      }}
    }}
  ]
}}

### Processing Rules:
1. Medication Names:
   - Standardize variations (e.g., "Mounjaro 5mg" → "Mounjaro")
   - Include dosage in separate field when available

2. Disease Handling:
   - Return null for entire disease object if:
     - No disease mentioned
     - Only vague references ("my condition")
     - Only implied through side effects

3. Sentiment Analysis:
   - Categorize as: "positive", "negative", "neutral"

4. Effectiveness:
   - Extract only explicit descriptions
   - Convert to standardized terms if possible ("very effective" → "high")

Text to Analyze:
{text}

Respond with ONLY the JSON output.
"""

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "system", "content": "You must only return output in strict JSON format as required"},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"fail: {str(e)[:100]}...")
        return None

################################################################################################################
# 2. Extract structured information from reviews
################################################################################################################

results = []

df_first10 = df.head(10)

for _, row in tqdm(df_first10.iterrows(), total=len(df_first10), desc="Processing first 10 reviews"):
    extracted = safe_extract(row['Textual Review'])
    if extracted:
        results.append({**row.to_dict(), **extracted})
final_df = pd.DataFrame(results)

final_df.to_csv('data_extracted/extracted_reviews_raw10.csv', index=False)

print("\nExtracted results (first 10):")
print(final_df.head(2))