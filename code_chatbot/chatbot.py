# chatbot.py
import os
import json
import warnings
import re
import ast
from io import StringIO
from contextlib import redirect_stdout
from typing import Optional, List, Any, Dict, Tuple
from collections import Counter, defaultdict
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch

# Set multiprocessing start method if necessary and not already set
# This can be important for libraries like HuggingFace Transformers
# if they use multiprocessing internally.
try:
    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError as e:
    # This might happen if the start method is already set and 'force=True' is problematic
    # or if 'spawn' is not available/appropriate on the system.
    print(f"Note: Could not set multiprocessing start_method: {e}. Using default.")


# Optional Third-party imports with error handling
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, CypherSyntaxError
except ImportError:
    print("Neo4j driver not installed. Please install with 'pip install neo4j'")
    GraphDatabase = None # Placeholder
    CypherSyntaxError = type("CypherSyntaxError", (Exception,), {}) # Placeholder to avoid NameError

try:
    import faiss
    from langchain.docstore.document import Document
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # For other embedders if needed by TableRAG's original retriever
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    # from langchain_google_vertexai import VertexAIEmbeddings
    # from langchain_community.retrievers import BM25Retriever
    # from langchain.retrievers import EnsembleRetriever
except ImportError:
    print("Some Langchain or FAISS components not installed. Please ensure TableRAG dependencies are met (e.g., 'pip install faiss-cpu langchain langchain-community langchain-openai')")
    FAISS = None # Placeholder
    Document = type("Document", (object,), {}) # Placeholder
    OpenAIEmbeddings = type("OpenAIEmbeddings", (object,), {}) # Placeholder
    HuggingFaceEmbeddings = type("HuggingFaceEmbeddings", (object,), {}) # Placeholder

try:
    from openai import OpenAI
    import tiktoken
except ImportError:
    print("OpenAI Python client not installed. Please install with 'pip install openai tiktoken'")
    OpenAI = None # Placeholder
    tiktoken = type("tiktoken", (object,), {}) # Placeholder

# SentenceTransformer might be used by HuggingFaceEmbeddings implicitly,
# but good to have an explicit check if direct usage was intended elsewhere.
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Please install with 'pip install sentence-transformers'")
    SentenceTransformer = None # Placeholder


logger = logging.getLogger(__name__) # Use logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Configuration ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "weightloss" # As provided by user

# IMPORTANT: User needs to configure their actual LLM model name and API key/access.
LLM_MODEL_NAME = "gpt-4.1-nano" # User wants "gpt-4.1-nano"
# Ensure this environment variable is set, or hardcode your key (not recommended for production).
OPENAI_API_KEY = "" # Add your API key here

# TableRAG DB paths (directories containing index.faiss and index.pkl)
SCHEMA_DB_PATH = os.path.join("database_table", "schema_db_standardized_reviews_all")
CELL_DB_PATH = os.path.join("database_table", "cell_db_standardized_reviews_all")

# TableRAG conceptual table ID and CSV data path
TABLE_ID = "standardized_reviews_all"
TABLE_CSV_PATH = os.path.join("data_standardized", "standardized_reviews_all.csv") # Path to the main data CSV for the solver
TABLE_CAPTION = "WebMD Drug Reviews for Wegovy, Ozempic, Rybelsus, Zepbound, Mounjaro, Victoza, Saxenda" # Descriptive caption

# TableRAG parameters
TABLE_RAG_TOP_K = 5
TABLE_RAG_MAX_ENCODE_CELL = 10000 # Used by original TableRAG for DB building, less relevant here for pathing if pre-built DB paths are exact
TABLE_RAG_EMBED_MODEL_NAME = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# --- Helper Utilities (Adapted from TableRAG utils) ---

def parse_code_from_string(input_string: str) -> str:
    """Parse executable code from a string, handling various markdown-like code block formats."""
    # Pattern for triple backticks with optional language specifier
    triple_backtick_pattern = r"```(?:python\s*|\w*\s*)?(.*?)```"
    match = re.search(triple_backtick_pattern, input_string, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Pattern for single backticks (less common for multi-line but good fallback)
    single_backtick_pattern = r"`(.*?)`"
    match = re.search(single_backtick_pattern, input_string, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no backticks, assume the whole string is code (after stripping)
    return input_string.strip()

def python_repl_ast(code: str, custom_globals: Optional[Dict] = None, custom_locals: Optional[Dict] = None, memory: Optional[Dict] = None) -> Tuple[str, Dict]:
    """
    Run Python code using ast, capturing stdout and the result of the last expression.
    Ensures that only the final expression's value is returned if it's not None,
    otherwise captures print statements.
    """
    if memory is None:
        memory = {} # Stores state across multiple calls if needed (though solver loop passes fresh df copy)
    
    # Setup globals and locals for execution
    # Inherit from actual globals to have access to libraries like pandas, numpy
    # but isolate execution as much as possible.
    exec_globals = globals().copy()
    if custom_globals:
        exec_globals.update(custom_globals)
    
    exec_locals = memory.copy() # Start with memory from previous steps
    if custom_locals:
        exec_locals.update(custom_locals)

    output_capture = StringIO()
    try:
        # Parse the code
        parsed_code = ast.parse(code)
        
        # If the code is a single expression, eval it.
        # Otherwise, exec it.
        if len(parsed_code.body) == 1 and isinstance(parsed_code.body[0], ast.Expr):
            # This is an expression, try to eval it
            last_expr = parsed_code.body[0].value
            # Compile the expression to run in eval
            compiled_expr = compile(ast.Expression(last_expr), '<string>', 'eval')
            
            with redirect_stdout(output_capture): # Capture any prints even during eval
                result = eval(compiled_expr, exec_globals, exec_locals)
            
            stdout_val = output_capture.getvalue()
            if result is not None:
                # If eval returned something, that's our primary result
                # Prepend any stdout that occurred during eval
                final_output = stdout_val + str(result)
            else:
                # If eval returned None, stdout is the result
                final_output = stdout_val
        else:
            # This is a statement or multiple statements, exec it
            # Compile the whole code block for exec
            compiled_code = compile(parsed_code, '<string>', 'exec')
            with redirect_stdout(output_capture):
                exec(compiled_code, exec_globals, exec_locals)
            final_output = output_capture.getvalue()

        memory.update(exec_locals) # Persist changes to locals in memory
        return final_output.strip(), memory

    except Exception as e:
        # Capture any prints that occurred before the exception
        # and append the error message.
        error_message = "{}: {}".format(type(e).__name__, str(e))
        # Ensure memory is updated even on error, as some statements might have run
        memory.update(exec_locals)
        return (output_capture.getvalue() + error_message).strip(), memory


def infer_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert columns in a DataFrame to a more appropriate data type."""
    for col in df.columns:
        # Try numeric
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except (ValueError, TypeError):
            pass
        # Try boolean if applicable (e.g., if values are consistently 'yes'/'no' or 'true'/'false')
        # This is a simple check; more sophisticated boolean inference could be added.
        if df[col].nunique() <= 2: # Example: check for 2 unique values
            lowered_unique = df[col].astype(str).str.lower().unique()
            if all(val in ['true', 'false', 'yes', 'no', '0', '1', 'nan', 'none', '<na>'] for val in lowered_unique):
                try:
                    # Convert common string representations of booleans
                    bool_map = {'true': True, 'yes': True, '1': True,
                                'false': False, 'no': False, '0': False}
                    # Handle potential NaNs carefully before mapping
                    original_series = df[col].copy()
                    # Apply mapping only to non-NA values that are in bool_map keys
                    # For values not in bool_map (e.g. actual NaN, None), keep them as is to become pd.NA or similar
                    df[col] = original_series.astype(str).str.lower().map(lambda x: bool_map.get(x, pd.NA if x in ['nan', 'none', '<na>'] else original_series[df[col].astype(str).str.lower() == x].iloc[0] )).astype('boolean')

                except Exception: # If conversion fails, revert or keep as object
                    df[col] = original_series # Revert to original if bool conversion is problematic
                    pass # Keep as object if complex error

    return df

def get_df_info(df: pd.DataFrame) -> str:
    """Get string representation of df.info()"""
    buf = StringIO()
    df.info(verbose=True, buf=buf)
    return buf.getvalue()

def read_json_from_llm(text: str) -> Any:
    """Extracts JSON from LLM response, robustly."""
    text = text.strip()
    # First, try to find a ```json ... ``` block
    match_block = re.search(r"```json\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if match_block:
        json_text = match_block.group(1).strip()
    else:
        # If no markdown block, assume the whole text might be JSON or contain it.
        # Try to find the first '{' or '[' and the last '}' or ']'
        # This is a common heuristic for extracting JSON from less structured LLM outputs.
        first_bracket = text.find('{')
        first_square_bracket = text.find('[')
        
        if first_bracket == -1 and first_square_bracket == -1:
            # No opening bracket found, unlikely to be JSON
            logger.warning(f"No JSON object or array markers found in text: {text[:100]}...")
            raise ValueError("No JSON object or array found in text")

        # Determine if it's an object or array based on the first character
        if first_bracket != -1 and (first_square_bracket == -1 or first_bracket < first_square_bracket):
            start_char = '{'
            end_char = '}'
            start_index = first_bracket
        else:
            start_char = '['
            end_char = ']'
            start_index = first_square_bracket
            
        # Find the matching end bracket by balancing
        balance = 0
        end_index = -1
        for i in range(start_index, len(text)):
            if text[i] == start_char:
                balance += 1
            elif text[i] == end_char:
                balance -= 1
                if balance == 0:
                    end_index = i
                    break
        
        if end_index == -1:
            logger.warning(f"Could not find matching end bracket for JSON in text: {text[start_index:start_index+100]}...")
            raise ValueError("Could not find matching end bracket for JSON")
        
        json_text = text[start_index : end_index+1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error: {e}. Text processed for JSON: '{json_text}' (original text sample: '{text[:200]}...')")
        raise

def is_numeric(s: Any) -> bool:
    if isinstance(s, (int, float, np.number)):
        return True
    if isinstance(s, str):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False

def parse_age_range(age_str):
    """Parses an age string (range or single) into lower and upper bounds."""
    if pd.isna(age_str):
        return pd.NA, pd.NA  # Return pandas NA for missing values
    age_str = str(age_str).strip().lower()  # Standardize to lower case
    # Handle '75 or over' or '75+'
    if 'or over' in age_str or '+' in age_str:
        match = re.search(r'(\d+)', age_str)
        return (int(match.group(1)), 999) if match else (pd.NA, pd.NA)
    # Handle '65-74'
    match = re.match(r'(\d+)\s*-\s*(\d+)', age_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Handle '< 18'
    if '<' in age_str:
        match = re.search(r'(\d+)', age_str)
        return (0, int(match.group(1)) - 1) if match else (pd.NA, pd.NA)
    # Handle单一年龄数字
    try:
        age_num = int(age_str)
        return age_num, age_num
    except ValueError:
        return pd.NA, pd.NA  # Return NA if no pattern matches

# --- LLM Model Wrapper (Adapted from TableRAG agent.model) ---
class LLMWrapper:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        if not OpenAI:
            raise ImportError("OpenAI client not available. Please install with 'pip install openai tiktoken'")
        if not api_key:
            logger.error("OpenAI API key not provided to LLMWrapper.")
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning(f"Encoding for model {model_name} not found. Using cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Context limits are approximate and can change. Refer to OpenAI documentation for specifics.
        if "gpt-4.1-nano" in model_name: # As per user's model
            self.context_limit = 128000
        elif "gpt-4" in model_name: # General gpt-4
            self.context_limit = 128000
        elif "gpt-3.5-turbo" in model_name: # Common gpt-3.5 variant
            self.context_limit = 16385 # For models like gpt-3.5-turbo-1106 or 0125
        else: # A fallback default
            logger.warning(f"Unknown model {model_name} for context limit, defaulting to 8192.")
            self.context_limit = 8192


    def query(self, prompt: str, system_message: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024, stop_sequences: Optional[List[str]] = None) -> str:
        if not prompt:
            logger.error("LLM query attempt with empty prompt.")
            return "Error: Prompt cannot be empty."
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API Error with model {self.model_name}: {e}", exc_info=True)
            return f"Error: LLM query failed. Details: {e}"

    def get_token_count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

# --- TableRAG Prompts (Adapted from TableRAG prompts) ---
def get_table_rag_prompt(prompt_type: str, **kwargs) -> str:
    prompts = {
        'extract_column_prompt': """
Given a large table regarding {table_caption}, I want to answer a question: {query}
Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question.
Please answer with a list of column names in JSON format without any additional explanation.
Example:
["column_name_1", "column_name_2", "column_name_3"]
""",
        'extract_cell_prompt': """
Given a large table regarding {table_caption}, I want to answer a question: {query}
Please extract some keywords which might appear in the table cells and help answer the question.
The keywords should be categorical values rather than numerical values.
The keywords should be contained in the question.
Please answer with a list of keywords in JSON format without any additional explanation.
Example:
["keyword_1", "keyword_2", "keyword_3"]
""",
        'solve_table_prompt': """
You are an expert Python data analyst working with a pandas DataFrame named `df`.
The DataFrame contains information about: {table_caption}.
Your goal is to answer the user's question: "{query}"

You MUST use the `python_repl_ast` tool to inspect and manipulate the `df` DataFrame.
Tool description:
- `python_repl_ast`: A Python interactive shell. Input must be a single, valid line of Python code. It returns the output of the code or an error.

Available Data:
- DataFrame `df`: This is the primary data source.
- Schema Information (retrieved column names that might be relevant):
{schema_retrieval_result}
- Cell Value Samples (retrieved cell values that might be relevant):
{cell_retrieval_result}

Follow this multi-step ReAct (Reason-Act-Observe) process:
1.  **Thought**: Analyze the question and the available data. Plan your Python action. If a previous action failed, analyze the error and plan a corrected action.
2.  **Action**: Write a single line of Python code to be executed using `python_repl_ast`. Access the DataFrame as `df`.
3.  **Observation**: This will be the output from `python_repl_ast` (the result of your Python code or an error message).

Repeat Thought/Action/Observation as needed.
After sufficient observations, provide a final thought summarizing your findings and then the final answer.

**CRITICAL RESPONSE FORMAT:**
Thought: Your reasoning and plan.
Action: `df.some_method()` or `some_function(df)`
Observation: Result from `python_repl_ast`.
... (repeat)
Thought: Final summary of observations and how they answer the question.
Final Answer: The answer to the user's question.

**IMPORTANT RULES:**
- Only use the `df` DataFrame. Do not assume other variables exist unless created by your previous actions.
- Your Python code in 'Action' MUST be a single line.
- If your action results in an error, the 'Observation' will show the error. In your next 'Thought', analyze the error and try a different approach.
- Do not generate code that prints directly (e.g., `print(df.head())`). The `python_repl_ast` handles output. If you want to see the value of an expression, just write the expression.
- Ensure the final line of your response is "Final Answer: [Your answer here]". No extra text after it.
- Base your answer ONLY on the observations from the `python_repl_ast`. Do not make up information.
- If you cannot answer the question with the provided data and tools after several attempts, state that in the Final Answer.
- If you need to filter by age (e.g., 'over 60'), use the numeric columns 'Age_Lower' and 'Age_Upper' for comparison. For example, to select users over 60, use: df[df['Age_Lower'] >= 60].
- **CRITICAL**: When filtering by drug names (like Ozempic, Wegovy, Zepbound, etc.), ALWAYS use the 'Brand Name' column, NOT the 'Drug Name' column. The 'Drug Name' column contains generic names (like Semaglutide, Tirzepatide), while 'Brand Name' contains the actual brand names that users refer to.
- **CRITICAL**: For age-related filtering (e.g., 'over 60', 'under 50'), ALWAYS use the 'Age_Lower' and 'Age_Upper' columns, NOT the 'Age' column. The 'Age' column contains string ranges like '65-74', but 'Age_Lower' and 'Age_Upper' are numeric columns specifically for filtering. Use 'Age_Lower >= 60' for 'over 60', 'Age_Upper < 50' for 'under 50', etc.

**FEW-SHOT EXAMPLES:**

**Example 1: Simple Aggregation**
Question: "What is the average Overall Rating?"
Thought: The question asks for the average of the 'Overall Rating' column. I can use the .mean() method on this column in the `df`.
Action: df['Overall Rating'].mean()
Observation: 4.5
Thought: The average Overall Rating is 4.5. This directly answers the question.
Final Answer: The average Overall Rating is 4.5.

**Example 2: Filtering and Counting**
Question: "How many reviews are there for Ozempic with a Satisfaction rating of 5?"
Thought: I need to filter the DataFrame `df` for rows where 'Brand Name' is 'Ozempic' and 'Satisfaction' is 5. Then I need to count how many such rows exist. The schema retrieval indicates 'Brand Name' and 'Satisfaction' are likely columns.
Action: len(df[(df['Brand Name'] == 'Ozempic') & (df['Satisfaction'] == 5)])
Observation: 120
Thought: There are 120 reviews for Ozempic with a Satisfaction rating of 5.
Final Answer: There are 120 reviews for Ozempic with a Satisfaction rating of 5.

**Example 3: Handling a potential error and finding unique values**
Question: "List unique conditions treated by Wegovy."
Thought: I need to find unique values in the 'Condition' column for rows where 'Brand Name' is 'Wegovy'. I should first check if 'Brand Name' and 'Condition' columns exist. The schema retrieval showed 'Brand Name' and 'Condition' as potential columns.
Action: df[df['Brand Name'] == 'Wegovy']['Condition'].unique()
Observation: array(['Obesity', 'Weight Loss', 'Prediabetes'], dtype=object)
Thought: The unique conditions treated by Wegovy are Obesity, Weight Loss, and Prediabetes.
Final Answer: Wegovy is used to treat Obesity, Weight Loss, and Prediabetes according to the data.

**Example 4: Query leading to no direct data, or requiring careful interpretation**
Question: "What is the most common age of patients who dislike Mounjaro?"
Thought: This requires multiple steps. First, filter for 'Mounjaro' and where 'Dislikes' is not null or empty (assuming 'Dislikes' column indicates a dislike if populated). Then find the mode of the 'Age' column for this subset. The schema retrieval suggests 'Brand Name', 'Dislikes', and 'Age' are available. I need to be careful about how 'Dislikes' is represented. For simplicity, I'll assume any non-empty 'Dislikes' string means a dislike.
Action: df[(df['Brand Name'] == 'Mounjaro') & (df['Dislikes'].fillna('').astype(str).str.strip() != '')]['Age'].mode()
Observation: 0    55
dtype: int64
Thought: The observation shows that 55 is the most common age. The dtype information is also provided.
Final Answer: The most common age of patients who reported dislikes for Mounjaro is 55.

**Example 5: Age Range Filtering**
Question: "What is the average satisfaction for Ozempic users over 60?"
Thought: I need to filter for rows where 'Brand Name' is 'Ozempic' and 'Age_Lower' >= 60, then calculate the mean of the 'Satisfaction' column.
Action: df[(df['Brand Name'] == 'Ozempic') & (df['Age_Lower'] >= 60)]['Satisfaction'].mean()
Observation: 4.2
Thought: The average satisfaction for Ozempic users aged 60 and above is 4.2.
Final Answer: The average satisfaction for Ozempic users aged 60 and above is 4.2.

**Example 6: Drug-specific Average Rating**
Question: "What is the average 'Effectiveness' for Zepbound?"
Thought: I need to filter the DataFrame for rows where 'Brand Name' is 'Zepbound' and then calculate the mean of the 'Effectiveness' column.
Action: df[df['Brand Name'] == 'Zepbound']['Effectiveness'].mean()
Observation: 4.1
Thought: The average Effectiveness rating for Zepbound is 4.1.
Final Answer: The average Effectiveness rating for Zepbound is 4.1.

**Example 7: Drug-specific Total Count**
Question: "What is the total 'Likes' count for Wegovy?"
Thought: I need to filter the DataFrame for rows where 'Brand Name' is 'Wegovy' and then sum up all the 'Likes' values.
Action: df[df['Brand Name'] == 'Wegovy']['Likes'].sum()
Observation: 1456
Thought: The total Likes count for Wegovy is 1456.
Final Answer: The total Likes count for Wegovy is 1456.

**Example 8: Multiple Rating Averages**
Question: "What is the average 'Satisfaction' for Mounjaro?"
Thought: I need to filter the DataFrame for rows where 'Brand Name' is 'Mounjaro' and then calculate the mean of the 'Satisfaction' column.
Action: df[df['Brand Name'] == 'Mounjaro']['Satisfaction'].mean()
Observation: 3.8
Thought: The average Satisfaction rating for Mounjaro is 3.8.
Final Answer: The average Satisfaction rating for Mounjaro is 3.8.

**Example 9: Age-based Statistics with Detailed Breakdown**
Question: "How satisfied are Ozempic users over 60? Give me specific numbers."
Thought: I need to filter for Ozempic users over 60 using Age_Lower >= 60, then calculate satisfaction statistics and distribution.
Action: ozempic_60plus = df[(df['Brand Name'] == 'Ozempic') & (df['Age_Lower'] >= 60)]; "Total: " + str(len(ozempic_60plus)) + " users, Average Satisfaction: " + str(round(ozempic_60plus['Satisfaction'].mean(), 2))
Observation: Total: 108 users, Average Satisfaction: 2.39
Thought: I have the basic statistics. Now I need to get the satisfaction distribution to show how many users gave each rating.
Action: ozempic_60plus['Satisfaction'].value_counts().sort_index()
Observation: 1    49
2    16
3     9
4    20
5    14
Name: Satisfaction, dtype: int64
Thought: Now I can provide detailed statistics showing both averages and the distribution of ratings.
Final Answer: Based on 108 Ozempic users over 60: Average satisfaction is 2.39. Distribution: 49 users (45.4%) gave 1-star, 16 users (14.8%) gave 2-star, 9 users (8.3%) gave 3-star, 20 users (18.5%) gave 4-star, and 14 users (13.0%) gave 5-star ratings.

**Example 10: Satisfaction Distribution with Percentages**
Question: "What's the satisfaction breakdown for Wegovy users?"
Thought: I need to get the value counts for Satisfaction ratings for Wegovy and calculate percentages.
Action: wegovy_data = df[df['Brand Name'] == 'Wegovy']; satisfaction_counts = wegovy_data['Satisfaction'].value_counts().sort_index(); "Total: " + str(len(wegovy_data)) + " users. 1-star: " + str(satisfaction_counts.get(1, 0)) + ", 5-star: " + str(satisfaction_counts.get(5, 0))
Observation: Total: 486 users. 1-star: 89, 5-star: 111
Thought: This provides a good start. Let me get the complete distribution with percentages.
Final Answer: Based on 486 Wegovy users: 89 users (18.3%) gave 1-star, 67 users (13.8%) gave 2-star, 94 users (19.3%) gave 3-star, 125 users (25.7%) gave 4-star, and 111 users (22.8%) gave 5-star satisfaction ratings.

**Example 11: Detailed Satisfaction Statistics with Age Filter**
Question: "How satisfied are Ozempic users over 60? I want the exact distribution."
Thought: I need to filter for Ozempic users over 60 and get their detailed satisfaction distribution with counts and percentages.
Action: ozempic_60plus = df[(df['Brand Name'] == 'Ozempic') & (df['Age_Lower'] >= 60)]; total = len(ozempic_60plus); satisfaction_dist = ozempic_60plus['Satisfaction'].value_counts().sort_index(); [(score, count, round(count/total*100, 1)) for score, count in satisfaction_dist.items()]
Observation: [(1, 49, 45.4), (2, 16, 14.8), (3, 9, 8.3), (4, 20, 18.5), (5, 14, 13.0)]
Thought: Now I have the complete satisfaction distribution with counts and percentages for 108 users over 60.
Final Answer: Among 108 Ozempic users over 60: 49 users (45.4%) gave 1-star, 16 users (14.8%) gave 2-star, 9 users (8.3%) gave 3-star, 20 users (18.5%) gave 4-star, and 14 users (13.0%) gave 5-star satisfaction ratings. Average satisfaction: 2.39/5.

Now, let's begin.
Question: {query}
Table Caption: {table_caption}
Retrieved Schema:
{schema_retrieval_result}
Retrieved Cells:
{cell_retrieval_result}

Begin!
"""
    }
    return prompts[prompt_type].format(**kwargs)

# --- TableRAG Retriever (Adapted from TableRAG agent.retriever) ---
class TableRAGRetriever:
    def __init__(self,
                 schema_db_path: str,
                 cell_db_path: str,
                 embed_model_name: str,
                 top_k: int = 5,
                 verbose: bool = False):
        self.schema_db_path = schema_db_path
        self.cell_db_path = cell_db_path
        self.embed_model_name = embed_model_name
        self.top_k = top_k
        self.verbose = verbose
        
        if not FAISS or not HuggingFaceEmbeddings:
            logger.critical("FAISS or HuggingFaceEmbeddings not installed. TableRAGRetriever cannot function.")
            raise ImportError("FAISS or HuggingFaceEmbeddings not installed.")

        try:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        except Exception as e:
            logger.critical(f"Failed to load HuggingFace embedding model {self.embed_model_name}: {e}", exc_info=True)
            raise

        self.schema_retriever = self._load_faiss_retriever(self.schema_db_path, "schema")
        self.cell_retriever = self._load_faiss_retriever(self.cell_db_path, "cell")

    def _load_faiss_retriever(self, db_path: str, db_type: str):
        if not os.path.exists(db_path) or not os.path.isdir(db_path): # Ensure it's a directory
            logger.error(f"{db_type} FAISS database directory not found at {db_path}.")
            raise FileNotFoundError(f"{db_type} database directory not found at {db_path}. Please ensure it's pre-built and the path is correct.")
        
        # Files expected by FAISS.load_local
        index_file = os.path.join(db_path, "index.faiss")
        pkl_file = os.path.join(db_path, "index.pkl")

        if not os.path.exists(index_file) or not os.path.exists(pkl_file):
            logger.error(f"Required FAISS files (index.faiss or index.pkl) not found in directory {db_path} for {db_type} DB.")
            raise FileNotFoundError(f"FAISS index files not found in {db_path}.")

        if self.verbose:
            logger.info(f"Loading {db_type} FAISS database from directory {db_path}")
        try:
            # FAISS.load_local expects the folder path, not individual file paths.
            db = FAISS.load_local(db_path, self.embedder, allow_dangerous_deserialization=True)
            return db.as_retriever(search_kwargs={'k': self.top_k})
        except Exception as e:
            logger.error(f"Error loading {db_type} FAISS DB from {db_path}: {e}", exc_info=True)
            raise

    def retrieve_schema(self, query: str) -> List[str]:
        if not self.schema_retriever:
            logger.warning("Schema retriever not available for retrieve_schema.")
            return []
        try:
            results = self.schema_retriever.invoke(query)
            observations = [doc.page_content for doc in results]
            return observations
        except Exception as e:
            logger.error(f"Error during schema retrieval for query '{query}': {e}", exc_info=True)
            return []


    def retrieve_cell(self, query: str) -> List[str]:
        if not self.cell_retriever:
            logger.warning("Cell retriever not available for retrieve_cell.")
            return []
        try:
            results = self.cell_retriever.invoke(query)
            observations = [doc.page_content for doc in results]
            return observations
        except Exception as e:
            logger.error(f"Error during cell retrieval for query '{query}': {e}", exc_info=True)
            return []

# --- TableRAG Agent (Adapted from TableRAG agent.rag_agent) ---
class TableRAGAgent:
    def __init__(self,
                 llm_wrapper: LLMWrapper,
                 retriever: TableRAGRetriever,
                 verbose: bool = False,
                 max_solver_depth: int = 7):
        self.llm = llm_wrapper
        self.retriever = retriever
        self.verbose = verbose
        self.max_solver_depth = max_solver_depth
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _llm_query_with_tracking(self, prompt: str, **kwargs) -> str:
        # Ensure system message is passed if provided in kwargs
        system_message = kwargs.pop('system_message', None)
        self.total_input_tokens += self.llm.get_token_count(prompt + (system_message or ""))
        response = self.llm.query(prompt, system_message=system_message, **kwargs)
        self.total_output_tokens += self.llm.get_token_count(response)
        return response

    def _is_terminal(self, text: str) -> bool:
        return 'final answer:' in text.lower()

    def _solver_loop(self, df: pd.DataFrame, initial_prompt: str) -> Tuple[str, int, str]:
        if self.verbose:
            logger.info("\n--- TableRAG Solver Loop ---")
            logger.info(f"Initial Prompt (first 500 chars):\n{initial_prompt[:500]}...\n")

        current_solution_trail = ""
        num_iterations = 0

        for i in range(self.max_solver_depth):
            num_iterations = i + 1
            
            # Append "Thought: " to the current trail to prompt the LLM for the next thought
            # The LLM is expected to continue from this "Thought: "
            prompt_for_llm = initial_prompt + current_solution_trail + "Thought: "
            
            # LLM generates the thought and the action part
            llm_response_segment = self._llm_query_with_tracking(
                prompt_for_llm,
                stop_sequences=["Observation:"], # LLM should stop before generating "Observation:"
                max_tokens=512 # Increased max_tokens for potentially longer thought/action pairs
            ).strip()
            
            # Append the LLM's generated thought and action to the trail
            current_solution_trail += "Thought: " + llm_response_segment # Prepend "Thought: " as it was part of the prompt structure
            
            if self.verbose:
                logger.info(f"LLM Response Segment ({i+1}):\n{llm_response_segment}")

            # Check if the LLM's response (now part of the trail) contains the final answer
            if self._is_terminal(current_solution_trail):
                if self.verbose: logger.info("Terminal condition (Final Answer) met in current solution trail.")
                break

            # Extract action from the llm_response_segment
            # The llm_response_segment is expected to be "actual thought text \nAction: actual_action_code"
            if 'Action:' not in llm_response_segment:
                if self.verbose:
                    logger.warning("No 'Action:' found in LLM response segment. Attempting to conclude or breaking solver loop.")
                break # Break if no action is provided and it's not a final answer
            
            try:
                action_code_part = llm_response_segment.split('Action:', 1)[-1]
                action_code = parse_code_from_string(action_code_part)
                
                if not action_code.strip():
                    if self.verbose: logger.warning(f"Extracted empty action string from: '{action_code_part}'. Skipping execution.")
                    observation = "Error: Empty action provided."
                else:
                    if self.verbose:
                        logger.info(f"Extracted Action ({i+1}): {action_code}")
                    
                    # Execute the action
                    # Pass a copy of df to python_repl_ast to ensure original df in this loop is not modified
                    # unless explicitly done so by an action like `df = ...`
                    observation, _ = python_repl_ast(action_code, custom_locals={'df': df.copy()})
                    
                    # Truncate long observations to prevent excessive token usage in next prompt
                    max_obs_len = 1000 # Increased max observation length
                    if len(observation) > max_obs_len:
                        observation = observation[:max_obs_len] + f"... (truncated, original length {len(observation)})"
            
            except Exception as e:
                logger.error(f"Error parsing or executing action in solver loop: {e}", exc_info=True)
                observation = f"Error during action processing: {type(e).__name__}: {str(e)}"
            
            # Append the observation to the trail
            current_solution_trail += f"\nObservation: {observation}\n" # Ensure newlines for readability
            if self.verbose:
                logger.info(f"Observation ({i+1}):\n{observation}")
            
            # Safety break if trail gets too long (e.g. runaway LLM)
            if self.llm.get_token_count(initial_prompt + current_solution_trail) > self.llm.context_limit * 0.8:
                logger.warning("Solver trail approaching context limit. Breaking loop.")
                current_solution_trail += "\nThought: The conversation history is too long. I must provide a final answer based on current information or state that I cannot proceed further."
                break


        # Extract final answer
        final_answer_match = re.search(r"Final Answer:\s*(.*)", current_solution_trail, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
        else:
            # If no "Final Answer:" keyword, try to take the last thought as answer, or the whole trail if desperate
            if self.verbose: logger.warning("No 'Final Answer:' found in solver trail. Using fallback.")
            last_thought_match = re.findall(r"Thought:\s*(.*?)(?=\nAction:|\nObservation:|\Z)", current_solution_trail, re.DOTALL | re.IGNORECASE)
            if last_thought_match:
                answer = last_thought_match[-1].strip() # Last thought
            elif llm_response_segment: # Fallback to the last raw segment from LLM if no clear thought
                answer = llm_response_segment
            else: # Absolute fallback
                answer = "Could not determine a final answer from the solver process."


        return answer, num_iterations, initial_prompt + current_solution_trail

    def run(self, user_query: str, table_df: pd.DataFrame, table_id: str, table_caption: str) -> Dict:
        self.total_input_tokens = 0 # Reset for the session/run
        self.total_output_tokens = 0

        if self.verbose:
            logger.info(f"\n--- Running TableRAG for query: '{user_query}' ---")
            logger.info(f"Table ID: {table_id}, Caption: {table_caption}")

        # 1. Extract Column Names (Schema Retrieval Query Expansion)
        col_extract_prompt = get_table_rag_prompt(
            'extract_column_prompt', table_caption=table_caption, query=user_query
        )
        col_query_response = self._llm_query_with_tracking(col_extract_prompt, max_tokens=256) # Increased tokens
        try:
            column_queries = read_json_from_llm(col_query_response)
            if not isinstance(column_queries, list): column_queries = [str(column_queries)] # Ensure list
            column_queries = [str(q) for q in column_queries if q] # Ensure all are strings and not empty
        except Exception as e:
            if self.verbose: logger.warning(f"Error parsing column queries from LLM response: {e}. Raw response: '{col_query_response}'. Using user query as fallback.")
            column_queries = [user_query]
        
        retrieved_schema_parts = set()
        if column_queries: # Only retrieve if queries were successfully generated
            for cq in column_queries:
                retrieved_schema_parts.update(self.retriever.retrieve_schema(str(cq)))
        
        schema_retrieval_text = "Schema Retrieval Queries: " + ", ".join(map(str,column_queries)) + "\n"
        schema_retrieval_text += "Schema Retrieval Results:\n" + ("\n".join(sorted(list(retrieved_schema_parts))) if retrieved_schema_parts else "No schema parts retrieved.")
        if self.verbose: logger.info(f"\n{schema_retrieval_text}")

        # 2. Extract Cell Keywords (Cell Retrieval Query Expansion)
        cell_extract_prompt = get_table_rag_prompt(
            'extract_cell_prompt', table_caption=table_caption, query=user_query
        )
        cell_query_response = self._llm_query_with_tracking(cell_extract_prompt, max_tokens=256) # Increased tokens
        try:
            cell_queries = read_json_from_llm(cell_query_response)
            if not isinstance(cell_queries, list): cell_queries = [str(cell_queries)] # Ensure list
            cell_queries = [str(c) for c in cell_queries if c and not is_numeric(c)] # Ensure strings, not empty, and filter numerics
        except Exception as e:
            if self.verbose: logger.warning(f"Error parsing cell queries from LLM response: {e}. Raw response: '{cell_query_response}'. Using user query as fallback.")
            cell_queries = [user_query]

        retrieved_cell_parts = set()
        if cell_queries: # Only retrieve if queries were successfully generated
            for cq_cell in cell_queries:
                retrieved_cell_parts.update(self.retriever.retrieve_cell(str(cq_cell)))

        cell_retrieval_text = "Cell Retrieval Queries: " + ", ".join(map(str,cell_queries)) + "\n"
        cell_retrieval_text += "Cell Retrieval Results:\n" + ("\n".join(sorted(list(retrieved_cell_parts))) if retrieved_cell_parts else "No cell parts retrieved.")
        if self.verbose: logger.info(f"\n{cell_retrieval_text}")

        # 3. Solve with Program-Aided LLM
        solver_prompt = get_table_rag_prompt(
            'solve_table_prompt',
            table_caption=table_caption,
            query=user_query,
            schema_retrieval_result=schema_retrieval_text,
            cell_retrieval_result=cell_retrieval_text
        )
        
        init_prompt_token_count = self.llm.get_token_count(solver_prompt)
        # This initial prompt token count is part of the first call to _llm_query_with_tracking inside _solver_loop,
        # so it's already accounted for if self.total_input_tokens is managed correctly there.
        # Let's ensure solver_loop's first call correctly uses this full initial_prompt.

        answer, n_iter, solution_trail = self._solver_loop(table_df.copy(), solver_prompt) # Pass copy of df

        return {
            'query': user_query,
            'answer': answer,
            'solution_trail': solution_trail,
            'schema_retrieval_text': schema_retrieval_text,
            'cell_retrieval_text': cell_retrieval_text,
            'iterations': n_iter,
            'initial_solver_prompt_tokens': init_prompt_token_count, # Tokens for the base prompt before ReAct loop
            'total_input_tokens_session': self.total_input_tokens, # Total tokens for this .run() call
            'total_output_tokens_session': self.total_output_tokens,
        }

# --- GraphRAG Module ---
class GraphRAGModule:
    def __init__(self, llm_wrapper: LLMWrapper, uri: str, user: str, password: str, verbose: bool = False):
        self.llm = llm_wrapper
        self.uri = uri
        self.user = user
        self.password = password
        self.verbose = verbose
        self._driver = None
        if not GraphDatabase:
            logger.critical("Neo4j driver not available. GraphRAGModule cannot function.")
            raise ImportError("Neo4j driver not available for GraphRAGModule.")
        self._connect()
        self.schema_representation = self._get_precise_schema_representation()
        if self.verbose and self._driver:
            logger.info(f"GraphRAG Schema:\n{self.schema_representation}")


    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity() # Check if connection is valid
            if self.verbose: logger.info("Successfully connected to Neo4j.")
        except ServiceUnavailable as e:
            logger.error(f"Neo4j connection failed (ServiceUnavailable): {e}. GraphRAG will be unavailable.", exc_info=True)
            self._driver = None
        except Exception as e: # Catch other potential auth errors or general issues
            logger.error(f"Neo4j general connection error: {e}. GraphRAG will be unavailable.", exc_info=True)
            self._driver = None

    def close(self):
        if self._driver is not None:
            self._driver.close()
            if self.verbose: logger.info("Neo4j connection closed.")

    def _get_precise_schema_representation(self) -> str:
        """
        Generates a precise schema description based on the provided loader script and user's data structure.
        """
        # This schema is derived from the user's description of standardized_info and standardized_relations
        # Node labels: Drug, Condition, SideEffect (initially user mentioned 'drug', 'condition', 'side_effects')
        # Relationship types: TREATS, CAUSES (initially user mentioned 'treats', 'causes')
        # Properties are aligned with the example standardized_info.
        return """
# Neo4j Graph Schema for WebMD Drug Reviews

## Node Labels and Properties:

1.  **Drug**: Represents a specific brand-name medication.
    * `name`: String (UNIQUE - This is the Brand Name, e.g., 'Ozempic'. **Use this for matching Drugs.**)
    * `generic_name`: String (e.g., 'Semaglutide'. Corresponds to 'Drug Name' column from CSV)
    * `dosage`: String (Can be null)
    * `dosage_form`: String (e.g., 'subcutaneous injection'. Can be null)
    * `duration`: String (e.g., '1 to less than 2 years'. Can be null)
    * `continued_use`: String ('yes' or 'no'. Can be null)
    * `alternative_drug_considered`: String (Can be null)

2.  **Condition**: Represents a medical condition the drugs might treat.
    * `name`: String (UNIQUE - e.g., 'Type 2 Diabetes Mellitus'. **Use this for matching Conditions.**)
    * `severity`: String (Can be null)

3.  **SideEffect**: Represents a side effect reported for a drug.
    * `name`: String (UNIQUE - e.g., 'dizziness'. **Use this for matching SideEffects.**)
    * `severity`: String (Can be null)
    * `associated_drug`: String (The brand name of the drug this side effect was reported with. Can be null)

## Relationship Types and Properties:

1.  **(Drug)-[:TREATS]->(Condition)**: Indicates a drug is used to treat a condition.
    * `approval`: String ('yes' or 'no'. Can be null)
    * `off_label`: String ('yes' or 'no'. Can be null)

2.  **(Drug)-[:CAUSES]->(SideEffect)**: Indicates a drug is reported to cause a side effect.
    * `severity`: String (Can be null, inherited from SideEffect node or specific to relation)
    * `dosage`: String (The dosage of the drug when the side effect occurred. Can be null)

**IMPORTANT NOTES FOR CYPHER GENERATION:**
* Node labels (`Drug`, `Condition`, `SideEffect`) and Relationship types (`TREATS`, `CAUSES`) are case-sensitive as defined here (PascalCase for Nodes, UPPERCASE for Relationships).
* When matching nodes, primarily use their `name` property as it's marked UNIQUE. E.g., `MATCH (d:Drug {name: 'Ozempic'})`.
* String property values in Cypher queries should be enclosed in single or double quotes.
* Pay close attention to property names and their potential for being null when writing WHERE clauses.
        """

    def _generate_cypher_query(self, user_query: str, failed_attempt: Optional[Dict] = None) -> Optional[str]:
        few_shot_examples = """
# Few-Shot Examples (Question -> Cypher Query):

Q: What are the side effects of Ozempic?
Cypher: MATCH (d:Drug {name: 'Ozempic'})-[:CAUSES]->(s:SideEffect) RETURN s.name AS SideEffect, s.severity AS Severity LIMIT 10

Q: Which drugs treat Type 2 Diabetes Mellitus?
Cypher: MATCH (d:Drug)-[:TREATS]->(c:Condition {name: 'Type 2 Diabetes Mellitus'}) RETURN d.name AS Drug, d.generic_name AS GenericName LIMIT 10

Q: Find severe side effects caused by Mounjaro.
Cypher: MATCH (d:Drug {name: 'Mounjaro'})-[r:CAUSES]->(s:SideEffect) WHERE r.severity = 'severe' OR s.severity = 'severe' RETURN s.name AS SevereSideEffect LIMIT 10

Q: Is Wegovy approved for treating Obesity?
Cypher: MATCH (d:Drug {name: 'Wegovy'})-[r:TREATS]->(c:Condition {name: 'Obesity'}) RETURN r.approval AS ApprovalStatus, r.off_label as OffLabelUse LIMIT 10
        
Q: What is the generic name for Zepbound?
Cypher: MATCH (d:Drug {name: 'Zepbound'}) RETURN d.generic_name AS GenericName LIMIT 1

Q: I heard Mounjaro can cause stomach issues. What are the side effects?
Cypher: MATCH (d:Drug {name: 'Mounjaro'})-[:CAUSES]->(s:SideEffect) RETURN s.name AS SideEffect, s.severity AS Severity LIMIT 15

Q: I heard Mounjaro can cause stomach issues. Is that true? What kind of problems do people usually run into?
Cypher: MATCH (d:Drug {name: 'Mounjaro'})-[:CAUSES]->(s:SideEffect) RETURN s.name AS SideEffect, s.severity AS Severity LIMIT 15

Q: How many people report each side effect for Ozempic?
Cypher: MATCH (d:Drug {name: 'Ozempic'})-[:CAUSES]->(s:SideEffect) RETURN s.name AS SideEffect, count(*) AS ReportCount, collect(DISTINCT s.severity) AS Severities ORDER BY ReportCount DESC LIMIT 10

Q: What conditions does Saxenda treat and what are its common side effects?
Cypher: MATCH (drug:Drug {name: 'Saxenda'})
OPTIONAL MATCH (drug)-[:TREATS]->(condition:Condition)
OPTIONAL MATCH (drug)-[:CAUSES]->(sideEffect:SideEffect)
RETURN drug.name AS Drug, collect(DISTINCT condition.name) AS ConditionsTreated, collect(DISTINCT sideEffect.name)[..5] AS CommonSideEffects LIMIT 1
"""

        system_message_base = f"""You are a Neo4j expert. Convert natural language questions into Cypher queries based ONLY on the provided schema.
Follow these rules strictly:
1. Use ONLY the node labels, properties, and relationship types defined in the schema. Adhere to case-sensitivity (e.g., `Drug`, `SideEffect`, `TREATS`, `CAUSES`).
2. Match nodes primarily using their `name` property (which is unique for each label type).
3. Pay attention to property names and their potential for being null. Use `exists()` or `IS NOT NULL` for checking property existence if needed.
4. Always include a `LIMIT` clause (e.g., `LIMIT 10` for lists, `LIMIT 1` for single specific facts) to prevent overly large results unless the query inherently limits results.
5. Return ONLY the Cypher query. No explanations, no markdown, no other text.
6. If the question cannot be answered with a Cypher query from this schema, return the exact string "Error: Cannot form Cypher query from schema."
7. **CRITICAL**: For questions about general symptoms or side effects (like 'stomach issues', 'gastrointestinal problems', 'stomach pain', 'nausea', etc.), do NOT add WHERE clauses to filter side effects by specific keywords. Instead, return ALL side effects and let the final LLM determine which ones are relevant. Side effects may use medical terminology that won't match simple keyword searches.

Database Schema:
{self.schema_representation}

{few_shot_examples}
"""

        prompt = f"Natural language question: {user_query}\nCypher query:"

        if failed_attempt:
            system_message = f"{system_message_base}\n\nIMPORTANT: The previous Cypher query attempt failed. \nPrevious Query: {failed_attempt['query']} \nError: {failed_attempt['error']}\nPlease analyze the error and schema, then provide a corrected Cypher query. If correction is not possible, return 'Error: Cannot form Cypher query from schema.'"
        else:
            system_message = system_message_base

        cypher_query = self.llm.query(prompt, system_message=system_message, temperature=0.05, max_tokens=400) # Low temp for precision

        if "Error: Cannot form Cypher query from schema." in cypher_query or not cypher_query.strip().upper().startswith("MATCH"):
            if self.verbose: logger.warning(f"LLM could not generate valid Cypher for: '{user_query}'. LLM Response: {cypher_query}")
            return None
            
        cypher_query = re.sub(r"```(?:cypher\s*)?|\s*```", "", cypher_query, flags=re.IGNORECASE).strip()
        if cypher_query.lower().startswith("cypher:"):
            cypher_query = cypher_query[7:].strip()
            
        return cypher_query

    def _execute_cypher(self, cypher_query: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        if not self._driver:
            logger.error("Neo4j connection not available for Cypher execution.")
            return None, "Neo4j connection not available."
        
        try:
            with self._driver.session(database="neo4j") as session: # Explicitly specify database if not default
                if self.verbose: logger.info(f"Executing Cypher: {cypher_query}")
                results = session.run(cypher_query)
                records = [record.data() for record in results]
                return records, None
        except CypherSyntaxError as e:
            error_msg = f"Cypher Syntax Error: {e.message} (Code: {e.code})"
            if self.verbose: logger.warning(f"{error_msg} for query: {cypher_query}")
            return None, error_msg
        except ServiceUnavailable as e: # Handle cases where DB might go down after initial connect
            logger.error(f"Neo4j ServiceUnavailable during Cypher execution: {e}", exc_info=True)
            return None, f"Neo4j Service Unavailable: {str(e)}"
        except Exception as e:
            error_msg = f"Execution Error: {type(e).__name__} - {str(e)}"
            if self.verbose: logger.error(f"Error executing Cypher query '{cypher_query}': {e}", exc_info=True)
            return None, error_msg

    def query_graph(self, user_query: str, max_retries: int = 1) -> str:
        if not self._driver:
            return "GraphRAG Error: Neo4j connection is not established."

        attempt = 0
        failed_info = None
        cypher_query = "" # Initialize cypher_query

        while attempt <= max_retries:
            cypher_query = self._generate_cypher_query(user_query, failed_attempt=failed_info)
            if not cypher_query:
                return "GraphRAG Error: Could not generate Cypher query from the question based on the schema."

            if self.verbose:
                logger.info(f"\n--- GraphRAG Query (Attempt {attempt+1}/{max_retries+1}) ---")
                logger.info(f"User Query: {user_query}")
                logger.info(f"Generated Cypher: {cypher_query}")

            records, error_message = self._execute_cypher(cypher_query)

            if error_message is None: # Success if error_message is None
                if not records: # Successfully executed, but no data returned
                    return f"GraphRAG: No specific results found in the graph for Cypher query: {cypher_query}"
                
                max_graph_res_len = 2500 # Increased length for potentially richer graph context
                try:
                    formatted_results = json.dumps(records, indent=2, default=str) # Use default=str for non-serializable types
                except TypeError as te:
                    logger.warning(f"TypeError during JSON serialization of graph results: {te}. Falling back to simple string conversion.")
                    formatted_results = str(records)

                if len(formatted_results) > max_graph_res_len:
                    formatted_results = formatted_results[:max_graph_res_len] + f"\n... (graph results truncated, original size {len(formatted_results)} chars)"
                return f"GraphRAG Context (from Cypher: {cypher_query}):\n{formatted_results}"
            else:
                # Failure, prepare for retry if attempts left
                failed_info = {"query": cypher_query, "error": error_message}
                attempt += 1
                if self.verbose: logger.warning(f"GraphRAG Cypher query failed. Error: {error_message}. Retrying ({attempt}/{max_retries+1})...")

        # If all attempts failed
        return f"GraphRAG Error: Failed to execute Cypher query after {max_retries+1} attempts. Last query: '{failed_info['query'] if failed_info else 'N/A'}'. Last error: {failed_info['error'] if failed_info else 'N/A'}"
    
# --- Main ChatBot Application ---
class MedicalChatBot:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        logger.info("Initializing MedicalChatBot...")

        if not OPENAI_API_KEY: # Check if it's empty or None
            logger.critical("ERROR: OPENAI_API_KEY is not set. Please set it in the script or as an environment variable.")
            raise ValueError("OPENAI_API_KEY not configured.")

        self.llm = LLMWrapper(model_name=LLM_MODEL_NAME, api_key=OPENAI_API_KEY)
        logger.info(f"LLM Wrapper initialized with model: {LLM_MODEL_NAME}")

        try:
            self.table_rag_retriever = TableRAGRetriever(
                schema_db_path=SCHEMA_DB_PATH,
                cell_db_path=CELL_DB_PATH,
                embed_model_name=TABLE_RAG_EMBED_MODEL_NAME,
                top_k=TABLE_RAG_TOP_K,
                verbose=self.verbose
            )
            self.table_rag_agent = TableRAGAgent(
                llm_wrapper=self.llm,
                retriever=self.table_rag_retriever,
                verbose=self.verbose
            )
            logger.info("TableRAG Agent initialized.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR initializing TableRAG components: {e}", exc_info=True)
            self.table_rag_agent = None # Ensure it's None if init fails
            
        try:
            self.graph_rag_module = GraphRAGModule(
                llm_wrapper=self.llm,
                uri=NEO4J_URI,
                user=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                verbose=self.verbose
            )
            logger.info("GraphRAG Module initialized.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR initializing GraphRAG Module: {e}", exc_info=True)
            self.graph_rag_module = None # Ensure it's None if init fails

        self.main_table_df = None
        self.table_rag_df = None  # 专门为TableRAG优化的DataFrame，不包含结构化列
        if self.table_rag_agent: # Only load if TableRAG agent is available
            try:
                logger.info(f"Loading main table data from {TABLE_CSV_PATH} for TableRAG solver...")
                self.main_table_df = pd.read_csv(TABLE_CSV_PATH)
                # Basic cleaning: remove unnamed columns that might appear from CSV export/import
                self.main_table_df = self.main_table_df.loc[:, ~self.main_table_df.columns.str.contains('^Unnamed')]
                # --- BEGIN NEW PREPROCESSING CODE ---
                logger.info("Preprocessing 'Age' column...")
                ACTUAL_AGE_COLUMN_NAME = 'Age'  # <--- !!! IMPORTANT: Verify this column name in your CSV !!!
                if ACTUAL_AGE_COLUMN_NAME in self.main_table_df.columns:
                    try:
                        # Apply the parsing function
                        age_bounds = self.main_table_df[ACTUAL_AGE_COLUMN_NAME].apply(parse_age_range)
                        # Create new columns from the parsed results
                        self.main_table_df[['Age_Lower', 'Age_Upper']] = pd.DataFrame(age_bounds.tolist(), index=self.main_table_df.index)
                        # Convert new columns to a numeric type, coercing errors to NaN
                        self.main_table_df['Age_Lower'] = pd.to_numeric(self.main_table_df['Age_Lower'], errors='coerce')
                        self.main_table_df['Age_Upper'] = pd.to_numeric(self.main_table_df['Age_Upper'], errors='coerce')
                        logger.info("Successfully added 'Age_Lower' and 'Age_Upper' columns.")
                        if self.verbose:
                            logger.info(f"Age columns head:\n{self.main_table_df[[ACTUAL_AGE_COLUMN_NAME, 'Age_Lower', 'Age_Upper']].head()}")
                    except Exception as e:
                        logger.error(f"Error during 'Age' preprocessing: {e}", exc_info=True)
                else:
                    logger.warning(f"Column '{ACTUAL_AGE_COLUMN_NAME}' not found in DataFrame. Skipping age preprocessing.")
                # --- END NEW PREPROCESSING CODE ---
                # Now, infer dtypes (it might handle the new numeric columns well, or they are already set)
                self.main_table_df = infer_dtype(self.main_table_df)
                
                # --- CREATE OPTIMIZED TABLERAG DATAFRAME ---
                logger.info("Creating optimized TableRAG DataFrame by removing structured columns...")
                # 定义要移除的结构化列
                structured_columns_to_remove = ['standardized_info', 'standardized_relations']
                # 创建TableRAG专用的DataFrame，移除结构化列
                tablerag_columns = [col for col in self.main_table_df.columns if col not in structured_columns_to_remove]
                self.table_rag_df = self.main_table_df[tablerag_columns].copy()
                
                logger.info(f"Loaded main table: {self.main_table_df.shape[0]} rows, {self.main_table_df.shape[1]} columns.")
                logger.info(f"Created TableRAG DataFrame: {self.table_rag_df.shape[0]} rows, {self.table_rag_df.shape[1]} columns.")
                logger.info(f"Removed columns for TableRAG: {structured_columns_to_remove}")
                if self.verbose: 
                    logger.info(f"Main table dtypes after infer_dtype:\n{self.main_table_df.dtypes}")
                    logger.info(f"TableRAG DataFrame columns: {list(self.table_rag_df.columns)}")
                # --- END TABLERAG OPTIMIZATION ---
            except FileNotFoundError:
                logger.error(f"ERROR: Main table CSV not found at {TABLE_CSV_PATH}. TableRAG solver will not function correctly.")
                self.main_table_df = pd.DataFrame()
                self.table_rag_df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error loading or processing main table CSV '{TABLE_CSV_PATH}': {e}", exc_info=True)
                self.main_table_df = pd.DataFrame()
                self.table_rag_df = pd.DataFrame()
        else:
            logger.warning("TableRAG agent not initialized, so main table CSV will not be loaded.")

    def route_query(self, user_query: str) -> Tuple[bool, bool]:
        """
        Determines whether to use TableRAG, GraphRAG, or both based on the query.
        (This can be kept simple as context gathering now happens per sub-query if needed)
        Returns: (use_table_rag, use_graph_rag)
        """
        use_table = self.table_rag_agent is not None and self.table_rag_df is not None and not self.table_rag_df.empty
        use_graph = self.graph_rag_module is not None and self.graph_rag_module._driver is not None
        
        query_lower = user_query.lower()
        
        # Heuristics - More advanced LLM routing could be used.
        # For now, let's assume GraphRAG is good for relationships/side effects,
        # and TableRAG for stats/user feedback. We'll try both if available for sub-queries.
        graph_keywords = ["side effect", "treats", "causes", "generic name", "approved for"]
        table_keywords = ["average", "rating", "satisfaction", "how many", "compare", "users say", "like", "dislike"]

        route_to_graph = use_graph and any(keyword in query_lower for keyword in graph_keywords)
        route_to_table = use_table and any(keyword in query_lower for keyword in table_keywords)

        # If no strong signal, or for general queries, try both.
        # Since we run per-sub-query, it's often safer to try both if available.
        final_route_table = use_table
        final_route_graph = use_graph

        # Simple override: if *only* graph keywords, maybe prioritize graph? (Can refine this)
        # For now, let's keep it simple: if available, try.
        
        if self.verbose: logger.info(f"Routing decision for '{query_lower[:50]}...': use_table_rag={final_route_table}, use_graph_rag={final_route_graph}")
        return final_route_table, final_route_graph

    def _is_summary_query(self, user_query: str) -> bool:
        """
        Determines if the query is asking for pros/cons or summary-style information about a drug.
        Returns True if it matches patterns for pros/cons/summary questions.
        """
        query_lower = user_query.lower()
        
        # Keywords that indicate pros/cons or summary queries
        pros_cons_keywords = [
            "best thing", "good thing", "positive", "pros", "advantages", "benefits",
            "worst thing", "bad thing", "negative", "cons", "disadvantages", "downsides",
            "biggest downside", "biggest problem", "main issue", "major side effect",
            "people say", "users say", "reviews say", "what do people think",
            "overall opinion", "general feedback", "user experience",
            "highlight", "standout", "notable", "remarkable",
            "summary", "overview", "tell me about"
        ]
        
        # Drug names that should be present in summary queries
        drug_names = ["wegovy", "ozempic", "rybelsus", "zepbound", "mounjaro", "victoza", "saxenda"]
        
        # Check if query contains both pros/cons keywords and drug names
        has_summary_keyword = any(keyword in query_lower for keyword in pros_cons_keywords)
        has_drug_name = any(drug in query_lower for drug in drug_names)
        
        if self.verbose and has_summary_keyword and has_drug_name:
            logger.info(f"Summary/pros-cons query detected: '{user_query}'")
        
        return has_summary_keyword and has_drug_name

    def _extract_drug_name(self, user_query: str) -> Optional[str]:
        """
        Extracts the drug name from a user query.
        Returns the properly capitalized drug name if found, None otherwise.
        """
        query_lower = user_query.lower()
        
        # Drug name mapping (lowercase -> proper case)
        drug_mapping = {
            "wegovy": "Wegovy",
            "ozempic": "Ozempic", 
            "rybelsus": "Rybelsus",
            "zepbound": "Zepbound",
            "mounjaro": "Mounjaro",
            "victoza": "Victoza",
            "saxenda": "Saxenda"
        }
        
        for drug_lower, drug_proper in drug_mapping.items():
            if drug_lower in query_lower:
                if self.verbose:
                    logger.info(f"Extracted drug name: {drug_proper} from query: '{user_query}'")
                return drug_proper
        
        if self.verbose:
            logger.warning(f"Could not extract drug name from query: '{user_query}'")
        return None

    def _get_pro_con_context(self, drug_name: str) -> Dict[str, Any]:
        """
        Fetches structured information representing pros and cons for a drug.
        Returns a dictionary of findings.
        """
        findings = {
            "drug_name": drug_name,
            "ratings": None,
            "treated_conditions": None,
            "side_effects": None,
            "likes_count": None,
            "error_messages": []
        }

        # --- Fetching Cons (Side Effects via GraphRAG) ---
        if self.graph_rag_module and self.graph_rag_module._driver:
            try:
                side_effects_query = f"What are the side effects of {drug_name}?"
                graph_context_side_effects = self.graph_rag_module.query_graph(side_effects_query)
                if "Error" not in graph_context_side_effects and "No specific results" not in graph_context_side_effects:
                    findings["side_effects"] = graph_context_side_effects
                elif "No specific results" in graph_context_side_effects:
                    findings["side_effects"] = f"No specific side effects found for {drug_name} in the knowledge graph."
                else:
                    findings["error_messages"].append(f"GraphRAG issue for side effects: {graph_context_side_effects}")
            except Exception as e:
                logger.error(f"Error querying GraphRAG for side effects of {drug_name}: {e}")
                findings["error_messages"].append(f"Could not query side effects for {drug_name} from graph.")
        else:
            findings["error_messages"].append("GraphRAG module is unavailable for side effect lookup.")

        # --- Fetching Pros (Ratings & Likes via TableRAG) ---
        if self.table_rag_agent and self.table_rag_df is not None and not self.table_rag_df.empty:
            # Query for ratings
            rating_queries = {
                "Overall Rating": f"What is the average 'Overall Rating' for {drug_name}?",
                "Effectiveness": f"What is the average 'Effectiveness' for {drug_name}?", 
                "Satisfaction": f"What is the average 'Satisfaction' for {drug_name}?",
                "Likes Count": f"What is the total 'Likes' count for {drug_name}?"
            }
            ratings_summary = []
            for aspect, query in rating_queries.items():
                try:
                    if self.verbose:
                        logger.info(f"TableRAG query for {aspect}: {query}")
                    result = self.table_rag_agent.run(query, self.table_rag_df.copy(), TABLE_ID, TABLE_CAPTION)
                    answer = result.get('answer', 'Not available')
                    if self.verbose:
                        logger.info(f"TableRAG answer for {aspect}: {answer}")
                    if "Error" not in answer and "Could not determine" not in answer and "No specific answer" not in answer:
                        ratings_summary.append(f"{aspect}: {answer}")
                    else:
                        if self.verbose:
                            logger.warning(f"TableRAG could not get {aspect} for {drug_name}: {answer}")
                        # Add debug information about the result
                        logger.info(f"Full TableRAG result for {aspect}: {result}")
                except Exception as e:
                    logger.error(f"TableRAG error for {aspect} of {drug_name}: {e}")
                    findings["error_messages"].append(f"Could not get {aspect} for {drug_name} from table.")
            
            if ratings_summary:
                findings["ratings"] = "\n".join(ratings_summary)
            else:
                # Log additional debug info if no ratings were found
                logger.warning(f"No ratings summary generated for {drug_name}. Check TableRAG queries.")
        else:
            findings["error_messages"].append("TableRAG module is unavailable for rating lookup.")

        # --- Fetching Pros (Treated Conditions via GraphRAG) ---
        if self.graph_rag_module and self.graph_rag_module._driver:
            try:
                conditions_query = f"What conditions does {drug_name} treat?"
                graph_context_conditions = self.graph_rag_module.query_graph(conditions_query)
                if "Error" not in graph_context_conditions and "No specific results" not in graph_context_conditions:
                    findings["treated_conditions"] = graph_context_conditions
                elif "No specific results" in graph_context_conditions:
                    findings["treated_conditions"] = f"No specific conditions treated by {drug_name} found in the knowledge graph."
                else:
                    findings["error_messages"].append(f"GraphRAG issue for treated conditions: {graph_context_conditions}")
            except Exception as e:
                logger.error(f"Error querying GraphRAG for conditions treated by {drug_name}: {e}")
                findings["error_messages"].append(f"Could not query conditions treated by {drug_name} from graph.")

        return findings

    def _decompose_query(self, user_query: str) -> Dict:
        """
        Uses LLM to detect comparison questions, extract entities, and generate sub-questions
        in a structured format.
        """
        system_message = """You are an assistant. Your task is to analyze a user's question about medical drugs.
1. Determine if it is a comparison question (comparing two or more drugs from the list: Wegovy, Ozempic, Rybelsus, Zepbound, Mounjaro, Victoza, Saxenda).
2. If it is, identify the drugs being compared.
3. Generate *separate*, *non-comparative* sub-questions for *each* drug, based on the original query's intent.
4. Output ONLY in JSON format like this:
   {"is_comparison": true/false, "results": [{"entity": "drug_name", "sub_questions": ["question_1", "question_2"]}, ...]}
5. If it's not a comparison, 'results' must be an empty list.
6. **CRITICAL**: Make sub-questions GENERAL and avoid specific medical terms. For nausea/stomach issues, ask about "side effects" generally. For effectiveness, ask about "effectiveness" or "user ratings" generally.

Example 1:
Q: 'Compare Wegovy and Ozempic for effectiveness.'
A: {"is_comparison": true, "results": [{"entity": "Wegovy", "sub_questions": ["What is the effectiveness of Wegovy based on user reviews?"]}, {"entity": "Ozempic", "sub_questions": ["What is the effectiveness of Ozempic based on user reviews?"]}]}

Example 2:
Q: 'What are the side effects of Mounjaro?'
A: {"is_comparison": false, "results": []}

Example 3:
Q: 'Which is less likely to cause nausea, Ozempic or Rybelsus? And which works better for blood sugar?'
A: {"is_comparison": true, "results": [{"entity": "Ozempic", "sub_questions": ["What are the side effects of Ozempic?", "How effective is Ozempic in controlling blood sugar?"]}, {"entity": "Rybelsus", "sub_questions": ["What are the side effects of Rybelsus?", "How effective is Rybelsus in controlling blood sugar?"]}]}

Example 4:
Q: 'Tell me about Zepbound.'
A: {"is_comparison": false, "results": []}
"""
        prompt = f"User Question: {user_query}\nJSON Output:"

        try:
            response = self.llm.query(prompt, system_message=system_message, temperature=0.0, max_tokens=768) # Increased max_tokens for potentially complex JSON
            if self.verbose: logger.info(f"Decomposition LLM response: {response}")
            decomp_result = read_json_from_llm(response)
            
            # New Validation
            if not isinstance(decomp_result, dict) or \
               'is_comparison' not in decomp_result or \
               'results' not in decomp_result or \
               not isinstance(decomp_result['results'], list):
                raise ValueError("LLM response is not in the expected JSON format (missing keys or wrong types).")

            if decomp_result['is_comparison']:
                if not decomp_result['results']: # If it claims comparison, must have results
                    logger.warning(f"Comparison marked true, but 'results' is empty. Treating as non-comparison. {decomp_result}")
                    decomp_result['is_comparison'] = False
                    decomp_result['results'] = [] # Ensure results is empty
                else:
                    # Check structure of each item in results
                    valid_results = []
                    all_items_valid = True
                    for item in decomp_result['results']:
                        if not isinstance(item, dict) or \
                           'entity' not in item or \
                           not isinstance(item.get('entity'), str) or \
                           not item.get('entity').strip() or \
                           'sub_questions' not in item or \
                           not isinstance(item.get('sub_questions'), list) or \
                           not item.get('sub_questions') or \
                           not all(isinstance(sq, str) and sq.strip() for sq in item['sub_questions']): # Ensure sub_questions list is not empty and contains valid strings
                            logger.warning(f"Inconsistent item in 'results'. Item: {item}, Full: {decomp_result}")
                            all_items_valid = False
                            break # Stop checking on first error
                        valid_results.append(item)
                    
                    if not all_items_valid:
                        decomp_result['is_comparison'] = False
                        decomp_result['results'] = [] # Clear results on error
                    else:
                        decomp_result['results'] = valid_results # Use only validated items

            else: # Not a comparison
                # If not comparison, ensure results is empty for consistency
                decomp_result['results'] = []

            return decomp_result

        except Exception as e:
            logger.error(f"Error during query decomposition or validation: {e}. Treating as non-comparison.", exc_info=True)
            return {"is_comparison": False, "results": []} # Return a consistent structure

    def _get_context_for_query(self, query: str) -> str:
        """
        Helper function to get TableRAG and GraphRAG context for a single query.
        It now tries both RAGs if available, as routing is less critical here.
        """
        use_table_rag = self.table_rag_agent is not None and self.table_rag_df is not None and not self.table_rag_df.empty
        use_graph_rag = self.graph_rag_module is not None and self.graph_rag_module._driver is not None
        
        context = ""

        if use_table_rag:
            if self.verbose: logger.info(f"--- Invoking TableRAG for sub-query: '{query}' ---")
            try:
                table_rag_result = self.table_rag_agent.run(
                    user_query=query,
                    table_df=self.table_rag_df.copy(),
                    table_id=TABLE_ID,
                    table_caption=TABLE_CAPTION
                )
                table_rag_answer = table_rag_result.get('answer', "No specific answer from TableRAG.")
                context += f"TableRAG Analysis:\n{table_rag_answer}\n\n"
            except Exception as e:
                logger.error(f"Error during TableRAG for sub-query '{query}': {e}", exc_info=True)
                context += "TableRAG Error: Could not process.\n\n"

        if use_graph_rag:
            if self.verbose: logger.info(f"--- Invoking GraphRAG for sub-query: '{query}' ---")
            try:
                graph_rag_context_raw = self.graph_rag_module.query_graph(query)
                # Avoid adding "GraphRAG Error:" or "No specific results" if they are the *only* thing.
                # Only add context if it seems meaningful or contains actual results/cypher.
                if "GraphRAG Context" in graph_rag_context_raw or ("GraphRAG:" in graph_rag_context_raw and "Error" not in graph_rag_context_raw and "No specific results" not in graph_rag_context_raw):
                    context += f"GraphRAG Analysis:\n{graph_rag_context_raw}\n\n"
                elif "Error" in graph_rag_context_raw:
                    context += f"GraphRAG Note: Could not retrieve graph data ({graph_rag_context_raw}).\n\n"
                else: # No results found, add a softer note.
                    context += f"GraphRAG Note: No specific graph results found for this sub-query.\n\n"

            except Exception as e:
                logger.error(f"Error during GraphRAG for sub-query '{query}': {e}", exc_info=True)
                context += "GraphRAG Error: Could not process.\n\n"
        
        return context.strip()

    def get_response(self, user_query: str) -> str:
        if self.verbose: logger.info(f"\nProcessing User Query: '{user_query}'")

        # Check for summary/pros-cons query FIRST
        if self._is_summary_query(user_query):
            drug_name = self._extract_drug_name(user_query)
            if drug_name:
                if self.verbose: logger.info(f"Pro/Con query detected for {drug_name}. Fetching structured info.")
                
                # Use the NEW function to get structured pro/con context
                pro_con_findings = self._get_pro_con_context(drug_name)
                
                # Construct context for the final LLM
                final_context_parts = [f"Structured Findings for {drug_name}:"]
                if pro_con_findings.get("ratings"):
                    final_context_parts.append(f"\nQuantitative User Feedback:\n{pro_con_findings['ratings']}")
                if pro_con_findings.get("treated_conditions"):
                    final_context_parts.append(f"\nReported Therapeutic Uses:\n{pro_con_findings['treated_conditions']}")
                if pro_con_findings.get("side_effects"):
                    final_context_parts.append(f"\nReported Side Effects (Potential Downsides):\n{pro_con_findings['side_effects']}")
                
                if len(final_context_parts) == 1: # Only the initial title, no data found
                    if pro_con_findings.get("error_messages"):
                        errors = "; ".join(pro_con_findings["error_messages"])
                        return f"I encountered some issues while trying to gather information for {drug_name}: {errors}. Therefore, I cannot fully answer your question."
                    return f"I could not find specific structured information about pros or cons for {drug_name} to answer your question."

                combined_context = "\n".join(final_context_parts)

                final_prompt_system = """You are a helpful medical information assistant. 
Your task is to answer the user's question about the "best things" and "biggest downsides" of a drug, based *only* on the provided structured findings.
The findings include quantitative user feedback (like average ratings), reported therapeutic uses (conditions treated), and reported side effects.
You *do not* have access to raw user review texts. Frame your answer accordingly.

1.  Analyze the 'Quantitative User Feedback' for positive indicators (e.g., high average ratings for Overall, Effectiveness, Satisfaction; high 'Likes Count'). These represent the "best things" from a data perspective.
2.  Analyze the 'Reported Therapeutic Uses' as positive aspects.
3.  Analyze the 'Reported Side Effects' as the "biggest downsides."
4.  Synthesize these structured findings into a balanced answer.
5.  Clearly state that your answer is based on these data points, not a direct summary of user statements.
6.  If some information is missing (e.g., no side effects listed, or ratings not available), mention that.
"""
                final_prompt_user = f"User's Original Question: {user_query}\n\nProvided Structured Findings:\n{combined_context}\n\nBased ONLY on these structured findings, what are the best things and biggest downsides?"
                
                if self.verbose: logger.info("\n--- Generating Final Pro/Con Answer with LLM ---")
                final_answer = self.llm.query(final_prompt_user, system_message=final_prompt_system, temperature=0.3, max_tokens=1024)
                return final_answer
            else:
                logger.warning("Summary/Pro/Con query detected, but couldn't extract drug name.")
                return "I detected that you're asking for pros and cons information, but I couldn't identify which specific drug you're asking about. Please mention one of the following drugs clearly: Wegovy, Ozempic, Rybelsus, Zepbound, Mounjaro, Victoza, or Saxenda."

        # --- Standard existing logic for other query types ---
        decomposition = self._decompose_query(user_query)
        combined_context = ""
        is_comparison = decomposition.get('is_comparison', False)
        results = decomposition.get('results', []) # Get the results list

        if is_comparison and results: # Check if it's comparison AND results are valid
            if self.verbose: logger.info(f"Comparison detected. Processing results: {results}")
            for item in results:
                entity = item.get('entity', 'Unknown Entity')
                sub_questions = item.get('sub_questions', [])
                
                combined_context += f"--- Information for {entity} ---\n"
                
                for sub_query in sub_questions:
                    if self.verbose: logger.info(f"   - Getting context for sub-query: '{sub_query}' for entity '{entity}'")
                    combined_context += f"Context for '{sub_query}':\n"
                    combined_context += self._get_context_for_query(sub_query) # Call RAG for each sub-query
                    combined_context += "\n\n" # Add spacing between sub-query results
                    
                combined_context += "="*40 + "\n\n" # Add separator between entities

        else: # Handle non-comparison or cases where decomposition failed
            if self.verbose: logger.info("Not a comparison query (or decomposition failed). Processing as single query.")
            combined_context = self._get_context_for_query(user_query)


        combined_context = combined_context.strip()

        if not combined_context:
            logger.warning(f"No context retrieved for query: '{user_query}'.")
            return "I could not retrieve any specific information for your query from my knowledge sources at this time. Please try rephrasing your question or ask something different."

        # LLM Answer Generation
        if self.verbose: logger.info("\n--- Generating Final Answer with LLM using Combined Context ---")

        if is_comparison:
            final_prompt_system = """You are a helpful medical information assistant.
Your task is to synthesize a **comparison** answer to the user's original question based ONLY on the provided context for each drug.
The context contains results from TableRAG (tabular data analysis) and/or GraphRAG (knowledge graph insights).
1.  Address the user's original comparison question directly.
2.  Highlight similarities and differences based *only* on the provided context.
3.  If the context is contradictory or insufficient for a clear comparison, acknowledge that.
4.  Do not use any external knowledge.
5.  If the context indicates errors or lack of data for a point, mention that information retrieval had issues or no data was found for that specific aspect.
6.  Present the comparison clearly and concisely.
"""
            final_prompt_user = f"User's Original Comparison Question: {user_query}\n\nProvided Context for Each Drug:\n{combined_context}\n\nFinal Comparison Answer:"
        else:
            final_prompt_system = """You are a helpful medical information assistant.
Synthesize an answer to the user's question based ONLY on the provided context from TableRAG (tabular data analysis) and/or GraphRAG (knowledge graph insights).
If the context is contradictory or insufficient to answer the question comprehensively, acknowledge that.
Do not use any external knowledge or make assumptions beyond the provided context.
If the context indicates errors from the RAG systems or lack of data, state that you encountered issues retrieving the information or that no specific data was found.
Present the answer clearly and concisely.
"""
            final_prompt_user = f"User Question: {user_query}\n\nProvided Context:\n{combined_context}\n\nFinal Answer:"

        # Context Truncation
        max_final_context_tokens = 100000 # Max tokens for the context part of the prompt
        # Estimate tokens for the static parts of the prompt (system message, user query, instructions)
        # This is a rough estimate; actual token count for these parts can vary.
        # For gpt-4.1-nano with 128k context, this should be fine.
        # If using models with smaller context windows, this needs to be more precise.
        prompt_overhead_approx = self.llm.get_token_count(final_prompt_system + user_query + "Final Answer:")
        
        # Calculate remaining tokens available for the combined_context
        available_tokens_for_context = self.llm.context_limit - prompt_overhead_approx - 1024 # Reserve 1024 for LLM output
        if available_tokens_for_context < 1000 : # Ensure at least some context can be passed
            available_tokens_for_context = 1000 # Minimum context tokens
            logger.warning(f"Very limited token space for context ({available_tokens_for_context}). Final answer quality might be affected.")

        current_context_tokens = self.llm.get_token_count(combined_context)
        
        if current_context_tokens > available_tokens_for_context:
            # Truncate combined_context if it's too long
            # This is a simple character-based truncation. More sophisticated methods (e.g., summarizing, selecting most relevant parts) could be used.
            estimated_chars_per_token = 3.5 # A rough average, can vary
            max_chars = int(available_tokens_for_context * estimated_chars_per_token)
            truncated_context = combined_context[:max_chars] + "\n... (Context truncated due to length)"
            logger.warning(f"Combined context was too long ({current_context_tokens} tokens, exceeding available {available_tokens_for_context}), truncated to approx {self.llm.get_token_count(truncated_context)} tokens.")
            
            # Update the user prompt part with the truncated context
            if is_comparison:
                final_prompt_user = f"User's Original Comparison Question: {user_query}\n\nProvided Context for Each Drug:\n{truncated_context}\n\nFinal Comparison Answer:"
            else:
                final_prompt_user = f"User Question: {user_query}\n\nProvided Context:\n{truncated_context}\n\nFinal Answer:"
        
        if self.verbose: logger.info(f"Final prompt to LLM (User part, context truncated if needed):\n{final_prompt_user[:1000]}...")
        final_answer = self.llm.query(final_prompt_user, system_message=final_prompt_system, temperature=0.3, max_tokens=1536) # Increased max_tokens for comparison
        return final_answer

    def chat_loop(self):
        print("\nMedical ChatBot is ready. Type 'exit' to quit.")
        print("Enter your medical questions (e.g., 'What are common side effects of Ozempic?', 'Compare satisfaction ratings for Wegovy and Mounjaro').")
        while True:
            try:
                user_input = input("You: ")
            except EOFError: # Handle EOF if input is piped
                print("EOF received, exiting.")
                break
            if user_input.lower() == 'exit':
                if self.graph_rag_module:
                    self.graph_rag_module.close()
                logger.info("Exiting chatbot. Goodbye!")
                break
            if not user_input.strip():
                continue
            
            response = self.get_response(user_input)
            print(f"Bot: {response}")

# --- Entry Point ---
if __name__ == "__main__":
    # Setup basic logging to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # For verbose output from the chatbot's operations:
    # Set to True for detailed logs, including intermediate RAG steps and prompts.
    # Set to False for cleaner, user-focused output.
    # chatbot_verbose = True
    chatbot_verbose = False
    
    try:
        chatbot = MedicalChatBot(verbose=chatbot_verbose)
        
        # Example automated queries for testing (optional)
        # To run these, uncomment them and potentially comment out chat_loop()
        # test_queries = [
        #     "What are the common side effects of Ozempic?",
        #     "Which drugs treat Type 2 Diabetes Mellitus?",
        #     "What is the average Overall Rating for Wegovy?",
        #     "How many reviews mention 'nausea' as a side effect for Mounjaro?", # This might be hard for current TableRAG
        #     "What is the generic name for Zepbound?",
        #     "Compare satisfaction ratings for patients taking Wegovy for 'Obesity' vs 'Weight Loss'.", # Complex TableRAG
        #     "Between Ozempic and Rybelsus, which one is less likely to make me feel nauseous? And which one works better for controlling blood sugar?"
        # ]
        # for t_query in test_queries:
        #     print(f"\n--- Testing with query: {t_query} ---")
        #     test_response = chatbot.get_response(t_query)
        #     print(f"Bot Response to '{t_query}': {test_response}\n")

        chatbot.chat_loop()

    except ValueError as ve: # Catch specific init errors like missing API key
        logger.critical(f"Initialization failed: {ve}")
        print(f"Chatbot initialization failed: {ve}. Please check your configuration.")
    except ImportError as ie:
        logger.critical(f"Initialization failed due to missing dependencies: {ie}")
        print(f"Chatbot initialization failed due to missing dependencies: {ie}. Please install required packages.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")

# conda activate weightloss && python code_chatbot/chatbot.py