# build_webmd_db_simple_embed.py

import os
import pandas as pd
from typing import Optional, List, Any
from collections import Counter
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json

# ==============================================================================
# 1. Configuration Parameters
# ==============================================================================
CONFIG = {
    "data_path": "data_standardized/standardized_reviews_all.csv",
    "db_dir": "database_table/", 
    # "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "embed_model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    # "embed_model_name": "BAAI/bge-large-en-v1.5", 
    "mode": "embed",  
    "max_encode_cell": 10000,
    "table_id": "standardized_reviews_all", 
    "top_k": 5,
}

# ==============================================================================
# 2. Define Retriever Class (Simplified)
# ==============================================================================
class Retriever:
    """
    A simplified Retriever class that only supports HuggingFace embeddings and FAISS storage.
    """
    def __init__(self, agent_type, mode, embed_model_name, top_k = 5, max_encode_cell = 10000, db_dir = 'db/', verbose = False):
        self.agent_type = agent_type
        # mode is fixed as embed, but the parameter is retained for future expansion (or to maintain structural consistency)
        self.mode = 'embed'
        self.embed_model_name = embed_model_name
        self.schema_retriever = None
        self.cell_retriever = None
        self.row_retriever = None
        self.column_retriever = None
        self.top_k = top_k
        self.max_encode_cell = max_encode_cell
        self.db_dir = db_dir
        self.verbose = verbose
        self.df = None
        os.makedirs(db_dir, exist_ok=True)

        # --- Embedding Model Initialization (Simplified) ---
        # Only initialize HuggingFaceEmbeddings
        try:
            self.embedder = HuggingFaceEmbeddings(model_name=self.embed_model_name)
            if self.verbose: print(f"Using embedding model: {self.embed_model_name}")
        except Exception as e:
            print(f"Failed to initialize HuggingFace model '{self.embed_model_name}': {e}")
            print("Please ensure that the 'sentence-transformers' library is installed (pip install sentence-transformers) and the model name is correct.")
            exit()


    def init_retriever(self, table_id, df):
        """
        Initializes the retriever, building or loading the database for the specified table.
        """
        self.df = df
        if self.verbose: print(f"Initializing retriever for table_id='{table_id}'...")

        if 'TableRAG' in self.agent_type:
            print("Building Schema retriever...")
            self.schema_retriever = self.get_retriever('schema', table_id, self.df)
            print("Building Cell retriever...")
            self.cell_retriever = self.get_retriever('cell', table_id, self.df)
            print("Retriever initialization complete.")
        # Keep TableSampling logic just in case, but it will only use embed mode
        elif self.agent_type == 'TableSampling':
            print("In TableSampling mode, sampling rows...")
            max_row = max(1, self.max_encode_cell // 2 // len(self.df.columns))
            self.df = self.df.iloc[:max_row]
            print(f"Remaining {len(self.df)} rows after sampling.")
            print("Building Row retriever...")
            self.row_retriever = self.get_retriever('row', table_id, self.df)
            print("Building Column retriever...")
            self.column_retriever = self.get_retriever('column', table_id, self.df)
            print("Retriever initialization complete.")
        else:
            print(f"Warning: Unknown agent_type '{self.agent_type}', no retriever built.")


    def get_retriever(self, data_type, table_id, df):
        """
        Gets or builds the specified type of retriever (Embed Mode Only).
        """
        docs = None
        embed_retriever = None

        # --- 仅处理 Embed 模式 ---
        safe_table_id = "".join(c for c in table_id if c.isalnum() or c in ('_', '-')).rstrip()
        db_path = os.path.join(self.db_dir, f'{data_type}_db_{safe_table_id}')

        if os.path.exists(db_path):
            if self.verbose: print(f'Loading {data_type} database from {db_path}...')
            try:
                db = FAISS.load_local(db_path, self.embedder, allow_dangerous_deserialization=True)
                if self.verbose: print("Database loaded successfully.")
            except Exception as e:
                print(f"Failed to load database {db_path}: {e}. Will attempt to rebuild.")
                docs = self.get_docs(data_type, df)
                if not docs:
                    print(f"Unable to generate documents for {data_type}, skipping database build.")
                    return None
                print(f"Building FAISS database for {data_type}...")
                db = FAISS.from_documents(docs, self.embedder)
                db.save_local(db_path)
                print(f"FAISS database saved to {db_path}.")
        else:
            if self.verbose: print(f"Database {db_path} does not exist, starting build...")
            docs = self.get_docs(data_type, df)
            if not docs:
                print(f"Unable to generate documents for {data_type}, skipping database build.")
                return None
            print(f"Building FAISS database for {data_type}...")
            db = FAISS.from_documents(docs, self.embedder)
            db.save_local(db_path)
            print(f"FAISS database saved to {db_path}.")

        embed_retriever = db.as_retriever(search_kwargs={'k': self.top_k})
        return embed_retriever


    def get_docs(self, data_type, df):
        """
        Calls the corresponding corpus building method based on the data type.
        """
        if self.verbose: print(f"Building {data_type} corpus...")
        if data_type == 'schema':
            return self.build_schema_corpus(df)
        elif data_type == 'cell':
            return self.build_cell_corpus(df)
        elif data_type == 'row':
            return self.build_row_corpus(df)
        elif data_type == 'column':
            return self.build_column_corpus(df)
        else:
            print(f"Error: Unknown data type '{data_type}'")
            return []

    def build_schema_corpus(self, df):
        """
        Builds the Schema corpus, one summary document per column.
        """
        docs = []
        for col_name, col in df.items():
            try:
                col_safe = col.dropna().astype(str)
                is_numeric = pd.to_numeric(col.dropna(), errors='coerce').notna().sum() > len(col.dropna()) * 0.8
                if not col_safe.empty and is_numeric:
                    col_numeric = pd.to_numeric(col.dropna(), errors='coerce').dropna()
                    if not col_numeric.empty:
                       result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col_numeric.min()}, "max": {col_numeric.max()}}}'
                    else:
                        result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "info": "Mostly numeric but no valid values found"}}'
                elif not col_safe.empty:
                    most_freq_vals = col_safe.value_counts().index.tolist()
                    example_cells = most_freq_vals[:min(3, len(most_freq_vals))]
                    result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "cell_examples": {example_cells}}}'
                else:
                    result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "info": "Empty or all NaN"}}'
                docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
            except Exception as e:
                print(f"Error processing Schema column '{col_name}': {e}")
                result_text = f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "info": "Error processing column"}}'
                docs.append(Document(page_content=col_name, metadata={'result_text': result_text}))
        if self.verbose: print(f"Schema corpus built, total {len(docs)} documents.")
        return docs

    def build_cell_corpus(self, df):
        """
        Builds the Cell corpus, including non-text column summaries and the most common N text cells.
        """
        docs = []
        text_dtypes = ['object', 'string', 'category']
        categorical_columns = df.columns[df.dtypes.isin(text_dtypes)]
        other_columns = df.columns[~df.dtypes.isin(text_dtypes)]

        for col_name in other_columns:
            col = df[col_name].dropna()
            if not col.empty:
                try:
                    col_numeric = pd.to_numeric(col, errors='coerce').dropna()
                    if not col_numeric.empty:
                        docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "min": {col_numeric.min()}, "max": {col_numeric.max()}}}')
                    else:
                        docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "info": "Non-text, non-numeric column"}}')
                except Exception as e:
                    print(f"Error processing Cell (non-text) column '{col_name}': {e}")
                    docs.append(f'{{"column_name": "{col_name}", "dtype": "{col.dtype}", "info": "Error processing column"}}')

        if len(categorical_columns) > 0:
            all_cells = []
            print("Collecting categorical cells...")
            for col_name in tqdm(categorical_columns, desc="Processing categorical columns"):
                valid_cells = df[col_name].dropna().astype(str)
                all_cells.extend([f'{{"column_name": "{col_name}", "cell_value": "{cell}"}}' for cell in valid_cells])

            if all_cells:
                print(f"Collected {len(all_cells)} categorical cells in total, counting frequency...")
                cell_cnt = Counter(all_cells)
                num_to_keep = self.max_encode_cell - len(docs)
                print(f"Will select up to {num_to_keep} most common cells...")
                most_common_cells = [cell for cell, _ in cell_cnt.most_common(num_to_keep)]
                docs.extend(most_common_cells)
            else:
                print("No valid categorical cells found.")

        final_docs = [Document(page_content=doc) for doc in docs]
        if self.verbose: print(f"Cell corpus built, total {len(final_docs)} documents.")
        return final_docs

    def build_row_corpus(self, df):
        row_docs = []
        for row_id, (_, row) in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            row_text = '|'.join(str(cell) for cell in row)
            row_doc = Document(page_content=row_text, metadata={'row_id': row_id})
            row_docs.append(row_doc)
        if self.verbose: print(f"Row corpus built, total {len(row_docs)} documents.")
        return row_docs

    def build_column_corpus(self, df):
        col_docs = []
        for col_id, (col_name, column) in tqdm(enumerate(df.items()), total=len(df.columns), desc="Processing columns"):
            col_text = '|'.join(str(cell) for cell in column)
            col_doc = Document(page_content=col_text, metadata={'col_id': col_id, 'col_name': col_name})
            col_docs.append(col_doc)
        if self.verbose: print(f"Column corpus built, total {len(col_docs)} documents.")
        return col_docs

# ==============================================================================
# 3. Main Function
# ==============================================================================
def build_database(config):
    """
    Loads data and builds the retrieval database (Simplified).
    """
    print(f"Starting database build (Embed-Only Mode)...")
    print(f"Data file: {config['data_path']}")
    print(f"Database directory: {config['db_dir']}")
    print(f"Embedding model: {config['embed_model_name']}")
    print(f"Maximum cells: {config['max_encode_cell']}")
    print(f"Table ID: {config['table_id']}")

    try:
        print(f"Loading {config['data_path']} ...")
        df = pd.read_csv(config['data_path'])
        print(f"Data loaded successfully, {len(df)} rows, {len(df.columns)} columns.")
        print("CSV column names:", df.columns.tolist())
        
        # --- OPTIMIZE DATAFRAME FOR TABLERAG (Remove structured columns) ---
        print("\n--- Optimizing DataFrame for TableRAG ---")
        structured_columns_to_remove = ['standardized_info', 'standardized_relations']
        original_columns = df.columns.tolist()
        
        columns_found = [col for col in structured_columns_to_remove if col in df.columns]
        columns_not_found = [col for col in structured_columns_to_remove if col not in df.columns]
        
        if columns_found:
            print(f"Removing structured columns for TableRAG optimization: {columns_found}")
            df = df.drop(columns=columns_found)
            print(f"Removed {len(columns_found)} structured columns.")
        else:
            print("No structured columns found to remove (this is expected if they don't exist).")
            
        if columns_not_found:
            print(f"Note: The following columns were not found in the data: {columns_not_found}")
            
        print(f"Optimized DataFrame: {len(df)} rows, {len(df.columns)} columns.")
        print(f"Remaining columns: {df.columns.tolist()}")
        print("--- TableRAG Optimization Complete ---\n")
        
        retriever = Retriever(
            agent_type='TableRAG',
            mode=config['mode'], 
            embed_model_name=config['embed_model_name'],
            top_k=config['top_k'],
            max_encode_cell=config['max_encode_cell'],
            db_dir=config['db_dir'],
            verbose=True
        )
        retriever.init_retriever(table_id=config['table_id'], df=df)
    except FileNotFoundError:
        print(f"Error: Data file '{config['data_path']}' not found. Please check the path.")
        return
    except Exception as e:
        print(f"Error: An error occurred while loading data: {e}")
        return

    print("="*30)
    print("Database build process complete!")
    print(f"Database files saved in '{config['db_dir']}' directory.")
    print("="*30)

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    build_database(CONFIG)