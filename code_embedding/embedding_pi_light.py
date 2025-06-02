import os
import re
import json
import pdfplumber
import camelot
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 1. Configuration
PDF_DIR = 'data_prescribing_information'
OUTPUT_PATH = 'data_embedded/embedded_pi.csv'
MODEL_NAME = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
CHUNK_SIZE = 500  # Maximum characters per chunk, adjustable
CHUNK_OVERLAP = 50  # Overlap characters between chunks

# 2. Utility Functions
def extract_text_and_tables(pdf_path):
    """Parse PDF, extract text and table content. Returns: Chunks organized by page."""
    all_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(tqdm(pdf.pages, desc=f'Processing {os.path.basename(pdf_path)}', leave=False)):
            # Extract text
            text = page.extract_text() or ''
            if text.strip():
                all_chunks.append({'type': 'text', 'page': i+1, 'content': text})
            # Extract tables using camelot
            try:
                # camelot reads all pages at once, so we filter later
                tables = camelot.read_pdf(pdf_path, pages=str(i+1), flavor='stream')
                for t in tables:
                    df = t.df
                    if not df.empty:
                        table_str = df.to_csv(index=False)
                        all_chunks.append({'type': 'table', 'page': t.page, 'content': table_str})
            except Exception as e:
                # print(f"[Warning] Table parsing failed on page {i+1} of {os.path.basename(pdf_path)}: {e}") # Too noisy
                pass # Suppress per-page warnings for cleaner output
    # Camelot reads all pages for all tables at once. This might lead to duplicate table entries
    # if camelot is called inside the page loop. A better approach is to call camelot once
    # outside the loop and then process results page by page. Let's revert camelot call outside.
    all_chunks = []
    tables = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    except Exception as e:
        print(f"[Warning] Initial camelot table parsing failed for {os.path.basename(pdf_path)}: {e}")
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(tqdm(pdf.pages, desc=f'Processing {os.path.basename(pdf_path)}', leave=False)):
            page_num = i + 1
            # Extract text
            text = page.extract_text() or ''
            if text.strip():
                all_chunks.append({'type': 'text', 'page': page_num, 'content': text})
            # Process tables extracted by camelot for this page
            page_tables = [t for t in tables if t.page == page_num]
            for t in page_tables:
                 df = t.df
                 if not df.empty:
                     table_str = df.to_csv(index=False)
                     all_chunks.append({'type': 'table', 'page': page_num, 'content': table_str})
    return all_chunks

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text by hierarchy and length, preserving paragraphs, lists, etc."""
    paras = re.split(r'\n{2,}', text)
    chunks = []
    for para in paras:
        para = para.strip()
        if not para:
            continue
        # Split long paragraphs
        start = 0
        while start < len(para):
            end = min(start + chunk_size, len(para))
            chunk = para[start:end]
            if chunk:
                chunks.append(chunk)
            start = end - overlap  # With overlap
    return chunks

def chunk_table(table_str, chunk_size=CHUNK_SIZE):
    """Split table by rows, ensuring chunks do not exceed chunk_size."""
    lines = table_str.strip().split('\n')
    header = lines[0]
    rows = lines[1:]
    chunks = []
    cur_chunk = [header]
    cur_len = len(header)
    for row in rows:
        if cur_len + len(row) > chunk_size and len(cur_chunk) > 1:
            chunks.append('\n'.join(cur_chunk))
            cur_chunk = [header, row]
            cur_len = len(header) + len(row)
        else:
            cur_chunk.append(row)
            cur_len += len(row)
    if len(cur_chunk) > 1:
        chunks.append('\n'.join(cur_chunk))
    return chunks

# 3. Main Process
def main():
    model = SentenceTransformer(MODEL_NAME)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    all_records = []
    for pdf_file in tqdm(pdf_files, desc='Processing PDF files'):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        doc_name = os.path.splitext(pdf_file)[0]
        chunks = extract_text_and_tables(pdf_path)
        for chunk in chunks:
            if chunk['type'] == 'text':
                sub_chunks = chunk_text(chunk['content'])
            elif chunk['type'] == 'table':
                sub_chunks = chunk_table(chunk['content'])
            else:
                continue
            for i, sub in enumerate(sub_chunks):
                record = {
                    'doc': doc_name,
                    'page': chunk['page'],
                    'type': chunk['type'],
                    'chunk_id': i,
                    'text': sub
                }
                all_records.append(record)
    # Generate embeddings
    print(f'Total {len(all_records)} chunks, starting embedding generation...')
    texts = [r['text'] for r in all_records]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # Save results
    emb_df = pd.DataFrame(all_records)
    emb_arr = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    out_df = pd.concat([emb_df, emb_arr], axis=1)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved to {OUTPUT_PATH}')

if __name__ == '__main__':
    main()