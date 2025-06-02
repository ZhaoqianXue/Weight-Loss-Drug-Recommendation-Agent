import os
import re
import json
import pdfplumber
import camelot
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    # Fallback if nltk is not available
    sent_tokenize = lambda x: re.split(r'(?<=[.!?])\s+', x)

# 1. Configuration
PDF_DIR = 'data_prescribing_information'
OUTPUT_PATH = 'data_embedded/embedded_pi.csv'
MODEL_NAME = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
CHUNK_SIZE = 500  # Maximum characters per chunk, adjustable
CHUNK_OVERLAP = 50  # Number of overlapping characters between chunks

# --------- Section Hierarchy Tracking Tools ---------
def get_heading_level(text):
    # e.g., 1., 1.1, 1.1.1 or all caps
    if re.match(r'^[A-Z][A-Z0-9 \-]{4,}$', text):
        return 1  # H1: All caps
    m = re.match(r'^(\d+)([.\d]*)', text)
    if m:
        dots = m.group(2).count('.')
        return 1 + dots  # 1, 1.1, 1.1.1
    return None

def update_section_stack(stack, heading, level):
    # Pop deeper levels than the current level
    while stack and stack[-1][1] >= level:
        stack.pop()
    stack.append((heading, level))
    return stack

def get_section_path(stack):
    return ' > '.join([h for h, l in stack]) if stack else None

# --------- Table to Markdown ---------
def table_to_markdown(df):
    try:
        if df.shape[0] > 1:
            # Try to use the first row as the header
            header = df.iloc[0].tolist()
            if all(isinstance(h, str) for h in header):
                df.columns = header
                df = df[1:]
        header = '| ' + ' | '.join(df.columns) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(df.columns)) + ' |'
        rows = ['| ' + ' | '.join(map(str, row)) + ' |' for row in df.values]
        return '\n'.join([header, sep] + rows)
    except Exception:
        # Fallback to CSV if markdown conversion fails
        return df.to_csv(index=False)

# --------- Main Structured Chunking Function ---------
def extract_structured_chunks(pdf_path):
    all_chunks = []
    # 1. Extract table areas using camelot (lattice first, then stream)
    table_areas = []
    tables = []
    for flavor in ['lattice', 'stream']:
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            for t in tables:
                if not t.df.empty:
                    # Record table area to avoid duplicate text extraction
                    area = getattr(t, '_bbox', None)
                    if area is None and hasattr(t, '_bbox'):  # fallback
                        area = t._bbox
                    if area is None and hasattr(t, 'parsing_report'):
                        # Approximate using table bbox from report
                        area = t.parsing_report.get('bbox', None)
                    table_areas.append({'page': t.page, 'area': area})
            if tables:
                break
        except Exception:
            continue
    # 2. Parse page by page with pdfplumber, supporting cross-page elements
    section_stack = []
    prev_para_buf = []
    prev_section_path = None
    prev_type = None
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(tqdm(pdf.pages, desc=f'Processing {os.path.basename(pdf_path)}', leave=False)):
            page_num = i + 1
            # 2.1 Identify table bounding boxes on the current page
            page_tables = [t for t in tables if t.page == page_num]
            table_bboxes = [getattr(t, '_bbox', None) for t in page_tables]
            # 2.2 Identify tables and convert to markdown
            for t in page_tables:
                try:
                    df = t.df
                    if not df.empty:
                        md = table_to_markdown(df)
                        all_chunks.append({
                            'type': 'table',
                            'page': page_num,
                            'section_path': get_section_path(section_stack),
                            'chunk_id': None,
                            'text': md
                        })
                except Exception:
                    # If markdown conversion fails, keep as CSV string
                    all_chunks.append({
                        'type': 'table',
                        'page': page_num,
                        'section_path': get_section_path(section_stack),
                        'chunk_id': None,
                        'text': t.df.to_csv(index=False)
                    })
            # 2.3 Extract text (prefer extract_text, use extract_words if needed)
            text = page.extract_text() or ''
            if not text.strip():
                words = page.extract_words(keep_blank_chars=True, use_text_flow=True)
                # Filter out words within table areas
                def in_table_bbox(word):
                    x0, top, x1, bottom = float(word['x0']), float(word['top']), float(word['x1']), float(word['bottom'])
                    for bbox in table_bboxes:
                        if bbox is None:
                            continue
                        bx0, by0, bx1, by1 = bbox
                        if (x0 >= bx0 and x1 <= bx1 and top >= by0 and bottom <= by1):
                            return True
                    return False
                words = [w for w in words if not in_table_bbox(w)]
                # Aggregate words into lines based on y-coordinate
                lines = {}
                for w in words:
                    y = round(w['top'], 1)
                    lines.setdefault(y, []).append(w)
                sorted_lines = [lines[y] for y in sorted(lines.keys())]
                line_texts = [''.join([w['text'] for w in line]).strip() for line in sorted_lines]
            else:
                # extract_text should automatically handle table removal
                line_texts = [l.strip() for l in text.split('\n') if l.strip()]
            # 2.4 Identify structure: multi-level headings, lists, paragraphs
            for line_text in line_texts:
                if not line_text:
                    continue
                # Heading identification
                heading_level = get_heading_level(line_text)
                if heading_level:
                    section_stack = update_section_stack(section_stack, line_text, heading_level)
                    continue
                # List item identification
                if re.match(r'^\s*([-*•\d]+[. )])', line_text):
                    # Merge same type across pages
                    if prev_type == 'list' and prev_section_path == get_section_path(section_stack):
                        prev_para_buf.append(line_text)
                    else:
                        if prev_para_buf:
                            all_chunks.extend(structured_chunk_from_para(prev_para_buf, prev_section_path, page_num-1, prev_type))
                        prev_para_buf = [line_text]
                        prev_section_path = get_section_path(section_stack)
                        prev_type = 'list'
                    continue
                # Paragraph
                if prev_type == 'paragraph' and prev_section_path == get_section_path(section_stack):
                    prev_para_buf.append(line_text)
                else:
                    if prev_para_buf:
                        all_chunks.extend(structured_chunk_from_para(prev_para_buf, prev_section_path, page_num-1, prev_type))
                    prev_para_buf = [line_text]
                    prev_section_path = get_section_path(section_stack)
                    prev_type = 'paragraph'
            # Process remaining buffer from the last page
            if prev_para_buf:
                all_chunks.extend(structured_chunk_from_para(prev_para_buf, prev_section_path, page_num, prev_type))
    # Add chunk_id
    for idx, c in enumerate(all_chunks):
        c['chunk_id'] = idx
    return all_chunks

# --------- Chunking (with overlap) ---------
def structured_chunk_from_para(para_buf, section_path, page_num, para_type):
    text = '\n'.join(para_buf)
    if len(text) <= CHUNK_SIZE:
        return [{
            'type': para_type,
            'page': page_num,
            'section_path': section_path,
            'chunk_id': None,
            'text': text
        }]
    # Sentence splitting + overlap
    sents = sent_tokenize(text)
    chunks = []
    cur = ''
    i = 0
    while i < len(sents):
        while len(cur) < CHUNK_SIZE and i < len(sents):
            cur = (cur + ' ' + sents[i]).strip() if cur else sents[i]
            i += 1
        chunks.append(cur)
        # overlap: move back by some sentences
        if CHUNK_OVERLAP > 0 and i < len(sents):
            overlap_sents = []
            overlap_len = 0
            j = i-1
            while j >= 0 and overlap_len < CHUNK_OVERLAP:
                overlap_sents.insert(0, sents[j])
                overlap_len += len(sents[j])
                j -= 1
            cur = ' '.join(overlap_sents)
        else:
            cur = ''
    return [{
        'type': para_type,
        'page': page_num,
        'section_path': section_path,
        'chunk_id': None,
        'text': c.strip()
    } for c in chunks]

def print_device_info():
    if torch.cuda.is_available():
        print("Using GPU (CUDA):", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("Using CPU")

# --------- Main Process ---------
def main():
    print_device_info()
    model = SentenceTransformer(MODEL_NAME)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
    all_records = []
    for pdf_file in tqdm(pdf_files, desc='Processing PDF files'):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        doc_name = os.path.splitext(pdf_file)[0]
        chunks = extract_structured_chunks(pdf_path)
        for c in chunks:
            record = {
                'doc': doc_name,
                'page': c['page'],
                'type': c['type'],
                'section_path': c.get('section_path'),
                'chunk_id': c['chunk_id'],
                'text': c['text']
            }
            all_records.append(record)
    print(f'Total {len(all_records)} chunks, starting embedding generation...')
    texts = [r['text'] for r in all_records]
    embeddings = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    emb_df = pd.DataFrame(all_records)
    emb_arr = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    out_df = pd.concat([emb_df, emb_arr], axis=1)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved to {OUTPUT_PATH}')

if __name__ == '__main__':
    main() 