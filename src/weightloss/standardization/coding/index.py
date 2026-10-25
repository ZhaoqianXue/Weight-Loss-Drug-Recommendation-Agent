"""Local BioLORD-2023 embeddings (mean pooling + L2 norm, as on the model card) and top-k LLT retrieval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = 'FremyCompany/BioLORD-2023'
MAX_TOKENS = 64


class Encoder:
    def __init__(self, model_id=MODEL_ID, device='cpu'):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        self.path = snapshot_download(model_id, allow_patterns=['*.json', '*.txt', 'model.safetensors'])
        self.revision = Path(self.path).name  # commit hash of the downloaded snapshot
        weights = Path(self.path) / 'model.safetensors'
        self.weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path).to(device).eval()

    def encode(self, texts, batch_size=256):
        torch = self.torch
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches pad less
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                batch = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_TOKENS,
                                       return_tensors='pt').to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                out.append((idx, emb.cpu().numpy().astype(np.float32)))
        result = np.zeros((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, emb in out:
            result[idx] = emb
        return result

    def meta(self):
        return {'model_id': MODEL_ID, 'revision': self.revision, 'weights_sha256': self.weights_sha256,
                'pooling': 'mean', 'normalized': True, 'max_tokens': MAX_TOKENS, 'device': self.device,
                'torch': self.torch.__version__}


def build_llt_index(dictionary, encoder, out_dir):
    """Embed every current LLT name; save vectors, codes and metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llts = sorted((l for l in dictionary.llts.values() if l.current), key=lambda l: l.code)
    vectors = encoder.encode([l.name for l in llts])
    np.save(out_dir / 'llt_vectors.npy', vectors)
    codes = [l.code for l in llts]
    (out_dir / 'llt_codes.json').write_text(json.dumps(codes))
    meta = {**encoder.meta(), 'llt_current': len(codes), 'dictionary_snapshot_sha256': dictionary.snapshot_sha256,
            'vectors_sha256': hashlib.sha256((out_dir / 'llt_vectors.npy').read_bytes()).hexdigest()}
    (out_dir / 'index-meta.json').write_text(json.dumps(meta, indent=1))
    return meta


def load_llt_index(out_dir):
    out_dir = Path(out_dir)
    return (np.load(out_dir / 'llt_vectors.npy'), json.loads((out_dir / 'llt_codes.json').read_text()),
            json.loads((out_dir / 'index-meta.json').read_text()))


def top_k(query_vectors, index_vectors, k, chunk=1024):
    """Indices and cosine scores of the k most similar index rows for every query (vectors are unit length)."""
    idx = np.zeros((len(query_vectors), k), dtype=np.int64)
    scores = np.zeros((len(query_vectors), k), dtype=np.float32)
    for start in range(0, len(query_vectors), chunk):
        sims = query_vectors[start:start + chunk] @ index_vectors.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind='stable')
        idx[start:start + chunk] = np.take_along_axis(part, order, axis=1)
        scores[start:start + chunk] = np.take_along_axis(part_scores, order, axis=1)
    return idx, scores


def build_synonym_index(dictionary, encoder, mrconso, out_dir):
    """UMLS English synonyms of every concept that holds a current MedDRA LLT, each pointing back to those LLT codes.

    Used only for local retrieval: the model still sees MedDRA LLT names. UMLS's own MDR atoms (an older MedDRA
    release, and HLT/SOC names) are not used as synonyms; suppressed atoms are skipped.
    """
    from collections import Counter, defaultdict
    out_dir = Path(out_dir)
    current = {code for code, l in dictionary.llts.items() if l.current}
    llt_names = {dictionary.llts[c].name.casefold() for c in current}
    cui_codes = defaultdict(set)
    with open(mrconso, encoding='utf-8') as f:
        for line in f:
            p = line.split('|')
            if p[11] == 'MDR' and p[1] == 'ENG' and p[16] == 'N' and p[12] in ('LLT', 'PT') and p[13] in current:
                cui_codes[p[0]].add(p[13])
    synonyms, sources = defaultdict(set), Counter()
    with open(mrconso, encoding='utf-8') as f:
        for line in f:
            p = line.split('|')
            if p[0] not in cui_codes or p[1] != 'ENG' or p[16] != 'N' or p[11] == 'MDR':
                continue
            text = ' '.join(p[14].split())
            key = text.casefold()
            if len(key) < 2 or key in llt_names:
                continue
            if key not in synonyms:
                sources[p[11]] += 1
            synonyms[key].update(cui_codes[p[0]])
    rows = sorted((k, sorted(v)) for k, v in synonyms.items())
    vectors = encoder.encode([k for k, _ in rows])
    np.save(out_dir / 'syn_vectors.npy', vectors)
    (out_dir / 'syn_rows.json').write_text(json.dumps(rows, ensure_ascii=False))
    meta = {**encoder.meta(), 'mrconso': str(mrconso), 'mrconso_bytes': Path(mrconso).stat().st_size,
            'concepts_with_current_llt': len(cui_codes), 'synonym_rows': len(rows),
            'top_sources': dict(sources.most_common(15)),
            'vectors_sha256': hashlib.sha256((out_dir / 'syn_vectors.npy').read_bytes()).hexdigest()}
    (out_dir / 'syn-meta.json').write_text(json.dumps(meta, indent=1))
    return meta


def load_synonym_index(out_dir):
    out_dir = Path(out_dir)
    return np.load(out_dir / 'syn_vectors.npy'), json.loads((out_dir / 'syn_rows.json').read_text())


def top_k_codes(query_vectors, llt_vectors, llt_codes, syn_vectors, syn_rows, k, rows=100):
    """Top-k distinct LLT codes over LLT names plus UMLS synonyms; a code's score is its best row's score."""
    vectors = np.concatenate([llt_vectors, syn_vectors])
    row_codes = [[c] for c in llt_codes] + [codes for _, codes in syn_rows]
    ix, scores = top_k(query_vectors, vectors, rows)
    out = []
    for i_row, s_row in zip(ix, scores):
        seen, codes, best = set(), [], []
        for i, s in zip(i_row, s_row):
            for c in row_codes[i]:
                if c not in seen:
                    seen.add(c); codes.append(c); best.append(float(s))
            if len(codes) >= k:
                break
        out.append((codes[:k], best[:k]))
    return out
