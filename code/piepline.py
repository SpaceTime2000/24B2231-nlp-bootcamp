import os
import json
import torch
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

device = "cuda" if torch.cuda.is_available() else "cpu"

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_trec(file_path, run_name, claims_data, corpus_dict, scores_matrix, top_k=100):
    doc_ids = list(corpus_dict.keys())
    lines = []
    for i, claim_obj in enumerate(claims_data):
        c_id = claim_obj["id"]
        claim_scores = scores_matrix[i]
        top_indices = np.argsort(claim_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_indices, 1):
            d_id = doc_ids[idx]
            score = claim_scores[idx]
            lines.append(f"{c_id} Q0 {d_id} {rank} {score:.6f} {run_name}\n")
            
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

def compute_ndcg_at_k(gold_dict, retrieved_dict, k=10):
    ndcgs = []
    for c_id, gold_docs in gold_dict.items():
        if not gold_docs:
            continue
        preds = retrieved_dict.get(c_id, [])[:k]
        dcg = 0.0
        for rank, doc_id in enumerate(preds, 1):
            if doc_id in gold_docs:
                dcg += 1.0 / np.log2(rank + 1)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(gold_docs), k) + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0

def trec_file_to_dict(trec_path):
    retrieved = {}
    with open(trec_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            c_id, d_id, rank = int(parts[0]), int(parts[2]), int(parts[3])
            if c_id not in retrieved:
                retrieved[c_id] = []
            retrieved[c_id].append((rank, d_id))
    for c_id in retrieved:
        retrieved[c_id].sort(key=lambda x: x[0])
        retrieved[c_id] = [d[1] for d in retrieved[c_id]]
    return retrieved

def main():
    data_dir = Path("data") if Path("data").exists() else Path(".")
    corpus_raw = load_jsonl(data_dir / "corpus.jsonl")
    corpus = {doc["doc_id"]: f"{doc.get('title', '')} {' '.join(doc.get('abstract', []))}" for doc in corpus_raw}
    doc_ids = list(corpus.keys())
    doc_texts = list(corpus.values())

    train_claims = load_jsonl(data_dir / "train_claims.jsonl")
    val_claims = load_jsonl(data_dir / "val_claims.jsonl")
    eval_claims = load_jsonl(data_dir / "eval_claims.jsonl")

    val_gold = {}
    for c in val_claims:
        c_id = c["id"]
        ev = c.get("evidence", {})
        if isinstance(ev, dict):
            val_gold[c_id] = list(ev.keys())
        elif isinstance(ev, list):
            docs = [item["doc_id"] if isinstance(item, dict) and "doc_id" in item else item for item in ev]
            val_gold[c_id] = docs
        else:
            val_gold[c_id] = []

    val_gold_int = {c_id: [int(d) for d in docs] for c_id, docs in val_gold.items()}
    Path("predictions").mkdir(exist_ok=True)

    # Task 1
    tokenized_corpus = [doc.lower().split() for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    val_bm25_scores = np.array([bm25.get_scores(c["claim"].lower().split()) for c in val_claims])
    eval_bm25_scores = np.array([bm25.get_scores(c["claim"].lower().split()) for c in eval_claims])

    write_trec("predictions/val_bm25.trec", "bm25", val_claims, corpus, val_bm25_scores)
    write_trec("predictions/t1_bm25.trec", "bm25", eval_claims, corpus, eval_bm25_scores)

if __name__ == "__main__":
    main()