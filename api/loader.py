"""
api/loader.py
==============
ZINOHK trained data loader.

Downloads from HuggingFace Hub on first use.
Caches locally — never re-downloads unless version changes.

Usage:
    from api.loader import load_wiki_index, load_finance_kb
    index, titles, summaries = load_wiki_index()
    fin_index, fin_meta      = load_finance_kb()
"""

import os
import pickle
import faiss
from huggingface_hub import hf_hub_download

REPO_ID   = "orbitaven/zinohk"
HF_TOKEN  = os.environ.get("HF_TOKEN", None)
CACHE_DIR = os.path.expanduser("~/.zinohk_cache")


def _download(filename: str) -> str:
    """
    Download a file from HuggingFace Hub.
    Returns local path. Uses cache if already downloaded.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"  Checking {filename}...")
    local_path = hf_hub_download(
        repo_id   = REPO_ID,
        filename  = f"trained/{filename}",
        repo_type = "model",
        token     = HF_TOKEN,
        cache_dir = CACHE_DIR,
    )
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"  ✅ {filename} ready ({size_mb:.0f} MB)")
    return local_path


def load_wiki_index():
    """
    Load 6.4M Wikipedia FAISS index from HuggingFace.
    Downloads once, cached forever.

    Returns
    -------
    index     : faiss.Index
    titles    : List[str]
    summaries : List[str]
    """
    print("Loading Wikipedia knowledge base...")

    index_path = _download("wiki_full.index")
    meta_path  = _download("meta_full.pkl")

    print("  Loading FAISS index into memory...")
    index = faiss.read_index(index_path)

    print("  Loading metadata...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    titles    = meta['titles']
    summaries = meta['summaries']

    print(f"✅ Wikipedia loaded: {index.ntotal:,} facts")
    return index, titles, summaries


def load_finance_kb():
    """
    Load finance knowledge base from HuggingFace.

    Returns
    -------
    fin_index : faiss.Index
    fin_meta  : dict with titles, answers, categories
    """
    print("Loading finance knowledge base...")

    index_path = _download("finance.index")
    meta_path  = _download("finance_meta.pkl")

    fin_index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        fin_meta = pickle.load(f)

    print(f"✅ Finance KB loaded: {fin_index.ntotal:,} facts")
    return fin_index, fin_meta


def load_trigram():
    """
    Load trigram language model from HuggingFace.

    Returns
    -------
    tri_data : dict with W, vocab_word2idx, vocab_idx2word
    """
    print("Loading trigram model...")
    path = _download("trigram_model.pkl")
    with open(path, 'rb') as f:
        tri_data = pickle.load(f)
    print(f"✅ Trigram loaded: {len(tri_data['vocab_word2idx']):,} words")
    return tri_data
