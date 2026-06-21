"""
Script untuk memproses data imbalance.
1. Membaca raw/imbalance/train.json
2. Menghapus document yang overlap dengan processed/balance/train.json (dan file balance lainnya untuk keamanan)
3. Membagi data yang tersisa menjadi train dan val untuk processed/imbalance/
"""

import json
import random
import os
from typing import List, Dict, Tuple

def get_document_ids_from_file(filepath: str) -> set:
    """Mengambil set of document IDs dari file JSON."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} tidak ditemukan.")
        return set()
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    doc_ids = {doc['document'] for doc in data}
    return doc_ids

def get_all_balance_doc_ids() -> set:
    """Mengambil semua document IDs dari folder balance (train, val, test)."""
    balance_dir = "data/processed/balance"
    overlap_ids = set()
    
    for filename in ["train.json", "val.json", "test.json"]:
        filepath = os.path.join(balance_dir, filename)
        overlap_ids.update(get_document_ids_from_file(filepath))
        
    return overlap_ids

def split_by_document(
    data: List[Dict],
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset berdasarkan document ID."""
    unique_docs = {}
    for doc in data:
        doc_id = doc['document']
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc
    
    doc_ids = list(unique_docs.keys())
    num_docs = len(doc_ids)
    
    print(f"\nTotal unique documents setelah filter: {num_docs}")
    
    random.seed(random_seed)
    random.shuffle(doc_ids)
    
    train_size = int(num_docs * train_ratio)
    
    train_doc_ids = set(doc_ids[:train_size])
    val_doc_ids = set(doc_ids[train_size:])
    
    train_data = [doc for doc in data if doc['document'] in train_doc_ids]
    val_data = [doc for doc in data if doc['document'] in val_doc_ids]
    
    print(f"\nSplit sizes (documents):")
    print(f"  Train: {len(train_doc_ids)} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_doc_ids)} ({val_ratio*100:.0f}%)")
    
    return train_data, val_data

def main():
    print("Mencari document IDs dari dataset balance untuk mencegah overlap...")
    # Kita ambil dari train, val, dan test agar 100% aman dari data leakage
    overlap_ids = get_all_balance_doc_ids()
    print(f"Total document IDs di dataset balance: {len(overlap_ids)}")
    
    raw_imbalance_path = "data/raw/imbalance/train.json"
    print(f"\nMembaca dataset raw imbalance dari {raw_imbalance_path}...")
    with open(raw_imbalance_path, 'r') as f:
        raw_data = json.load(f)
        
    print(f"Total documents di raw imbalance: {len(raw_data)}")
    
    # Filter data yang overlap
    print("\nMenghapus documents yang overlap dengan balance dataset...")
    filtered_data = [doc for doc in raw_data if doc['document'] not in overlap_ids]
    
    docs_removed = len(raw_data) - len(filtered_data)
    print(f"Documents dihapus karena overlap: {docs_removed}")
    print(f"Documents tersisa: {len(filtered_data)}")
    
    # Split data yang sudah difilter
    train_data, val_data = split_by_document(filtered_data, train_ratio=0.80, val_ratio=0.20)
    
    # Save split
    output_dir = "data/processed/imbalance"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nMenyimpan split data ke {output_dir}/...")
    
    train_path = os.path.join(output_dir, "train.json")
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ {train_path}")
        
    val_path = os.path.join(output_dir, "val.json")
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ {val_path}")
    
    print("\nSelesai!")

if __name__ == "__main__":
    main()
