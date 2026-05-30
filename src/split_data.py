"""
Split dataset menjadi train/validation/test.
Split berdasarkan document ID (bukan token), untuk fair evaluation.
"""

import json
import random
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split


def split_by_document(
    data: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset berdasarkan document ID.
    
    Penting: Split berdasarkan document, bukan token random.
    Satu document hanya ada di satu split (train/val/test).
    
    Args:
        data: List of documents dari Kaggle JSON
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        random_seed: Seed untuk reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    
    # Get unique document IDs
    unique_docs = {}
    for doc in data:
        doc_id = doc['document']
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc
    
    doc_ids = list(unique_docs.keys())
    num_docs = len(doc_ids)
    
    print(f"Total unique documents: {num_docs}")
    print(f"Total documents in full data: {len(data)}")
    
    # Set seed for reproducibility
    random.seed(random_seed)
    
    # Calculate split sizes
    train_size = int(num_docs * train_ratio)
    val_size = int(num_docs * val_ratio)
    test_size = num_docs - train_size - val_size
    
    print(f"\nSplit sizes (documents):")
    print(f"  Train: {train_size} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {val_size} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {test_size} ({test_ratio*100:.0f}%)")
    
    # Shuffle and split
    random.shuffle(doc_ids)
    
    train_doc_ids = set(doc_ids[:train_size])
    val_doc_ids = set(doc_ids[train_size:train_size + val_size])
    test_doc_ids = set(doc_ids[train_size + val_size:])
    
    # Split data by document
    train_data = [doc for doc in data if doc['document'] in train_doc_ids]
    val_data = [doc for doc in data if doc['document'] in val_doc_ids]
    test_data = [doc for doc in data if doc['document'] in test_doc_ids]
    
    print(f"\nSplit sizes (tokens):")
    train_tokens = sum(len(doc['tokens']) for doc in train_data)
    val_tokens = sum(len(doc['tokens']) for doc in val_data)
    test_tokens = sum(len(doc['tokens']) for doc in test_data)
    total_tokens = train_tokens + val_tokens + test_tokens
    
    print(f"  Train: {train_tokens} tokens ({train_tokens/total_tokens*100:.1f}%)")
    print(f"  Val:   {val_tokens} tokens ({val_tokens/total_tokens*100:.1f}%)")
    print(f"  Test:  {test_tokens} tokens ({test_tokens/total_tokens*100:.1f}%)")
    
    return train_data, val_data, test_data


def save_split(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict], 
               output_dir: str = "data/processed") -> None:
    """
    Save split data menjadi JSON files.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSON
    with open(f"{output_dir}/train.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(f"{output_dir}/val.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(f"{output_dir}/test_internal.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nData saved to {output_dir}/")
    print(f"  ✓ train.json")
    print(f"  ✓ val.json")
    print(f"  ✓ test_internal.json")


if __name__ == "__main__":
    # Load raw data
    print("Loading train.json from Kaggle...")
    with open("train.json", 'r') as f:
        raw_data = json.load(f)
    
    # Split data
    print(f"\nSplitting {len(raw_data)} documents...")
    train_data, val_data, test_data = split_by_document(
        raw_data,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # Save split
    save_split(train_data, val_data, test_data)
