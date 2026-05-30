"""
Utility functions untuk PII Shield project.
"""

import json
import os
from typing import List, Dict


def load_json(filepath: str) -> any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: any, filepath: str) -> None:
    """Save data sebagai JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def ensure_dir_exists(dirpath: str) -> None:
    """Ensure directory exists."""
    os.makedirs(dirpath, exist_ok=True)


def count_tokens_in_dataset(data: List[Dict]) -> int:
    """Count total tokens in dataset."""
    return sum(len(doc['tokens']) for doc in data)


def count_labels_in_dataset(data: List[Dict]) -> Dict[str, int]:
    """Count frequency of each label in dataset."""
    label_counts = {}
    for doc in data:
        for label in doc['labels']:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    return dict(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))


def get_dataset_stats(data: List[Dict]) -> Dict:
    """Get comprehensive statistics about dataset."""
    total_docs = len(data)
    total_tokens = count_tokens_in_dataset(data)
    label_counts = count_labels_in_dataset(data)
    
    pii_labels = {k: v for k, v in label_counts.items() if k != 'O'}
    total_pii_tokens = sum(pii_labels.values())
    pii_percentage = (total_pii_tokens / total_tokens * 100) if total_tokens > 0 else 0
    
    return {
        'total_documents': total_docs,
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': total_tokens // total_docs if total_docs > 0 else 0,
        'label_counts': label_counts,
        'pii_labels_count': len(pii_labels),
        'pii_tokens': total_pii_tokens,
        'pii_percentage': round(pii_percentage, 2),
        'o_tokens': label_counts.get('O', 0),
        'o_percentage': round(label_counts.get('O', 0) / total_tokens * 100, 2) if total_tokens > 0 else 0
    }


def print_dataset_stats(data: List[Dict], dataset_name: str = "Dataset") -> None:
    """Print dataset statistics."""
    stats = get_dataset_stats(data)
    
    print(f"\n{'='*70}")
    print(f"{dataset_name} Statistics".center(70))
    print(f"{'='*70}")
    print(f"Total documents: {stats['total_documents']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens per document: {stats['avg_tokens_per_doc']}")
    
    print(f"\nLabel Distribution:")
    print(f"  O (non-PII): {stats['o_tokens']:,} tokens ({stats['o_percentage']}%)")
    print(f"  PII tokens: {stats['pii_tokens']:,} tokens ({stats['pii_percentage']}%)")
    
    print(f"\nTop PII labels:")
    for label, count in list(stats['label_counts'].items())[:10]:
        if label != 'O':
            percentage = (count / stats['total_tokens'] * 100)
            print(f"  {label:20s}: {count:7,} ({percentage:5.2f}%)")


if __name__ == "__main__":
    # Test utils
    from data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.load_raw_json("train.json")
    
    print_dataset_stats(data, "Train Dataset")
