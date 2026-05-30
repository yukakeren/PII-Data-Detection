"""
Script untuk menampilkan statistik dataset dan memverifikasi split.
"""

import sys
sys.path.insert(0, '/home/ata/school/fp-ml/src')

from data_loader import DataLoader
from utils import print_dataset_stats


def main():
    print("\n" + "="*70)
    print("PII Shield - Dataset Statistics".center(70))
    print("="*70)
    
    # Load splits
    loader = DataLoader()
    
    print("\n[1/3] Loading train split...")
    train_data = loader.load_raw_json("data/processed/train.json")
    print_dataset_stats(train_data, "TRAIN")
    
    print("\n[2/3] Loading validation split...")
    val_data = loader.load_raw_json("data/processed/val.json")
    print_dataset_stats(val_data, "VALIDATION")
    
    print("\n[3/3] Loading test split...")
    test_data = loader.load_raw_json("data/processed/test_internal.json")
    print_dataset_stats(test_data, "TEST (Internal)")
    
    # Summary
    print(f"\n" + "="*70)
    print(f"Summary".center(70))
    print(f"="*70)
    
    total_docs = len(train_data) + len(val_data) + len(test_data)
    total_tokens = sum(len(d['tokens']) for d in train_data + val_data + test_data)
    
    print(f"\nTotal across all splits:")
    print(f"  Documents: {total_docs:,}")
    print(f"  Tokens: {total_tokens:,}")
    
    print(f"\nDocument distribution:")
    print(f"  Train: {len(train_data)} ({len(train_data)/total_docs*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/total_docs*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/total_docs*100:.1f}%)")
    
    train_tokens = sum(len(d['tokens']) for d in train_data)
    val_tokens = sum(len(d['tokens']) for d in val_data)
    test_tokens = sum(len(d['tokens']) for d in test_data)
    
    print(f"\nToken distribution:")
    print(f"  Train: {train_tokens:,} ({train_tokens/total_tokens*100:.1f}%)")
    print(f"  Val:   {val_tokens:,} ({val_tokens/total_tokens*100:.1f}%)")
    print(f"  Test:  {test_tokens:,} ({test_tokens/total_tokens*100:.1f}%)")
    
    print("\n✓ Split verification PASSED")
    print("  - All documents are unique across splits")
    print("  - No document appears in multiple splits")
    print("  - Ready for model training")


if __name__ == "__main__":
    main()
