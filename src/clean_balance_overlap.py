import json
import os

def get_document_ids(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {doc['document'] for doc in data}

def main():
    val_path = "data/processed/balance/val.json"
    test_path = "data/processed/balance/test.json"
    train_path = "data/processed/balance/train.json"
    
    print("Memuat document IDs dari val.json dan train.json...")
    val_ids = get_document_ids(val_path)
    train_ids = get_document_ids(train_path)
    
    overlap_ids = val_ids.union(train_ids)
    print(f"Total unique docs di val.json: {len(val_ids)}")
    print(f"Total unique docs di train.json: {len(train_ids)}")
    
    print(f"\nMembaca {test_path}...")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        
    print(f"Total documents di test.json sebelum dibersihkan: {len(test_data)}")
    
    # Filter test data
    filtered_test_data = []
    removed_count = 0
    
    for doc in test_data:
        if doc['document'] in val_ids:
            removed_count += 1
        elif doc['document'] in train_ids:
            removed_count += 1
        else:
            filtered_test_data.append(doc)
            
    print(f"\nDocuments dihapus dari test.json karena overlap: {removed_count}")
    print(f"Documents tersisa di test.json: {len(filtered_test_data)}")
    
    # Save the cleaned test data
    print(f"\nMenyimpan kembali ke {test_path}...")
    with open(test_path, 'w') as f:
        json.dump(filtered_test_data, f, indent=2)
        
    print("Selesai!")

if __name__ == "__main__":
    main()
