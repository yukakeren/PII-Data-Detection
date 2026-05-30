"""
Data loader untuk PII Shield project.
Menghandle dataset Kaggle dalam format JSON.
"""

import json
import os
from typing import List, Dict, Tuple
from pathlib import Path


class DataLoader:
    """Load dan save dataset PII dari format Kaggle JSON."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize DataLoader.
        
        Args:
            config_path: Path ke label_schema.json (opsional)
        """
        self.config_path = config_path or "configs/label_schema.json"
        self.label_schema = self._load_label_schema()
        self.id_to_label = {v: k for k, v in self.label_schema.items()}
    
    def _load_label_schema(self) -> Dict[str, int]:
        """Load label schema dari JSON."""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def load_raw_json(self, filepath: str) -> List[Dict]:
        """
        Load raw Kaggle JSON dataset.
        
        Args:
            filepath: Path ke file train.json atau test.json
            
        Returns:
            List of documents dengan keys: document, tokens, labels, full_text, trailing_whitespace
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    
    def save_processed_json(self, data: List[Dict], output_path: str) -> None:
        """
        Save processed dataset sebagai JSON.
        
        Args:
            data: List of processed documents
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_processed_csv(self, data: List[Dict], output_path: str) -> None:
        """
        Save processed dataset sebagai CSV (format: document_id, token, true_label, pred_label).
        Untuk label true, gunakan true_label. Untuk pred_label, isi dengan true_label dulu.
        
        Args:
            data: List of processed documents
            output_path: Output file path
        """
        import csv
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['document_id', 'token', 'true_label', 'pred_label'])
            writer.writeheader()
            
            for doc in data:
                doc_id = doc['document']
                tokens = doc['tokens']
                labels = doc['labels']
                
                for token, label in zip(tokens, labels):
                    writer.writerow({
                        'document_id': doc_id,
                        'token': token,
                        'true_label': label,
                        'pred_label': label  # Initially same as true_label
                    })
    
    def get_label_id(self, label: str) -> int:
        """Get ID untuk label string."""
        return self.label_schema.get(label, 0)
    
    def get_label_name(self, label_id: int) -> str:
        """Get label string dari ID."""
        return self.id_to_label.get(label_id, 'O')


def load_raw_data(filepath: str = "train.json") -> List[Dict]:
    """Convenience function: load raw Kaggle data."""
    loader = DataLoader()
    return loader.load_raw_json(filepath)


def load_processed_data(filepath: str) -> List[Dict]:
    """Convenience function: load processed data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_processed_data(data: List[Dict], output_path: str) -> None:
    """Convenience function: save processed data."""
    loader = DataLoader()
    loader.save_processed_json(data, output_path)


if __name__ == "__main__":
    # Test: load and inspect data
    loader = DataLoader()
    
    print("Loading train.json...")
    train_data = loader.load_raw_json("train.json")
    print(f"Total documents: {len(train_data)}")
    
    print(f"\nFirst document:")
    doc = train_data[0]
    print(f"  Document ID: {doc['document']}")
    print(f"  Number of tokens: {len(doc['tokens'])}")
    print(f"  Number of labels: {len(doc['labels'])}")
    print(f"  First 5 tokens: {doc['tokens'][:5]}")
    print(f"  First 5 labels: {doc['labels'][:5]}")
