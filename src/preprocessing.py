"""
Preprocessing untuk PII Shield.

PENTING: Preprocessing harus ringan karena PII detection sensitif terhadap informasi.
- Lowercasing, punctuation removal, stopword removal harus OPSIONAL per model
- Default: hanya tokenization dan whitespace normalization

Setiap model bisa customize preprocessing sesuai kebutuhan.
"""

import re
import json
from typing import List, Dict, Tuple


class Preprocessor:
    """Lightweight preprocessing untuk PII task."""
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Remove extra whitespace dan normalize newlines.
        
        Aman untuk PII: tidak menghilangkan informasi penting.
        """
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove extra spaces (tapi jaga single spaces)
        text = re.sub(r' +', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    @staticmethod
    def normalize_html_entities(text: str) -> str:
        """
        Normalize HTML entities (e.g., &amp; → &).
        
        Aman untuk PII: email dan URL bisa mengandung entities.
        """
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        return text
    
    @staticmethod
    def basic_preprocessing(text: str) -> str:
        """
        Preprocessing ringan yang aman untuk PII.
        """
        text = Preprocessor.normalize_html_entities(text)
        text = Preprocessor.normalize_whitespace(text)
        return text
    
    @staticmethod
    def lowercase(text: str) -> str:
        """
        Lowercase semua text.
        
        ⚠️ OPSIONAL: Gunakan hanya jika model spesifik butuh ini.
        Nama orang dan email sering sensitif terhadap case.
        """
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """
        Remove punctuation.
        
        ⚠️ OPSIONAL: BERBAHAYA untuk PII!
        Email (@, .), URL (/, :, .), nama ('), alamat (-, .) butuh punctuation.
        """
        # Keep @ . / : - ' karena penting untuk PII
        text = re.sub(r'[^\w\s@./:\'\\-]', '', text)
        return text
    
    @staticmethod
    def remove_special_chars(text: str) -> str:
        """
        Remove special characters.
        
        ⚠️ OPSIONAL: SANGAT BERBAHAYA untuk PII!
        Jangan gunakan ini untuk PII detection.
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def preprocess_dataset(
    data: List[Dict],
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_special: bool = False
) -> List[Dict]:
    """
    Preprocess seluruh dataset.
    
    Args:
        data: List of documents
        lowercase: Apakah lowercase text (default: False, opsional per model)
        remove_punctuation: Apakah remove punctuation (default: False, BERBAHAYA untuk PII)
        remove_special: Apakah remove special chars (default: False, SANGAT BERBAHAYA)
        
    Returns:
        Preprocessed data
    """
    preprocessor = Preprocessor()
    processed_data = []
    
    for doc in data:
        processed_doc = doc.copy()
        full_text = doc['full_text']
        
        # Always do basic preprocessing
        full_text = preprocessor.basic_preprocessing(full_text)
        
        # Optional preprocessing (hati-hati dengan default)
        if lowercase:
            full_text = preprocessor.lowercase(full_text)
        
        if remove_punctuation:
            print("⚠️  WARNING: remove_punctuation dapat merusak PII information (email, URL, etc.)")
            full_text = preprocessor.remove_punctuation(full_text)
        
        if remove_special:
            print("⚠️  WARNING: remove_special_chars sangat BERBAHAYA untuk PII detection")
            full_text = preprocessor.remove_special_chars(full_text)
        
        processed_doc['full_text'] = full_text
        processed_data.append(processed_doc)
    
    return processed_data


if __name__ == "__main__":
    # Test preprocessing
    test_text = "My email is john@example.com & my phone is +1-234-567-8900"
    
    print("Original:")
    print(f"  {test_text}")
    
    print("\nAfter basic preprocessing:")
    processed = Preprocessor.basic_preprocessing(test_text)
    print(f"  {processed}")
    
    print("\nAfter basic + lowercase:")
    processed = Preprocessor.basic_preprocessing(test_text)
    processed = Preprocessor.lowercase(processed)
    print(f"  {processed}")
    
    print("\n⚠️  Danger zone - jangan gunakan untuk PII:")
    print("  remove_punctuation akan menghapus @ dan . dari email!")
    print("  remove_special_chars akan menghapus hampir semua yang bukan alphanumeric!")
