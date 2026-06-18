"""
Feature extraction untuk Boosting Models (XGBoost & LightGBM).
Mengekstrak fitur token-level dari data PII untuk token classification.

Features:
- Character-level features (length, case, digit, special chars)
- Regex pattern matching (email, URL, phone, ID, zip, username, street)
- Keyword distance features (jarak ke keyword terdekat dalam window)
- Colon pattern features
- Context features (fitur dari token tetangga)
- Character n-gram features via HashingVectorizer
"""

import re
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, csr_matrix


# ============================================================
# Regex Patterns
# ============================================================
RE_EMAIL     = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_URL       = re.compile(r"https?://\S+|www\.\S+")
RE_PHONE     = re.compile(r"(\+?\d[\d\s\-().]{6,}\d)")
RE_PHONE2    = re.compile(r"^\+?\d{7,15}$")
RE_ID_NUM    = re.compile(r"\b\d{5,}\b")
RE_USERNAME  = re.compile(r"^@\w+$|^\w{3,20}$")
RE_USERNAME2 = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.]{2,19}\d+$")
RE_ZIP       = re.compile(r"\b\d{5}(?:-\d{4})?\b")
RE_STREET_KW = re.compile(
    r"^(jl|jalan|street|st|ave|avenue|rd|road|blvd|boulevard|drive|dr|lane|ln|apt|suite|no)\.?$",
    re.IGNORECASE
)

# ============================================================
# Keyword Groups (untuk keyword distance features)
# ============================================================
KEYWORD_GROUPS = {
    "email_kw":    {"email", "e-mail", "mail", "contact"},
    "phone_kw":    {"phone", "telp", "telepon", "hp", "mobile", "call", "contact", "whatsapp", "wa"},
    "name_kw":     {"name", "nama", "by", "author", "written", "student", "i", "i'm", "im", "saya"},
    "address_kw":  {"address", "alamat", "street", "jalan", "live", "located", "location"},
    "username_kw": {"username", "user", "handle", "account", "profile", "ig", "instagram", "twitter"},
    "id_kw":       {"id", "nim", "nik", "number", "no", "identity", "student"},
}


def keyword_distance_features(tokens, idx, window=8):
    """
    Hitung fitur jarak ke keyword group terdekat dalam window.
    
    Returns:
        List[float]: satu nilai per keyword group (0.0 = jauh, 1.0 = dekat)
    """
    feats = []
    for group_name, kws in KEYWORD_GROUPS.items():
        min_dist = window + 1
        for d in range(1, window + 1):
            j = idx - d
            if j < 0:
                break
            if tokens[j].lower().strip(":,.") in kws:
                min_dist = d
                break
        feats.append(1.0 - (min_dist / (window + 1)))
    return feats


def colon_pattern_feature(tokens, idx):
    """
    Cek apakah token sebelumnya diakhiri colon (pola "Label: Value").
    
    Returns:
        List[float]: [has_colon_before, has_colon_2back]
    """
    if idx == 0:
        return [0.0, 0.0]
    prev_tok = tokens[idx - 1]
    has_colon_before = float(prev_tok.endswith(":"))
    has_colon_2back = float(idx >= 2 and tokens[idx - 2].endswith(":"))
    return [has_colon_before, has_colon_2back]


def _char_feats(t):
    """Extract character-level features dari satu token."""
    if not t or t == "<PAD>":
        return [0] * 12
    return [
        len(t),
        float(t.isupper()),
        float(t.islower()),
        float(t.istitle()),
        float(t.isdigit()),
        float(t.isalpha()),
        float(t.isalnum()),
        float("@" in t),
        float("." in t),
        float("-" in t or "_" in t),
        float("/" in t or ":" in t),
        float(bool(re.search(r"\d", t))),
    ]


def token_features(tokens, idx):
    """
    Extract semua fitur untuk satu token pada posisi idx.
    
    Features:
    - 12 char features token utama
    - 12 regex pattern features
    - 2 prefix/suffix hash features
    - 6 * 12 = 72 context char features (window -3..+3)
    - 6 keyword distance features
    - 2 colon pattern features
    
    Total: 106 hand-crafted features
    
    Args:
        tokens: List of tokens dalam satu dokumen
        idx: Index token yang akan diekstrak fiturnya
        
    Returns:
        List[float]: vektor fitur
    """
    token = tokens[idx]
    n = len(tokens)

    def safe_token(i):
        return tokens[i] if 0 <= i < n else "<PAD>"

    ctx_positions = [-3, -2, -1, 1, 2, 3]
    ctx_tokens = [safe_token(idx + p) for p in ctx_positions]

    # Character features token utama
    f = _char_feats(token)

    # Regex pattern features
    f += [
        float(bool(RE_EMAIL.fullmatch(token))),
        float(bool(RE_URL.match(token))),
        float(bool(RE_PHONE.fullmatch(token))),
        float(bool(RE_PHONE2.fullmatch(token))),
        float(bool(RE_ID_NUM.fullmatch(token))),
        float(bool(RE_ZIP.fullmatch(token))),
        float(bool(RE_USERNAME.match(token)) and len(token) >= 3),
        float(bool(RE_USERNAME2.match(token))),
        float(bool(RE_STREET_KW.match(token))),
        float(bool(RE_URL.search(token))),
        float("@" in token and "." in token),
        float(token[0].isdigit() if token else False),
    ]

    # Prefix/suffix hash features
    prefix = token[:3].lower() if len(token) >= 3 else token.lower().ljust(3)
    suffix = token[-3:].lower() if len(token) >= 3 else token.lower().ljust(3)
    f += [hash(prefix) % 1000 / 1000, hash(suffix) % 1000 / 1000]

    # Context features (char features dari token tetangga)
    for ctx in ctx_tokens:
        f += _char_feats(ctx)

    # Keyword distance features
    f += keyword_distance_features(tokens, idx, window=8)

    # Colon pattern features
    f += colon_pattern_feature(tokens, idx)

    return f


def build_feature_matrix(data, vectorizer=None, fit_vectorizer=False):
    """
    Build feature matrix dari seluruh dataset.
    
    Menggabungkan hand-crafted features dengan character n-gram features
    dari HashingVectorizer.
    
    Args:
        data: List of documents (dict dengan keys: tokens, labels, document)
        vectorizer: HashingVectorizer yang sudah di-fit (None jika fit_vectorizer=True)
        fit_vectorizer: Jika True, buat vectorizer baru
        
    Returns:
        X: sparse feature matrix
        all_labels: list of label strings
        meta: list of (doc_id, token, true_label) tuples
        vectorizer: HashingVectorizer instance
    """
    all_hand, all_tokens, all_labels, meta = [], [], [], []

    for doc in data:
        tokens, labels, doc_id = doc["tokens"], doc["labels"], doc["document"]
        for i, (tok, lbl) in enumerate(zip(tokens, labels)):
            all_hand.append(token_features(tokens, i))
            all_tokens.append(tok.lower())
            all_labels.append(lbl)
            meta.append((doc_id, tok, lbl))

    X_hand = np.array(all_hand, dtype=np.float32)

    if fit_vectorizer:
        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=2**15,
            alternate_sign=False,
            norm="l2",
        )

    X_tfidf = vectorizer.transform(all_tokens)
    X = hstack([csr_matrix(X_hand), X_tfidf])

    return X, all_labels, meta, vectorizer


def encode_labels(labels, le=None, fit=False):
    """
    Encode string labels ke integer menggunakan LabelEncoder.
    
    Args:
        labels: List of label strings
        le: LabelEncoder instance (None jika fit=True)
        fit: Jika True, fit encoder baru
        
    Returns:
        encoded: array of integer labels
        le: LabelEncoder instance
    """
    if fit:
        le = LabelEncoder()
        le.fit(labels)
    return le.transform(labels), le


def compute_class_weights(y, power=0.5):
    """
    Hitung class weights berdasarkan inverse frequency.
    
    Args:
        y: array of integer labels
        power: eksponen untuk smoothing (0.5 = sqrt inverse frequency)
        
    Returns:
        Dict[int, float]: mapping class_id -> weight
    """
    counts = defaultdict(int)
    for label in y:
        counts[label] += 1
    total = len(y)
    raw = {cls: total / (len(counts) * cnt) for cls, cnt in counts.items()}
    return {cls: w**power for cls, w in raw.items()}


def get_sample_weights(y, class_weights):
    """
    Konversi class weights ke per-sample weights.
    
    Args:
        y: array of integer labels
        class_weights: Dict[int, float] dari compute_class_weights()
        
    Returns:
        np.array of sample weights
    """
    return np.array([class_weights[label] for label in y], dtype=np.float32)


def quick_f1_pii(true_labels, pred_labels):
    """
    Hitung quick macro F1 score untuk PII labels saja (exclude O).
    Digunakan untuk monitoring cepat saat training/tuning.
    
    Args:
        true_labels: List of true label strings
        pred_labels: List of predicted label strings
        
    Returns:
        float: macro F1 score untuk PII classes
    """
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        if t != "O":
            if t == p:
                tp[t] += 1
            else:
                fn[t] += 1
        if p != "O" and p != t:
            fp[p] += 1
    f1s = []
    for lbl in set(list(tp) + list(fn)):
        prec = tp[lbl] / (tp[lbl] + fp[lbl] + 1e-9)
        rec = tp[lbl] / (tp[lbl] + fn[lbl] + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return np.mean(f1s) if f1s else 0.0
