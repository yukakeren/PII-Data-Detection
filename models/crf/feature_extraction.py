import re
import string


def word_shape(token):
    """
    Convert token into shape pattern
    Example:
    Farras -> Xxxxxx
    ITS2025 -> XXXdddd
    """
    shape = ""

    for char in token:
        if char.isupper():
            shape += "X"
        elif char.islower():
            shape += "x"
        elif char.isdigit():
            shape += "d"
        else:
            shape += char

    return shape


def is_email(token):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, token))


def is_url(token):
    pattern = r"(http|https|www\.)"
    return bool(re.search(pattern, token.lower()))


def is_phone(token):
    digits = re.sub(r"\D", "", token)
    return len(digits) >= 8


def token_features(tokens, i):
    token = tokens[i]

    features = {
        "bias": 1.0,

        # token info
        "token.lower": token.lower(),
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.len": len(token),

        # prefix suffix
        "prefix1": token[:1],
        "prefix2": token[:2],
        "prefix3": token[:3],
        "suffix1": token[-1:],
        "suffix2": token[-2:],
        "suffix3": token[-3:],

        # pattern
        "word_shape": word_shape(token),

        # punctuation
        "is_punct": token in string.punctuation,

        # regex based features
        "has_email_pattern": is_email(token),
        "has_url_pattern": is_url(token),
        "has_phone_pattern": is_phone(token),
        "contains_at": "@" in token,
        
        "has_digit": any(c.isdigit() for c in token),
        "all_caps": token.isupper(),
        "title_case": token.istitle(),
        "is_alpha": token.isalpha(),
        "contains_dash": "-" in token,
        "contains_dot": "." in token,
    }

    # previous 2 tokens
    for offset in [1, 2]:
        if i - offset >= 0:
            prev = tokens[i - offset]
            features[f"-{offset}:token.lower"] = prev.lower()
            features[f"-{offset}:isupper"] = prev.isupper()
            features[f"-{offset}:shape"] = word_shape(prev)
        else:
            features[f"BOS-{offset}"] = True

    # next 2 tokens
    for offset in [1, 2]:
        if i + offset < len(tokens):
            nxt = tokens[i + offset]
            features[f"+{offset}:token.lower"] = nxt.lower()
            features[f"+{offset}:isupper"] = nxt.isupper()
            features[f"+{offset}:shape"] = word_shape(nxt)
        else:
            features[f"EOS+{offset}"] = True

    return features


def sentence_features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]