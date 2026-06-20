import re
import string


def word_shape(token):
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
    return bool(
        re.match(
            r"^[\w\.-]+@[\w\.-]+\.\w+$",
            token
        )
    )


def is_url(token):
    return bool(
        re.search(
            r"(http|https|www\.)",
            token.lower()
        )
    )


def is_phone(token):
    return bool(
        re.match(
            r"^[+]?[0-9()\-\s]+$",
            token
        )
    )


def token_features(tokens, i):

    token = tokens[i]

    features = {
        "bias": 1.0,

        # basic
        "token.lower": token.lower(),
        "token.len": len(token),
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.isalpha": token.isalpha(),

        # prefixes suffixes
        "prefix1": token[:1],
        "prefix2": token[:2],
        "prefix3": token[:3],

        "suffix1": token[-1:],
        "suffix2": token[-2:],
        "suffix3": token[-3:],

        # shape
        "shape": word_shape(token),

        # symbols
        "contains_digit":
            any(c.isdigit() for c in token),

        "contains_at":
            "@" in token,

        "contains_dot":
            "." in token,

        "contains_dash":
            "-" in token,

        "contains_slash":
            "/" in token,

        "is_punct":
            token in string.punctuation,

        # regex feature
        "is_email":
            is_email(token),

        "is_url":
            is_url(token),

        "is_phone":
            is_phone(token),

        "looks_username":
            token.startswith("@"),

        "all_caps":
            token.isupper(),

        "title_case":
            token.istitle(),
    }

    # previous token
    if i > 0:
        prev = tokens[i - 1]

        features.update({
            "-1:lower": prev.lower(),
            "-1:shape": word_shape(prev),
            "-1:istitle": prev.istitle(),

            "-1:is_email":
                is_email(prev),

            "-1:is_url":
                is_url(prev),

            "-1:is_phone":
                is_phone(prev),
        })
    else:
        features["BOS"] = True

    # next token
    if i < len(tokens) - 1:
        nxt = tokens[i + 1]

        features.update({
            "+1:lower": nxt.lower(),
            "+1:shape": word_shape(nxt),
            "+1:istitle": nxt.istitle(),

            "+1:is_email":
                is_email(nxt),

            "+1:is_url":
                is_url(nxt),

            "+1:is_phone":
                is_phone(nxt),
        })
    else:
        features["EOS"] = True

    return features


def sentence_features(tokens):
    return [
        token_features(tokens, i)
        for i in range(len(tokens))
    ]