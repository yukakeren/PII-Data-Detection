from collections import Counter
import json

with open("data/raw/COMPETITION_FOLD_0.json") as f:
    data = json.load(f)

counter = Counter()

for doc in data:
    counter.update(doc["labels"])

for k, v in sorted(counter.items()):
    print(k, v)