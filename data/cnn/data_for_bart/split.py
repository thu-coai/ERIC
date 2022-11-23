import re
import nltk
from tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer(method="nltk")
import json
data = []
with open("stories.txt") as fin:
    for k, line in enumerate(fin):
        if k % 1000 == 0:
            print(k, len(data))
        tmp = json.loads(line.strip())["story"]
        tmp = " ".join(" ".join(tmp).split())
        tmp = nltk.tokenize.sent_tokenize(tmp)

        newtmp = []
        for t in tmp:
            if len(t.strip()):
                newtmp.append(
                    tokenizer.convert_tokens_to_sentence(tokenizer.tokenize(t.strip()))
                )
        if len(newtmp) > 5:
             data.append({
                "src": newtmp[0],
                "tgt": "<mask>" + "<mask>".join(newtmp[1:16]) + "</s><mask>",
             })
import numpy as np
np.random.seed(42)
np.random.shuffle(data)
print(len(data))
split = int(len(data) / 20)
with open("train.source", "w", encoding="utf-8") as f1:
    with open("train.target", "w", encoding="utf-8") as f2:
        for d in data[:split*18]:
            f1.write("%s\n"%d["src"].strip())
            f2.write("%s\n"%d["tgt"].strip())
with open("val.source", "w", encoding="utf-8") as f1:
    with open("val.target", "w", encoding="utf-8") as f2:
        for d in data[split*18:split*19]:
            f1.write("%s\n"%d["src"].strip())
            f2.write("%s\n"%d["tgt"].strip())
with open("test.source", "w", encoding="utf-8") as f1:
    with open("test.target", "w", encoding="utf-8") as f2:
        for d in data[split*19:]:
            f1.write("%s\n"%d["src"].strip())
            f2.write("%s\n"%d["tgt"].strip())