from dataclasses import replace
from tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer(method="nltk")
import nltk
import numpy as np
from nltk import ngrams
from collections import Counter
import operator
from scipy import stats
import os
from multiset_distances import MultisetDistances
import mauve
import torch
import tqdm

def bleu(refs, cands):
    result = {}
    for i in range(1, 5):
        result["corpus-bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu(list_of_references=[[r] for r in refs], hypotheses=cands, weights=tuple([1./i for j in range(i)])))
    for i in range(1, 5):
        result["r-corpus-bleu-%d"%i] = "%.4f"%(nltk.translate.bleu_score.corpus_bleu(list_of_references=[[c] for c in cands], hypotheses=refs, weights=tuple([1./i for j in range(i)])))
    for i in range(1, 5):
        result["sent-bleu-%d"%i] = []
        for r, c in zip(refs, cands):
            result["sent-bleu-%d"%i].append(nltk.translate.bleu_score.sentence_bleu(references=[r], hypothesis=c, weights=tuple([1./i for j in range(i)])))
        result["sent-bleu-%d"%i] = "%.4f"%np.mean(result["sent-bleu-%d"%i])
    return result

def mauve_score(inputs, refs, cands, device):
    input_refs = [i+" "+r for i, r in zip(inputs, refs)]
    input_cands = [i+" "+c for i, c in zip(inputs, cands)]
    score = {}
    model_path = "/data/guanjian/transformers_model/gpt2"
    out = mauve.compute_mauve(p_text=input_refs, q_text=input_cands, device_id=int(device[-1]), max_text_length=512, featurize_model_name=model_path, verbose=False)
    score["mauve_score_%d"%(512)] = out.mauve
    return score

def distinct(name, cands):
    result = {}
    for i in range(1, 6):
        all_ngram, all_ngram_num = {}, 0.
        for k, cand in enumerate(cands):
            ngs = ["_".join(c) for c in ngrams(cand, i)]
            all_ngram_num += len(ngs)
            for s in ngs:
                if s in all_ngram:
                    all_ngram[s] += 1
                else:
                    all_ngram[s] = 1
        result["distinct-%d"%i] = "%.4f"%(len(all_ngram) / float(all_ngram_num))
    return result

def length(cands, name):
    length = []
    for c in cands:
        length.append(len(c))
    return {"length": "%.4f"%np.mean(length)}

def msj(refs, cands):
    ref_avg_len = 0
    hyp_avg_len = 0
    for line in refs:
        ref_avg_len += len(line)
    ref_avg_len /= len(refs)
    for line in cands:
        hyp_avg_len += len(line)
    hyp_avg_len /= len(cands)

    # print("Reference avg length: {}".format(ref_avg_len))
    # print("Hypothesis avg length: {}".format(hyp_avg_len))

    msd = MultisetDistances(references=refs, min_n=1, max_n=5)
    msj_distance = msd.get_jaccard_score(sentences=cands)
    # print("MSJ distance: {}".format(msj_distance))
    tmp_result = {}
    for k in msj_distance:
        tmp_result["msj-%d"%k] = msj_distance[k]
    return tmp_result

def zipf(cands):
    cnt = Counter()
    for tokens in cands:
        cnt.update(tokens)

    xs = np.arange(1, min(len(cnt), 5000)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:5000])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    return {"zipf": -a}

def tok_repeat_l(cands, device):
    def build_dict(texts):
        t2i = {}
        for text in texts:
            for tok in text:
                if tok not in t2i:
                    t2i[tok] = len(t2i)
        return t2i
    metrics = {}
    for c_len in [16,32,64]:
        metrics.update({f"tok_repeat_{c_len}": 0.0})

    dictionary = build_dict(cands)
    for hyp in cands:
        hyp_id = []
        for tok in hyp:
            hyp_id.append(dictionary[tok])

        for c_len in [16,32,64]:
            hypo = torch.tensor(hyp_id).long().to(device)
            T = hypo.size(0)
            
            # prev_hypo[t, :] = [y_1, y_2, ... , y_t-1, -1 ,-1, ... , -1]
            prev_hypo = hypo.expand(T, T).masked_fill(torch.ones(T, T).triu().bool().to(device), -1)

            # prev_hypo[t, :] = [-1, ... , -1, y_t-k-1, ..., y_t-1, -1 ,-1, ... , -1]
            prev_hypo = prev_hypo.masked_fill(torch.ones(T, T).tril(-c_len).bool().to(device), -1)

            repeat = (hypo[:, None] == prev_hypo)
            has_repeat = repeat.sum(1).gt(0)
            total_repeat = has_repeat.sum().cpu()

            metrics[f"tok_repeat_{c_len}"] += total_repeat * 1.0 / T 
    for k, v in metrics.items():
        metrics[k] = "%.4f"%float((v * 1.0 / len(cands)).numpy())
    return metrics

def pro(s, name=""):
    s = s.strip().replace("</s>", "")
    ss = []
    for ts in s.split("<mask>"):
        if len(ts.strip()):
            ss.append(ts.strip())
        if len(ss) >= 15:
            break
    s = " ".join(" ".join(ss).split())
    s = tokenizer.convert_tokens_to_sentence(s.strip().split())
    return " ".join(s.strip().split())

import sys
device = "cuda:%s"%sys.argv[1]
data_name = sys.argv[2]
result_list_wiki = [
    "../stage2/result/model_sample_stage2_merge",
]
result_list_cnn = [
    "../stage2/result/model_sample_stage2_merge",
]

if data_name == "wiki":
    data_dir = "../data/wikiplots/data_for_bart/"
    result_list = result_list_wiki
elif data_name == "cnn":
    data_dir = "../data/cnn/data_for_bart/"
    result_list = result_list_cnn
with open("%s/test.source"%(data_dir), "r", encoding="utf-8") as fin:
    ipt = [pro(line) for line in fin][:1000]
with open("%s/test.target"%(data_dir), "r", encoding="utf-8") as fin:
    truth = [pro(line) for line in fin][:1000]

def get_result(name, ipt, truth, cand):
    result = {}
    cand_sen = []
    for c in cand:
        cand_sen.append([cc.strip() for cc in c.split("<mask>") if len(cc.strip())])
    cand = [pro(c) for c in cand]

    ipt_token, truth_token, cand_token = [tokenizer.tokenize(i) for i in ipt], [tokenizer.tokenize(t) for t in truth], [tokenizer.tokenize(c) for c in cand]

    result.update(bleu(truth_token, cand_token))
    result.update(msj(truth_token, cand_token))
    result.update(mauve_score(ipt, truth, cand, device=device))
    result.update(distinct(name, cand_token))
    result.update(tok_repeat_l(cand_token, device=device))
    result.update(length(cand_token, name))
    result.update(zipf(cand_token))
    key = sorted(result.keys())
    for k in key:
        print(name, k, result[k])
    print("="*10)
    return result
all_result = {}
for name in result_list:
    if os.path.isdir("./%s"%name):
        name_list = []
        for _, _, fl in os.walk("./%s"%name):
            for f in fl:
                name_list.append(os.path.join("%s"%name, f.split(".")[0]))
            break
        name_list = list(sorted(name_list))
    else:
        name_list = [name]
    for name in name_list:
        cand = []
        with open("%s.txt"%name, "r", encoding="utf-8") as fin:
            for line in fin:
                cand.append(line.strip().split("|||")[-1])
        cand = cand[:1000]
        result = get_result(name, ipt, truth, cand)
        all_result[name] = result
