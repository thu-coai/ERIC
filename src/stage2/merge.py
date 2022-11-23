from re import T
from tokenizer import SimpleTokenizer
tokenizer = SimpleTokenizer(method="nltk")
import numpy as np
import string
punct = string.punctuation
allp = ["<p%d>"%k for k in range(100)]
import sys
src_name = sys.argv[1]
output_suffix = sys.argv[2]

with open("%s.txt"%(src_name), encoding="utf-8") as fin:
    gen_data = [line.strip().replace(">", "> ").replace("<", " <").split() for line in fin]
allname = {}
with open("%s_%s.txt"%(src_name, output_suffix), encoding="utf-8") as fin:
    fout = open("%s_%s_merge.txt"%(src_name, output_suffix), "w", encoding="utf-8")
    for k, line in enumerate(fin):
        line = line.replace(">", "> ").replace("<", " <")
        tmpdict = {}
        tmp = line.strip().split() + ["<p99>"]
        last_w, last_str = "", ""
        for w in tmp:
            if w in allp:
                last_w, last_str = last_w.strip(), last_str.strip()
                if last_w in tmpdict:
                    tmpdict[last_w].append(last_str)
                else:
                    if last_w != "":
                        tmpdict[last_w] = [last_str]
                if " " not in last_str.strip():
                    if last_str in allname:
                        allname[last_str] += 1
                    else:
                        allname[last_str] = 1
                last_w, last_str = w.strip(), ""
            else:
                last_str += " " + w.strip()
        new_tmpdict = {}
        for w in tmpdict:
            flag = False
            new_tmpdict_w = []
            for w1 in tmpdict[w]:
                tmpflag = True
                for w2 in tmpdict[w]:
                    if w2 not in w1:
                        tmpflag = False
                        break
                if tmpflag:
                    flag = True
                    break
            if flag:
                for ww in new_tmpdict:
                    if len(set(tmpdict[w]) & set(new_tmpdict[ww])):
                        flag = False
                        break
            if flag or (len(tmpdict[w]) == 0):
                new_tmpdict[w] = list(tmpdict[w])
            else:
                flag_one_word = True
                for ent in tmpdict[w]:
                    if " " in ent.strip():
                        flag_one_word = False
                        break
                if flag_one_word:
                    tmp_name = ""
                    for ent in tmpdict[w]:
                        flag = True
                        for pun in punct:
                            if pun in ent:
                                flag = False
                                break
                        for ww in new_tmpdict:
                            if ent in new_tmpdict[ww]:
                                flag = False
                        if flag:
                            tmp_name = ent
                            break
                    if len(tmp_name) == 0:
                        if len(allname):
                            tmp_name = np.random.choice(list(allname.keys()))
                        else:
                            tmp_name = np.random.choice(tmpdict[w])
                    new_tmpdict[w] = [www if " " in www.strip() else tmp_name for www in tmpdict[w]]
                else:
                    new_tmpdict[w] = list(tmpdict[w])
        new_tmpdict = tmpdict
        pointer_dict = {}
        for pp in range(100):
            pointer_dict["<p%d>"%pp] = 0
        new_gen_data = []
        for word in gen_data[k]:
            if word in allp:
                try:
                    new_gen_data.append(new_tmpdict[word][pointer_dict[word]])
                except:
                    try:
                        new_gen_data.append(new_tmpdict[word][-1])
                    except:
                        if len(allname):
                            new_gen_data.append(np.random.choice(list(allname.keys())))
                        else:
                            new_gen_data.append(np.random.choice([nn for ww in new_tmpdict for nn in new_tmpdict[ww]]))
                pointer_dict[word] += 1
            else:
                new_gen_data.append(word)
        new_gen = " ".join(new_gen_data)
        s = "<mask>".join([ts.strip() for ts in new_gen.split("<mask>")])
        fout.write("%s\n"%(s.strip()))