import json

for name in ["train", "val", "test"]:
    fout1 = open("./%s.source"%name.replace("val", "val"), "w")
    fout2 = open("./%s.target"%name.replace("val", "val"), "w")
    with open("../source/wikiplots_%s_disc.json"%name) as fin:
        data = json.load(fin)
        for d in data:
            d = d["edus"]
            if len(d) > 5:
                src = d[0]
                tgt = "<mask>" + "<mask>".join(d[1:16]) + "</s><mask>"
                fout1.write("%s\n"%src)
                fout2.write("%s\n"%tgt)
