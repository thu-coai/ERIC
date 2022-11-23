import spacy

nlp = spacy.load("en_core_web_sm")
# doc = nlp("Cole arrives in Baltimore in 1990, not 1996 as planned.")

# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)

for name in ["train", "val", "test"]:
    with open("../data_for_bart/%s.source"%name, encoding='utf-8') as fin1:
        with open("../data_for_bart/%s.target"%name, encoding='utf-8') as fin2:
            fo1 = open("./1/%s.source"%name, "w", encoding='utf-8')
            fo2 = open("./1/%s.target"%name, "w", encoding='utf-8')
            fo3 = open("./2/%s.source"%name, "w", encoding='utf-8')
            fo4 = open("./2/%s.target"%name, "w", encoding='utf-8')

            for l1, l2 in zip(fin1, fin2):
                ipt = l1.strip()
                opt = l2.strip().replace("</s>", "")
                senlist = opt.split("<mask>")[1:-1]
                allent = {}
                id_ = 0
                sen = opt.strip().replace("<mask>", " ||| ")
                tmpdoc = nlp(sen)
                st = 0
                tmpsen = ""
                optsen = ""
                for ent in tmpdoc.ents:
                    print(ent.text, ent.start_char, ent.end_char, ent.label_)
                    if ent.label_ == "PERSON" or ent.label_ == "ORG":
                        # entlist.append([ent.text, ent.start_char, ent.end_char])
                        if ent.text not in allent:
                            flag = 0
                            for key in allent:
                                if ent.text in key or key in ent.text:
                                    flag = 1
                                    allent[ent.text] = allent[key]
                                    break
                            if flag == 0:
                                allent[ent.text] = "<p%d>"%id_
                                id_ += 1
                        tmpsen += sen[st:ent.start_char] + allent[ent.text]
                        optsen += allent[ent.text] + ent.text
                        st = ent.end_char
                tmpsen += sen[st:]
                tmpsen = tmpsen.replace(" ||| ", "<mask>").strip()
                print(sen)
                print(tmpsen)
                print("="*10)
                fo1.write("%s\n"%ipt.strip())
                fo2.write("%s</s><mask>\n"%tmpsen[:-6].strip())
                if len(optsen.strip()):
                    fo3.write("%s|||%s\n"%(ipt.strip(), tmpsen.strip()))
                    fo4.write("%s\n"%(optsen.strip()))