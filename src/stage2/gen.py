from transformers import BartTokenizer
import torch
import sys
import os
import numpy as np
from transformers import BartForConditionalGeneration

device = "cuda:%s"%sys.argv[1]
print("using %s"%device)

model_name_path = sys.argv[2]
name = "../stage1/data_for_eric/1"
data_name = sys.argv[3]
with open("%s/%s.source"%(name, data_name), "r") as fin:
    ipt = [line.strip() for line in fin][:1000]
target_name = sys.argv[4]
output_suffix = sys.argv[5]

def pro(token_list, tokenizer):
    string = tokenizer.decode(token_list, skip_special_tokens=False)
    string = string.replace("<s>", " ")
    string = string[:string.find("</s>")].strip()
    return string

from tokenizer import SimpleTokenizer
nltk_tokenizer = SimpleTokenizer(method="nltk")
tokenizer = BartTokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
model = BartForConditionalGeneration.from_pretrained(model_name_path, return_dict=True).to(device)
model.eval()

with open("./%s.txt"%(target_name), encoding="utf-8") as fin:
    gen_data = ["<mask>".join([nltk_tokenizer.convert_tokens_to_sentence(s.strip().split()) for s in line.strip().split("<mask>")]) + "<mask>" for line in fin]
file_out = "%s_%s.txt"%(target_name, output_suffix)
print("write to %s"%file_out)
with open(file_out, "w", encoding="utf-8") as fout:
    batch_size = 6
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            tmpipt = ["%s|||%s"%(iipt, ggen) for iipt, ggen in zip(ipt[st:ed], gen_data[st:ed])]
            input_ids = tokenizer(tmpipt, return_tensors="pt", padding=True, truncation=True, max_length=768)
            attention_mask = input_ids.attention_mask.to(device)
            input_ids = input_ids.input_ids.to(device)
            gen = model.generate(input_ids, 
                attention_mask=attention_mask, 
                do_sample=True, 
                num_beams=1, 
                top_p=0.9,
                decoder_start_token_id=0, 
                max_length=512, 
                early_stopping=True)
            for ip, op in zip(input_ids, gen):
                print(pro(ip, tokenizer) + "|||" + pro(op, tokenizer))
                fout.write(pro(op, tokenizer)+"\n")
