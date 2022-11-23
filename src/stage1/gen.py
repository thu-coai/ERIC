from transformers import BartTokenizer
import torch
import sys
import os
import numpy as np
from modeling_bart import BartForConditionalGeneration
device = "cuda:%s"%sys.argv[1]
print("using %s"%device)

model_name_path = sys.argv[2]
print(model_name_path)
name = "./data_for_eric/1"
data_name = sys.argv[3]
with open("%s/%s.source"%(name, data_name), "r") as fin:
    ipt = [line.strip() for line in fin][:1000]
with open("%s/%s.target"%(name, data_name), "r") as fin:
    opt = [line.strip() for line in fin][:1000]

def pro(token_list, tokenizer):
    string = tokenizer.decode(token_list, skip_special_tokens=False)
    string = string.replace("<s>", " ")
    string = string[:string.find("</s>")].strip()
    string = " ".join(string.strip().split()).strip()
    return string
tokenizer = BartTokenizer.from_pretrained(model_name_path)
pad_token_id = tokenizer.pad_token_id
model = BartForConditionalGeneration.from_pretrained(model_name_path, return_dict=True).to(device)
model.eval()

file_out = "./result/%s_sample.txt"%(model_name_path.replace(".", "").replace("/", ""))
print("write to %s"%file_out)
with open(file_out, "w", encoding="utf-8") as fout:
    batch_size = 24
    st, ed = 0, 0
    all_loss = []
    with torch.no_grad():
        while ed < len(ipt):
            st, ed = ed, (ed + batch_size) if (ed + batch_size < len(ipt)) else len(ipt)
            input_ids = tokenizer(ipt[st:ed], return_tensors="pt", padding=True, truncation=True, max_length=512)
            attention_mask = input_ids.attention_mask.to(device)
            input_ids = input_ids.input_ids.to(device)
            gen = model.generate(input_ids, 
                    attention_mask=attention_mask, 
                    do_sample=True, 
                    top_p=0.9, 
                    num_beams=1,
                    decoder_start_token_id=0, 
                    max_length=512, 
                    early_stopping=False, 
                    use_cache=True)
            for ip, op in zip(input_ids, gen):
                print(pro(ip, tokenizer) + "|||" + pro(op, tokenizer))
                fout.write(pro(op, tokenizer)+"\n")