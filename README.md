# ERIC

A narrative generation model equipped with dynamic and discrete **E**ntity state **R**epresentat**I**ons, which are learned through a **C**ontrastive framework. More details can be found in [Generating Coherent Narratives by Learning Dynamic and Discrete Entity States with a Contrastive Framework](https://arxiv.org/abs/2208.03985) (AAAI 2023 Long Paper).



## Prerequisites

The code is written in PyTorch library. Main dependencies are as follows:

- Python: 3.6.9
- torch: 1.8.1
- transformers: 4.6.1

Other dependencies can be found in `requirements.txt`



## Computing infrastructure

We train HINT based on the platform: 

- OS: Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-98-generic x86_64)
- CUDA Version: 10.1
- GPU: NVIDIA Tesla V100



## Quick Start

###1. Datasets

The full data can be downloaded from [THUcloud](https://cloud.tsinghua.edu.cn/f/3423fcf320a34447a07e/?dl=1).

- **Wikiplots**
  - **Source**: 
    - `./data/wikiplots/source/plots.zip`, which is from the [Github Repo](https://github.com/markriedl/WikiPlots).
    - `./data/wikiplots/source/wikiplots_xxx_disc.json` (`xxx` is one of `train/val/test`), which is from the [DiscoDVT paper](https://aclanthology.org/2021.emnlp-main.347/) and contains annotations of elementary discourse units (EDU). 
  - **Data for BART-style Models**: saved under the `./data/wikiplots/data_for_bart/` directory.
    - **Input**: `xxx.source` (Each line is a piece of example)
    - **Output**: `xxx.target`
    - **How to construct the data**: `cd ./data/wikiplots/data_for_bart/ && python3 pro.py`
  - **Data for ERIC**: saved under the `./data/wikiplots/data_for_eric` directory.
    - **Data for the 1st stage of ERIC**: `./data/wikiplots/data_for_eric/1`
      - **Input**: `xxx.source` 
      - **Output**: `xxx.target`
    - **Data for the 2nd stage of ERIC**: `./data/wikiplots/data_for_eric/1`
      - **Input**: `xxx.source` 
      - **Output**: `xxx.target`
    - **How to construct the data**:  `cd ./data/wikiplots/data_for_eric/ && python3 pro.py`
- **CNN News**:
  - **Source**: `./data/source/cnn_stories.tgz`, which is from [the address](https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ).
  - **Data for BART-style Models**: saved under the `./data/cnn/data_for_bart/` directory.
    - **How to construct the data**: `cd ./data/cnn/data_for_bart/ && python3 pro_raw_data.py && python3 split.py` (The processing code `pro_raw_data.py` is adapted from the [Github Rep](https://github.com/tanyuqian/progressive-generation))
  - **Data for ERIC**: saved under the `./data/cnn/data_for_eric/` directory.
    - **How to construct the data**:  `cd ./data/cnn/data_for_eric/ && python3 pro.py`



###2. Training ERIC

The initial checkpoint of BART can be downloaded from [HuggingFace](https://huggingface.co/facebook/bart-base/tree/main). We use the base version of BART for both training stages.

- The 1st stage: Execute the following command (or run `cd ./src/stage1 && bash ./finetune.sh` directly): 

  ```shell
  env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u finetune.py \
      --data_dir ../data_for_eric/1 \
      --output_dir=./model \
      --save_top_k 80 \
      --train_batch_size=12 \
      --eval_batch_size=12 \
      --num_train_epochs 20 \
      --model_name_or_path facebook/bart-base \
      --learning_rate=1e-4 \
      --gpus 1 \
      --do_train \
      --n_val 4000 \
      --val_check_interval 1.0 \
      --overwrite_output_dir
  ```

- The 2nd stage: Execute the following command (or run `cd ./src/stage2 && bash ./finetune.sh` directly): 

  ```shell
  env CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 -u finetune.py \
      --data_dir ../data_for_eric/2 \
      --output_dir=./model \
      --save_top_k 80 \
      --train_batch_size=12 \
      --eval_batch_size=12 \
      --num_train_epochs 20 \
      --model_name_or_path facebook/bart-base \
      --learning_rate=1e-4 \
      --gpus 1 \
      --do_train \
      --n_val 4000 \
      --val_check_interval 1.0 \
      --overwrite_output_dir
  ```

  The 2nd training stage is exactly the same as fine-tuning the standard BART model.



### 3. Inference

The generation results are provided in the `generation_results` directory. Execute the following command to generate texts: 

```shell
cd ./stage1
gpu=0 # GPU ID
model=./model # directory of the 1st-stage model 
truth=../data_for_eric/1/test # ground-truth input
CUDA_LAUNCH_BLOCKING=1 python3 ./gen.py $gpu $model $truth

cp ./result/${model}_sample.txt ../stage2/result/
cd ../stage2
model=./model # directory of the 2nd-stage model
target_name=./result/${model}_sample # the generation result of the 1st stage
output_suffix=stage2
python3 -u ./gen2_gen.py $gpu $model $truth $target_name $output_suffix
python3 -u ./merge.py $target_name $output_suffix
# The generation results will be saved in ./stage2/result/${model}_sample_${output_suffix}_merge.txt
```



###4. Evaluation

Execute the following command for evaluation (or run `cd ./src/eval && bash ./eval.sh` directly): 

```shell
cd ./src/eval
gpu=0
data_name=cnn
python3 ./eval.py $gpu $data_name
```

You can change `result_list` in the script `eval.py` to specify the results you want to evaluate.



### Citation

Please kindly cite our paper if this paper and it is helpful.

```
@article{guan2022generating,
  title={Generating Coherent Narratives by Learning Dynamic and Discrete Entity States with a Contrastive Framework},
  author={Guan, Jian and Yang, Zhenyu and Zhang, Rongsheng and Hu, Zhipeng and Huang, Minlie},
  journal={arXiv preprint arXiv:2208.03985},
  year={2022}
}
```
