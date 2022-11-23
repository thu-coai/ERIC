#!/usr/bin/env python
import os
from nltk import data
data.path.append(os.environ["HOME"]+"/nltk_data")

import argparse
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
np.random.seed(42)
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.manual_seed(42)
torch.cuda.manual_seed(42)
from callbacks import Seq2SeqLoggingCallback, get_checkpoint_callback, get_early_stopping_callback
from transformers import T5ForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import (
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_bleu,
    check_output_dir,
    flatten_list,
    freeze_embeds,
    freeze_params,
    label_smoothed_nll_loss,
    lmap,
    pickle_save,
    save_json,
    use_task_specific_params,
)


# need the parent dir module
# sys.path.insert(2, str(Path(__file__).resolve().parents[1]))
from lightning_base import BaseTransformer, add_generic_args, generic_train  # noqa


logger = logging.getLogger(__name__)


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss", "loss_lm", "loss_sen_plan", "loss_char"]
    metric_names = [] # ROUGE_KEYS
    default_val_metric = "loss"

    def __init__(self, hparams, **kwargs):
        if hparams.sortish_sampler and hparams.gpus > 1:
            hparams.replace_sampler_ddp = False
        elif hparams.max_tokens_per_batch is not None:
            if hparams.gpus > 1:
                raise NotImplementedError("Dynamic Batch size does not work for multi-gpu training")
            if hparams.sortish_sampler:
                raise ValueError("--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<p%d>"%k for k in range(100)]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model.model.decoder_encoder.resize_token_embeddings(len(self.tokenizer))

        # save_git_info(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        # self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size
        self.vocab_size = len(self.tokenizer)

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams.data_dir,
            max_source_length=self.hparams.max_source_length,
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams.max_target_length,
            "val": self.hparams.val_max_target_length,
            "test": self.hparams.test_max_target_length,
        }
        assert self.target_lens["train"] <= self.target_lens["val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens["test"], f"target_lens: {self.target_lens}"
        if self.hparams.freeze_embeds:
            freeze_embeds(self.model)
        if self.hparams.freeze_encoder:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        # self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams.num_workers
        self.tokenizer.decoder_start_token_id = self.decoder_start_token_id = 0# self.tokenizer.mask_token_id
        self.config.eos_token_id = self.tokenizer.eos_token_id# self.tokenizer.mask_token_id
        # print(self.tokenizer.eos_token_id)
        # exit()

        self.dataset_class = Seq2SeqDataset

        self.already_saved_batch = False
        self.eval_beams = self.model.config.num_beams if self.hparams.eval_beams is None else self.hparams.eval_beams
        if self.hparams.eval_max_gen_length is not None:
            self.eval_max_length = self.hparams.eval_max_gen_length
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric

        self.max_sen_num = self.config.max_sen_num = 16
        self.training_steps = 0
        num_param = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_param += torch.numel(param)
        print("="*10)
        print("# Parameters:", num_param)

    def save_readable_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """A debugging utility"""
        readable_batch = {
            k: self.tokenizer.batch_decode(v.tolist()) if "mask" not in k else v.shape for k, v in batch.items()
        }
        save_json(readable_batch, Path(self.output_dir) / "text_batch.json")
        save_json({k: v.tolist() for k, v in batch.items()}, Path(self.output_dir) / "tok_batch.json")

        self.already_saved_batch = True
        return readable_batch

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def gather_nd(self, x, indices):
        newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
        indices = indices.view(-1, indices.shape[-1]).tolist()
        out = torch.cat([x.__getitem__(tuple(i)) for i in indices]).reshape(newshape)
        return out

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id, decoder_start_token_id=self.tokenizer.decoder_start_token_id)
        if not self.already_saved_batch:  # This would be slightly better if it only happened on rank zero
            batch["decoder_input_ids"] = decoder_input_ids
            self.save_readable_batch(batch)
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False, output_hidden_states=True)
        lm_logits = outputs["logits"]
        if self.hparams.label_smoothing == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss_lm = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss_lm, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams.label_smoothing, ignore_index=pad_token_id
            )
        loss_sen_plan = torch.tensor(0.).to(loss_lm.device)
        loss_char = torch.tensor(0.).to(loss_lm.device)

        # [bsz, sen_num, hidden_size]
        plan_states_soft = outputs["sentence_representation"][0]
        plan_states_hard = outputs["sentence_representation"][1]
        # [bsz, sen_num]
        sen_person_id = outputs["sentence_representation"][2]
        plan_states_soft_negative = outputs["sentence_representation"][4]
        plan_states_hard_negative = outputs["sentence_representation"][5]
        sen_person_id_negative = outputs["sentence_representation"][6]

        # [bsz, sen_num, hidden_size]
        pred_attn_output = outputs["sentence_representation"][-3]
        # [bsz, sen_num, char_num]
        char_logits = outputs["sentence_representation"][-2]
        is_normal_char = outputs["sentence_representation"][-1]


        bsz, _, _ = lm_logits.size()
        # [batch_size, tgt_len]
        mask_pos = torch.eq(decoder_input_ids, self.tokenizer.mask_token_id).type_as(loss_lm)
        # [batch_size, tgt_len]
        mask_pos_sum0 = torch.cumsum(mask_pos, 1)
        mask_pos_sum = (mask_pos_sum0 - mask_pos + 1) * (1 - mask_pos)
        # [1, sen_num]
        sen_range = torch.arange(0, self.max_sen_num)[None, :].to(loss_lm.device)

        # [batch_size, sen_num, tgt_len]
        mask_pos_matrix = torch.eq(sen_range[:, :, None] + 1, mask_pos_sum[:, None, :]).type_as(loss_lm)

        # [bsz*sen_num, bsz*sen_num]
        loss_mask_plan = torch.eye(bsz*self.max_sen_num).to(loss_lm.device)

        # [batch_size, sen_num]
        bsz_zero = torch.zeros([bsz, 1]).type_as(loss_lm).to(loss_lm.device)
        masknum = torch.cat([bsz_zero, 
                    torch.gt(torch.sum(mask_pos_matrix, 2), 0)[:, 2:].type_as(loss_lm),
                    torch.eq(torch.sum(mask_pos, 1).int(), self.max_sen_num).type_as(loss_lm)[:, None]], 1).to(loss_lm.device)
        masknum_plan_origin = torch.cat([bsz_zero, masknum[:, 2:], bsz_zero], 1)
        masknum_plan = masknum_plan_origin * is_normal_char
        mm_mask_plan = (1 - torch.mm(masknum_plan.view(-1)[:, None], masknum_plan.view(-1)[None, :])) * (-1000.)

        char_num = 100
        # [bsz, sen_num]
        sen_person_id_transform = ((sen_person_id - self.tokenizer.mask_token_id) * is_normal_char).type_as(src_ids)
        sen_person_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # [bsz*sen_num]
        loss_char_all = sen_person_fct(char_logits.view(-1, char_num), sen_person_id_transform.view(-1))
        loss_char = torch.sum(loss_char_all * masknum_plan_origin.view(-1)) / (torch.sum(masknum_plan_origin) + 1e-20)
        # print("sen_person_id_transform:", sen_person_id_transform.size(), sen_person_id_transform[:2].int().cpu().numpy().tolist())
        # print("loss_char_all:", loss_char_all.size(), loss_char_all[:2].cpu().numpy().tolist())

        plan_states = plan_states_soft + plan_states_hard
        plan_states = plan_states / (torch.norm(plan_states, 2, dim=-1, keepdim=True) + 1e-20)
        pred_attn_output = pred_attn_output / (torch.norm(pred_attn_output, 2, dim=-1, keepdim=True) + 1e-20)
        plan_states, pred_attn_output = plan_states.view(-1, plan_states.shape[-1]), pred_attn_output.view(-1, plan_states.shape[-1])

        if plan_states_soft_negative is not None and plan_states_hard_negative is not None:
            is_normal_char_negative = torch.gt(sen_person_id_negative, 0).type_as(loss_lm)
            plan_states_negative = plan_states_soft_negative + plan_states_hard_negative
            plan_states_negative = plan_states_negative / (torch.norm(plan_states_negative, 2, dim=-1, keepdim=True) + 1e-20)
            plan_states = torch.cat([plan_states, plan_states_negative.view(-1, plan_states.shape[-1])], 0)
            loss_mask_plan = torch.cat([loss_mask_plan, torch.zeros([bsz*self.max_sen_num, bsz*self.max_sen_num]).to(loss_lm.device)], 1)
            masknum_plan_negative = masknum_plan_origin * is_normal_char_negative
            mm_mask_plan = torch.cat([mm_mask_plan, 
                (1 - torch.mm(masknum_plan.view(-1)[:, None], masknum_plan_negative.view(-1)[None, :])) * (-1000.)
            ], 1)

        temp = 0.1
        # [batch_size*sen_num, batch_size*sen_num]
        plan_logits = torch.mm(pred_attn_output, plan_states.transpose(0, 1)) + mm_mask_plan
        plan_score = F.softmax(plan_logits / temp, -1)
        plan_log_score = -torch.log(torch.sum(plan_score * loss_mask_plan, 1) + 1e-10)
        loss_sen_plan = torch.sum(plan_log_score * masknum_plan.view(-1)) / (torch.sum(masknum_plan) + 1e-20)

        self.training_steps += 1
        loss = loss_lm + loss_sen_plan + loss_char

        return (loss, loss_lm, loss_sen_plan, loss_char)


    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        logs["val_loss"] = logs["loss"]
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        # self._step(batch)
        # exit()
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        self.metrics[prefix].append(all_metrics)  # callback writes this to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_bleu(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()
        loss_tensors = self._step(batch)
        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        generated_ids = batch["input_ids"]
        # self.model.generate(
        #     batch["input_ids"],
        #     attention_mask=batch["attention_mask"],
        #     use_cache=True,
        #     decoder_start_token_id=self.decoder_start_token_id,
        #     num_beams=self.eval_beams,
        #     max_length=self.eval_max_length,
        # )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        bleu: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=target, **bleu)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams.sortish_sampler and type_path != "test" and type_path != "val":
            sampler = dataset.make_sortish_sampler(batch_size, distributed=self.hparams.gpus > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams.max_tokens_per_batch is not None and type_path != "test" and type_path != "val":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams.max_tokens_per_batch, distributed=self.hparams.gpus > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        add_generic_args(parser, root_dir)
        parser.add_argument(
            "--max_source_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--max_target_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--val_max_target_length",
            default=512,  # these defaults are optimized for CNNDM. For xsum, see README.md.
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        parser.add_argument(
            "--test_max_target_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument("--freeze_encoder", action="store_true")
        parser.add_argument("--freeze_embeds", action="store_true")
        parser.add_argument("--sortish_sampler", action="store_true", default=False)
        parser.add_argument("--overwrite_output_dir", action="store_true", default=False)
        parser.add_argument("--max_tokens_per_batch", type=int, default=None)
        parser.add_argument("--logger_name", type=str, choices=["default", "wandb", "wandb_shared"], default="default")
        parser.add_argument("--n_train", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_val", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument("--n_test", type=int, default=-1, required=False, help="# examples. -1 means use all.")
        parser.add_argument(
            "--task", type=str, default="summarization", required=False, help="# examples. -1 means use all."
        )
        parser.add_argument("--label_smoothing", type=float, default=0.0, required=False)
        parser.add_argument("--src_lang", type=str, default="", required=False)
        parser.add_argument("--tgt_lang", type=str, default="", required=False)
        parser.add_argument("--eval_beams", type=int, default=None, required=False)
        parser.add_argument(
            "--val_metric", type=str, default=None, required=False, choices=["bleu", "loss", None]
        )
        parser.add_argument("--eval_max_gen_length", type=int, default=200, help="never generate more than n tokens")
        parser.add_argument("--save_top_k", type=int, default=1, required=False, help="How many checkpoints to save")
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.",
        )
        return parser


def main(args, model=None) -> SummarizationModule:
    Path(args.output_dir).mkdir(exist_ok=True)
    check_output_dir(args, expected_items=3)

    if model is None:
        if "summarization" in args.task:
            model: SummarizationModule = SummarizationModule(args)
        else:
            model: SummarizationModule = TranslationModule(args)
    dataset = Path(args.data_dir).name
    if (
        args.logger_name == "default"
        or args.fast_dev_run
        or str(args.output_dir).startswith("/tmp")
        or str(args.output_dir).startswith("/var")
    ):
        logger = True  # don't pollute wandb logs unnecessarily
    elif args.logger_name == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        project = os.environ.get("WANDB_PROJECT", dataset)
        logger = WandbLogger(name=model.output_dir.name, project=project)

    elif args.logger_name == "wandb_shared":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(name=model.output_dir.name, project=f"hf_{dataset}")

    if args.early_stopping_patience >= 0:
        es_callback = get_early_stopping_callback(model.val_metric, args.early_stopping_patience)
    else:
        es_callback = False

    lower_is_better = args.val_metric == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        logging_callback=Seq2SeqLoggingCallback(),
        checkpoint_callback=get_checkpoint_callback(
            args.output_dir, model.val_metric, args.save_top_k, lower_is_better
        ),
        early_stopping_callback=es_callback,
        logger=logger,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")

    if not args.do_predict:
        return model

    model.hparams.test_checkpoint = ""
    checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "*.ckpt"), recursive=True)))
    print(os.path.join(args.output_dir, "*.ckpt"), checkpoints)
    if checkpoints:
        model.hparams.test_checkpoint = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    # trainer.checkpoint_callback.best_model_path = "./roc_base_sen/val_avg_loss=1.8971-step_count=60.ckpt"
    trainer.test()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SummarizationModule.add_model_specific_args(parser, os.getcwd())

    args = parser.parse_args()

    main(args)
