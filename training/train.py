# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
基于mlm的训练文件，已实现mlm_cls和mlm_margin两种
（此文件为train的最新版本）
"""

from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import random
import numpy as np
import torch
from transformers.models.roberta.modeling_roberta import RobertaModel
from torch.utils.data import DataLoader

from tqdm import tqdm
from data_utils import (MODEL_TYPES, MODEL_CLASSES,
                        load_config_tokenizer_model,
                        count_parameters,
                        load_loss_fn,
                        load_optimizer_scheduler,
                        #synthesize_QA_data_processed,
                        MLMDataset,
                        mlm_CollateFn,
                        mlm_step,
                        cls_step,
                        mlm_evaluate_dataset_acc)
import json

from CQA_evaluate import evaluate_CQA_datasets
from multiprocessing import Pool


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_cached_data",default=True,type=str, required=False)
    parser.add_argument('--ComQA_dev_path', default='data/commonsenseqa/dev_data.jsonl', type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_path', default='data/socialiqa/socialiqa-train-dev/dev.jsonl', type=str,
                        required=False)
    parser.add_argument('--SocialiQA_dev_label_path', default='data/socialiqa/socialiqa-train-dev/dev-labels.lst',
                        type=str,
                        required=False)
    parser.add_argument("--output_dir", default='model_save/Q_only_based_synthesize_filtered_0.8_with_hard_candidates/roberta-mlm-margin', type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default='Q_only_based_synthesize_QA_full_filtered_0.8_with_hard_candidates_roberta', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_type", default='roberta-mlm', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='roberta-large', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            MODEL_TYPES))
    parser.add_argument("--config_name", default=None, type=str,required=False,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,required=False,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    #暂时
    #修改loss函数
    parser.add_argument("--loss_type",default='mlm_margin') #mlm_CE/mlm_margin
    #parser.add_argument("--loss_type", default='mlm_CE')  # mlm_CE/mlm_margin
    parser.add_argument("--margin", default=1.0, type=float,
                        help="The margin for ranking loss")  # param for mlm_margin loss function
    parser.add_argument("--mask_type", default='Q_and_A')  # Q_only/Q_and_A
    parser.add_argument("--eval_skip_stopwords", action="store_true")

    #parser.add_argument("--candidates_num", default=3, type=int, required=False)
    parser.add_argument("--max_loop_times", default=20, type=int, required=False,help="构建候选选项的最多尝试尝试次数")
    parser.add_argument("--max_words_to_mask", default=6, type=int,
                        help="The maximum number of tokens to mask when computing scores")

    # 暂时
    #parser.add_argument('--save_steps', type=int, default=10,
    #                    help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2000,
                        help="Save checkpoint every X updates steps.")
    #暂时
    #2555
    #parser.add_argument('--seed', type=int, default=12345,
    #                    help="random seed for initialization")
    parser.add_argument('--seed', type=int, default=17293,
                        help="random seed for initialization")
    parser.add_argument("--is_cuda", default=True, type=bool, required=False)
    parser.add_argument("--gpu_id", default=0, type=int, required=False)

    parser.add_argument("--max_sequence_per_time", default=10, type=int,
                        help="The maximum number of sequences to feed into the model")
    parser.add_argument('--train_batch_size', default=1, type=int, required=False)
    parser.add_argument('--eval_batch_size', default=1, type=int, required=False)
    #parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
    #                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_seq_length", default=90, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Linear warmup over warmup proportion.")

    parser.add_argument("--with_sep", action='store_true', required=False)

    args = parser.parse_args()

    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer):
    cached_file = os.path.join(args.cache_dir, 'cached_{}_{}'.format(
        str(args.model_type),
        str(args.max_seq_length)))
    if args.load_cached_data and os.path.exists(cached_file):
        print("load data from cache path")
        tokenized_datas_dict = torch.load(cached_file)
        tokenized_train_datas=tokenized_datas_dict['train']
        tokenized_test_datas = tokenized_datas_dict['test']

        #暂时
        random.shuffle(tokenized_test_datas)
        tokenized_test_datas=tokenized_test_datas[:10000]
        #tokenized_test_datas = tokenized_test_datas[:5000]
    else:
        raise Exception("need to preprocess data first")

    print('max_words_to_mask is %s' % (args.max_words_to_mask))
    return MLMDataset(tokenized_train_datas, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask),\
           MLMDataset(tokenized_test_datas, tokenizer.pad_token_id, tokenizer.mask_token_id, args.max_words_to_mask)


def train(args, train_dataset, eval_dataset, model, tokenizer, loss_fn):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                  collate_fn=mlm_CollateFn)

    optimizer,scheduler=load_optimizer_scheduler(args, len(train_dataloader), (model,args.learning_rate))

    set_seed(args)
    global_step = 0
    print_loss=0.0
    model.zero_grad()
    curr_best = 0.0
    CE = torch.nn.CrossEntropyLoss(reduction='none')
    #loss_fct = torch.nn.MultiMarginLoss(margin=args.margin)
    #loss_fct=torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.num_train_epochs):
        pbar=tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(pbar):
            batch_input_ids, batch_input_mask, batch_input_labels, batch_label_ids = batch
            if len(batch_input_ids[0])==0:
                continue

            batch_choice_loss = mlm_step(batch_input_ids, batch_input_mask,
                                         batch_input_labels, args, model, CE)
            loss=loss_fn(batch_choice_loss, batch_label_ids.to(args.device))

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            print_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #model.zero_grad()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_description('loss:{}'.format(print_loss))
                print_loss=0.0

                if global_step % args.save_steps == 0:
                    eval_acc = mlm_evaluate_dataset_acc(args, model, eval_dataset)
                    model.eval()
                    with torch.no_grad():
                        evaluate_CQA_datasets(args, tokenizer, model)

                    if eval_acc > curr_best:
                        curr_best = eval_acc
                        # Save model checkpoint
                        output_dir = args.output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        #暂时
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        print("Saving model checkpoint to {}".format(output_dir))

                    model.train()

    eval_acc = mlm_evaluate_dataset_acc(args, model, eval_dataset)
    model.eval()
    with torch.no_grad():
        evaluate_CQA_datasets(args, tokenizer, model)
    if eval_acc > curr_best:
        # Save model checkpoint
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        #暂时
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        print("Saving model checkpoint to {}".format(output_dir))


def main():
    args=set_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() and args.is_cuda else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    config,tokenizer,model=load_config_tokenizer_model(args)
    model.to(args.device)


    loss_fn=load_loss_fn(args)

    count = count_parameters(model)
    print("parameters count {}".format(count))

    train_dataset, test_dataset = load_and_cache_examples(args, tokenizer)
    train(args, train_dataset, test_dataset, model, tokenizer, loss_fn)


if __name__ == "__main__":
    main()